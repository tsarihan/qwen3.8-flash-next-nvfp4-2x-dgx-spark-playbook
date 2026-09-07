# Journey: what failed, in order

Chronological, including the wrong turns, because the wrong turns are most of the cost.

## 0. Getting the weights onto two nodes at all

The NVFP4 build is 135,253,622,894 bytes across 419 files. Node 1 had 128.27 GiB free and
the payload is 125.97 GiB, a 2.3 GiB margin, which would have left the root filesystem at
99.94%. Not acceptable on a compute node, so 74.85 GiB was freed first by relocating an
unrelated checkpoint after verifying all 63 of its files matched by sha256 at source and
destination.

Lesson that cost a rebuild once already: compare **keyed by path**, not by diffing sorted
output. Dotfiles sort differently on different hosts and an order sensitive `diff` will
report a mismatch on byte identical data.

The second copy went node to node over the 200 Gbps RoCE link rather than pulling from
HuggingFace twice. 6m50s at about 314 MB/s, disk bound rather than network bound. Both
copies were then verified against HuggingFace's own LFS sha256, not merely against each
other, which is what catches a resumed `.incomplete` file that happens to be the right size.

## 1. Page cache eats the GPU's memory

On GB10 the CPU, the page cache and the GPU all draw on one 121 GiB pool. Writing 135 GB
through the page cache drives `MemAvailable` toward zero even with nothing loaded.

`scripts/cache-warden.py` walks the destination every 20 s and calls
`posix_fadvise(POSIX_FADV_DONTNEED)` on any file that has stopped growing. Through the
transfer it released 84 GB and held `MemAvailable` at about 115 GB, costing nothing
measurable at 314 MB/s.

One false lesson recorded and then retracted: at one point loading appeared to speed up 9x
immediately after "stopping" the warden. The warden had not stopped. `pkill -f cache-warden.py`
matched its own shell command line and killed that instead. The speedup was a second loading
pass starting with a fresh progress bar. Use a pattern that cannot self match, such as
`pkill -f "[c]ache-warden"`, and verify with `pgrep` afterwards.

## 2. The model card says SGLang, and it is not wrong

`RadixArk/Qwen3.8-Flash-Next-NVFP4` documents only an SGLang launch line and states it was
validated on GB300 and B300. Given an earlier finding on this cluster that SGLang had no
working DSA decode kernel for sm_121, the vLLM route was tried first.

vLLM does auto-detect the checkpoint. A cheap probe before moving 135 GB:

```
DETECTED QUANT: modelopt_fp4
ARCH: ['Qwen4ExpForConditionalGeneration']
```

That looked like a green light. It was not sufficient.

## 3. Rank 1 dies 144 s in, rank 0 keeps loading, unaware

```
ValueError: There is no module or parameter named 'ngram_embedding.weight_scale'
in Qwen3_8FlashNextNGramEmbedding. The available parameters belonging to
ngram_embedding (VocabParallelEmbedding) are: {'ngram_embedding.weight'}
```

Worth noting for anyone debugging a multi node launch: the failure surfaced only on rank 1.
Rank 0 continued loading shards for minutes afterwards with no indication anything was wrong.
Check both ranks.

Diagnosis, in the order it actually went:

1. Found 128 `ngram_embedding.shard_N.weight` tensors plus exactly **one**
   `ngram_embedding.weight_scale` in the index. vLLM's mapper consolidates the shards into a
   single `ngram_embedding.weight` and has nowhere to put the aggregate scale.
2. **Checked the FP8 checkpoint for the same structure.** It has it: 128 shards plus one
   scale, and it loads. That is the observation that redirected the search from the tensors
   to the config path.
3. **Checked whether the scale was vestigial** before considering dropping it. It is not.
   The shards are `F8_E4M3` `[2500012, 160]` and the scale is a BF16 scalar. Dropping it
   would have loaded FP8 bytes into a BF16 embedding and produced silent garbage rather than
   an error. Offline dequantisation is not an option either: about 51 GB of fp8 becomes about
   102 GB of bf16.
4. Read the config difference. FP8 declares `quant_method: fp8` in `config.json` and does not
   exclude PLE, so PLE is FP8 and vLLM builds an FP8 capable embedding. NVFP4 declares
   `quant_method: modelopt` with `ignore` containing `*.ple.*`, so vLLM treats PLE as
   unquantized while the bytes on disk are still FP8.
5. Found `modelopt_mixed` in the quantization registry and briefly thought it was the answer.
   `ModelOptMixedPrecisionConfig` exists exactly for "FP8 for dense layers and NVFP4 for MoE
   experts". **It fails the same check**, because it is not an `Fp8Config` either. Declaring
   `MIXED_PRECISION` alone fixes nothing.
6. Found the actual gate, one line in `ple_layer.py`, reproduced in the README.

## 4. The fix, and why it is safe

`Qwen3_8FlashNextPLEFp8EmbeddingMethod` takes no constructor arguments and registers its own
FP8 weight and BF16 `PerTensorScaleParameter`. It has no dependency on the quant config that
gates it. So returning it for a ModelOpt config is a widening of the gate, not a
reinterpretation of the checkpoint. Kept behind `VLLM_QWEN38FN_PLE_FP8=1` so it is opt-in and
reversible, and bind-mounted over the image rather than baked in.

Both ranks then load, 64.06 GiB per node, and the engine reports:

```
GPU KV cache size: 2,228,932 tokens, Maximum concurrency for 262,144 tokens per request: 8.50x
init engine (profile, create kv cache, warmup model) took 118.96 s
```

## 5. What the comparison actually showed

The expectation going in was that NVFP4 would trade quality and long-context recall for
speed. It did not. It won the entire concurrency ladder, won long-context prefill, held 5/5
needles at every depth up to 245K, and delivered 3.74x the KV pool.

The one place the first data point misled: TTFT at c=1 is worse for NVFP4 (1.400 s vs
0.715 s), and reading that single point suggested a latency tradeoff. Across the rest of the
ladder NVFP4's TTFT is better, decisively so at c=64 (0.989 s vs 3.228 s). One point is not a
trend.

The KV result also needed re-reading. 3.74x more KV cache does **not** mean longer context,
because both lanes stop at `max_model_len` 262,144. It means more simultaneous long sessions:
8.50 full length requests against 2.27.

## 6. Known gap in this data

The FP8 ladder was measured before `MAX_NUM_SEQS` was pinned, and the value used could not be
recovered afterwards because the containers had been removed and the logs not retained. The
NVFP4 ladder used a pinned 64. The FP8 ladder rising smoothly to c=64 rules out a low
scheduler cap, and the deltas are much larger than a cap would explain, but the FP8 lane is
being re-run at the identical pinned configuration before these throughput numbers should be
treated as final. KV cache size and NIAH do not depend on `max_num_seqs` and are unaffected.

Record the launch configuration next to the results. Recovering it afterwards from a removed
container is not possible.
