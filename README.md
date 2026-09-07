# Qwen3.8-Flash-Next-NVFP4 on 2x DGX Spark (GB10, sm_121)

Serving `RadixArk/Qwen3.8-Flash-Next-NVFP4` with vLLM TP=2 across two DGX Sparks, and
measured like-for-like against the FP8 build of the same model on the same rig.

**The checkpoint does not load in stock vLLM.** The fix is one line, and it is the main
reason this repository exists. Everything else is measurement.

## Headline

| | FP8 | NVFP4 | delta |
|---|---|---|---|
| Weights per node | 88.07 GiB | **64.06 GiB** | -24.01 GiB |
| GPU KV cache | 607,890 tok | **2,228,932 tok** | **3.67x** |
| Max concurrency @ 262,144 tok/req | 2.32x | **8.50x** | 3.67x |
| Decode @ c=1 | 24.42 tok/s | **32.16 tok/s** | +32% |
| Aggregate @ c=64 | 176.75 tok/s | **257.59 tok/s** | +46% |
| TTFT @ 245K prompt | 118.8 s | **96.9 s** | -18% |
| Decode @ 245K prompt | 29.78 tok/s | **49.40 tok/s** | +66% |
| NIAH 5 needles, 4K to 245K | 5/5 at every depth | 5/5 at every depth | no loss |

W4A4 quantization did not degrade long-context retrieval: 5/5 needles at 4,096 / 32,768 /
131,072 / 200,000 / 245,000 on both lanes.

The KV pool is the real story. NVFP4 does **not** raise maximum context: both lanes cap at
`max_model_len` 262,144. What the 24 GiB per node of freed weights buys is **concurrency at
long context**, 8.50 simultaneous full length requests against FP8's 2.27.

## The blocker: `ngram_embedding.weight_scale`

Stock vLLM dies during weight load:

```
ValueError: There is no module or parameter named 'ngram_embedding.weight_scale'
in Qwen3_8FlashNextNGramEmbedding. The available parameters belonging to
ngram_embedding (VocabParallelEmbedding) are: {'ngram_embedding.weight'}
```

The checkpoint is genuinely mixed precision:

* NVFP4 W4A4 routed experts (group size 16)
* **FP8 PLE embedding tables**: 128 shards of `F8_E4M3` `[2500012, 160]` in
  `model-plefp8-*.safetensors`, plus exactly one BF16 scalar `weight_scale`
* BF16 attention, QSA, Gated DeltaNet, mHC

Its `quantization_config.ignore` excludes `*.ple.*` from NVFP4, so vLLM treats PLE as
unquantized and builds a plain `VocabParallelEmbedding` with no `weight_scale` slot, while
the bytes on disk are still FP8. The gate is one line in
`vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py`:

```python
def _get_ple_embedding_quant_method(quant_config, prefix):
    """Select global-scale FP8 only for quantized PLE checkpoint shards."""
    if not isinstance(quant_config, Fp8Config):   # NVFP4 config lands here
        return None                               # -> plain embedding -> ValueError
    ...
    return Qwen3_8FlashNextPLEFp8EmbeddingMethod()
```

### Three things checked before patching, each of which ruled out a wrong fix

1. **The FP8 checkpoint has the identical layout**, 128 shards plus one scale, and loads
   fine. So the tensors are not the problem, the quant config path is.
2. **The scale is not vestigial.** The shards really are `F8_E4M3`, so dropping the stray
   scale to get past the error would have silently corrupted the embeddings. Dequantising
   offline is not viable either: ~51 GB of fp8 becomes ~102 GB of bf16.
3. **`modelopt_mixed` does not fix it.** `ModelOptMixedPrecisionConfig` exists precisely for
   NVFP4 experts plus FP8 dense layers, but it is not an `Fp8Config` either, so it fails the
   same `isinstance` check. Declaring `MIXED_PRECISION` alone changes nothing.

### The fix

Widen the gate behind an opt-in environment variable and bind-mount the patched file over
the image. Safe because `Qwen3_8FlashNextPLEFp8EmbeddingMethod` takes no constructor
arguments and registers its own FP8 weight plus BF16 `PerTensorScaleParameter`.

```bash
python3 patches/patch_ple.py          # edits a copy of ple_layer.py in place
# then, in docker run:
#   -v $HOME/patches/ple_layer.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro
#   -e VLLM_QWEN38FN_PLE_FP8=1
```

Both ranks then load at **64.06 GiB per node** (139 s on rank 1, 567 s on rank 0).

This substantiates rather than contradicts the model card's "serve with SGLang" guidance:
SGLang handles the mixed FP8 PLE that vLLM's ModelOpt path does not.

## Kernel init allocates ~88 GiB before any weight is read

From `results/memcurve-nvfp4-run1-failed.log`, sampling `MemAvailable` every 2 s and tagging
each sample with the newest engine log line:

```
T+0051s available=111.7GiB | SymmMemCommunicator: Device capability 12.1 not supported
T+0053s available= 70.6GiB | Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend      -41.1 GiB
T+0055s available= 30.9GiB | Using FlashAttention version 2                    -39.7 GiB
T+0057s available= 23.1GiB | Loading safetensors checkpoint shards: 0/206      -7.8 GiB
```

Memory then holds flat through weight loading, which indicates the weights are filling an
already reserved pool.

**Settled by experiment: this is the reserved pool, not kernel workspace.** Re-running the
identical configuration with only `--gpu-memory-utilization` changed:

| | 0.85 | 0.70 |
|---|---|---|
| before init | 111.7 GiB | 111.7 GiB |
| at first weight shard | 23.1 GiB | 47.4 GiB |
| consumed before any weight | **88.6 GiB** | **64.3 GiB** |
| weights loaded | 64.06 GiB | 64.06 GiB |
| GPU KV cache | **2,228,932 tok** | **1,001,815 tok** |
| concurrency @ 262,144 | 8.50x | 3.82x |

Lowering utilization by 0.15 removed 24.3 GiB from the pre-weight allocation, while weight
loading was byte-identical at 64.06 GiB in both runs. The reserved memory is not consumed,
it becomes the KV pool: 24.3 GiB of pool difference against 1,227,117 tokens of KV
difference is about 21 KB per token, a sane bf16 figure for this architecture.

So the order of operations is: reserve the pool, load weights into it, and whatever remains
becomes KV cache. Flat memory during weight loading is the signature.

One consequence worth knowing: `weight_utils.py` reads `psutil.virtual_memory().available`
**after** the pool is reserved, so it measures a system vLLM itself just depleted. That is
why it can conclude the checkpoint exceeds 90% of available RAM and disable auto-prefetch on
exactly the large models where prefetch would help most.

Practical rule on a 121 GB unified node: pick utilization by how much KV you need. Trying to
shrink the initial allocation is aiming at the wrong thing, because the allocation is the
point.

### The MoE backend is selectable, but there is no usable alternative on GB10

`--moe-backend` is a real flag (an `EngineArgs` field, `arg_utils.py:499`, registered at
`:1600`). There is **no** `VLLM_FLASHINFER_MOE_BACKEND` environment variable. Candidates
printed at boot: `FLASHINFER_TRTLLM`, `FLASHINFER_CUTEDSL`, `FLASHINFER_CUTEDSL_BATCHED`,
`FLASHINFER_CUTLASS`, `VLLM_CUTLASS`, `MARLIN`, `HUMMING`, `EMULATION`.

Both plausible alternatives were tried on this hardware and both fail:

`--moe-backend marlin` loads the NVFP4 experts (it warns "Your GPU does not have native
support for FP4 computation ... Weight-only FP4 compression will be used leveraging the
Marlin kernel"), then dies at engine init:

```
ValueError: moe_backend='marlin' is not supported for unquantized MoE.
Expected one of ['triton', 'batched_triton', 'flashinfer_trtllm', 'flashinfer_cutlass', 'aiter'].
```

`--moe-backend flashinfer_trtllm` fails earlier:

```
ValueError: NvFp4 MoE backend 'FLASHINFER_TRTLLM' does not support the deployment
configuration since kernel does not support current device cuda.
```

The reason is the mixed-precision checkpoint again. `--moe-backend` is a single global
setting, and this model has **both** NVFP4 routed experts and unquantized MoE modules (the
ones `ignore` excludes, `*.mlp.shared_expert.*` and `*.mlp.gate*`). The chosen backend has
to be valid for both. Intersecting the NVFP4 candidate list with the unquantized MoE list
leaves only `flashinfer_trtllm`, whose kernel does not support sm_121, and
`flashinfer_cutlass`. So **`FLASHINFER_CUTLASS` is the only workable option**, and vLLM's
automatic selection was already picking it.

The practical consequence: the large pre-weight allocation above is **not** avoidable by
changing the MoE backend on this hardware. If it is to be reduced, it has to be somewhere
else. Testing whether it scales with `--gpu-memory-utilization` remains the open experiment.

During the marlin attempt the pre-weight drop was smaller (111.7 to 47.7 GiB, against 111.7
to 23.1 GiB for cutlass), which suggests backend choice does move the allocation. That run
never reached KV allocation, so the difference cannot be converted into usable cache and is
recorded here as an observation only, not as a benefit.

## Concurrency ladder

`MAX_NUM_SEQS=64`, `max_model_len` 262,144, 512 output tokens, 0 failures on both lanes.

| conc | FP8 agg | NVFP4 agg | delta | FP8 per-stream | NVFP4 per-stream | FP8 TTFT | NVFP4 TTFT |
|---|---|---|---|---|---|---|---|
| 1 | 23.02 | 29.61 | +28.6% | 24.42 | 32.16 | 1.320 | 1.400 |
| 2 | 42.08 | 55.49 | +31.9% | 22.59 | 29.35 | 0.521 | 0.651 |
| 4 | 61.90 | 92.42 | +49.3% | 16.70 | 24.86 | 1.135 | 0.877 |
| 8 | 94.28 | 143.53 | +52.2% | 12.96 | 19.93 | 1.455 | 1.188 |
| 16 | 145.84 | 212.09 | +45.4% | 10.18 | 14.86 | 2.459 | 2.129 |
| 32 | 152.66 | 231.30 | +51.5% | 9.38 | 13.76 | 3.770 | 1.330 |
| 64 | 176.75 | **257.59** | +45.7% | 8.46 | 12.38 | 3.246 | 0.989 |

NVFP4's TTFT is worse only at c=1, and better at every other point. The c=1 figure looks
like a cold start artifact on the first request after warmup.

### On the FP8 column: the confound was checked, and it was immaterial

The first FP8 ladder was measured before `MAX_NUM_SEQS` was pinned, and the value used could
not be recovered afterwards. Rather than reason about it, the whole FP8 lane was re-run at
the identical pinned configuration. The two agree closely:

| conc | FP8 original | FP8 pinned |
|---|---|---|
| 16 | 140.62 | 145.84 |
| 32 | 158.02 | 152.66 |
| 64 | 177.46 | 176.75 |

So the original numbers were sound and the NVFP4 advantage was never an artifact of the
scheduler cap. Every FP8 figure in this document is now from the pinned run.

KV cache size also proved independent of `max_num_seqs`, as expected: 595,137 tokens on the
original run and 607,890 on the pinned one, a 2.1% difference.

## Needle in a haystack, 5 needles

| context | FP8 TTFT | NVFP4 TTFT | FP8 decode | NVFP4 decode | needles |
|---|---|---|---|---|---|
| 4,096 | 3.6 s | **3.1 s** | 37.24 | **50.95** | 5/5 both |
| 32,768 | 15.5 s | **12.1 s** | 37.20 | **41.51** | 5/5 both |
| 131,072 | 63.4 s | **50.6 s** | **43.54** | 42.74 | 5/5 both |
| 200,000 | 95.7 s | **77.3 s** | 38.15 | **49.64** | 5/5 both |
| 245,000 | 118.8 s | **96.9 s** | 29.78 | **49.40** | 5/5 both |

258,000 fails on FP8 because the rendered prompt reaches 261,583 tokens, which leaves no room
for generation inside a 262,144 window. The practical ceiling on both lanes is about 248K.

## SWE-bench Pro

40 enterprise-flavoured instances (10 per language, weighted toward web / FastAPI / LLM use /
multiuser / crypto / compliance), run from a separate host so the agent harness and its
container images never compete for the sparks' unified memory.

| model | resolved | score | wall clock |
|---|---|---|---|
| **Qwen3.8-Flash-Next-NVFP4** | **36/40** | **90.0%** | **2h30m** |
| Qwen3.8-Flash-Next-FP8 | 36/40 | 90.0% | 3h21m |
| GLM-5.3-Flash NVFP4 (z.ai sampling server side) | 36/40 | 90.0% | ~13h |
| Qwen3.8-27B fast2 (RTX 5090) | 30/40 | 75.0% | |

NVFP4 ties FP8 exactly on score while finishing 25% faster, which tracks the throughput
result rather than contradicting it.

The tie is not an artifact of which instances happened to pass. The two lanes fail on
different work: 2 instances failed in both, 2 only under NVFP4, 2 only under FP8. Both
struggle on the same repository (flipt, Go), which accounts for 3 of NVFP4's 4 failures and
2 of FP8's.

One NVFP4 trajectory produced a degenerate prediction: the harness captured
`cat: patch.txt: No such file or directory` as the patch, because the agent's final step
read a file it never wrote. That instance could not resolve regardless of model quality.
It is `flipt-0fd09def`, and **FP8 failed the same instance**, so it costs NVFP4 nothing in
this comparison. Raw per instance results for both lanes are in `results/`.

## Environment

| | |
|---|---|
| Hardware | 2x NVIDIA DGX Spark (GB10, sm_121), 121 GiB unified LPDDR5X each |
| Interconnect | ConnectX-7 RoCE, 200 Gbps, `enP2p1s0f0np0` |
| Driver / CUDA | 580.173.02 / 13.0 |
| Image | `vllm/vllm-openai:qwen38-flash-next` |
| vLLM | `0.1.dev20073+g8e685d198` (not an upstream tag) |
| Model | `RadixArk/Qwen3.8-Flash-Next-NVFP4`, 135,253,622,894 bytes, 419 files |

## Reproducing

```bash
# 1. verify the checkpoint against HuggingFace LFS sha256 before serving 135 GB
python3 scripts/verify-nvfp4.py

# 2. patch the PLE gate, then bind-mount it (see scripts/serve-qwen38fn-nvfp4-vllm.sh)
python3 patches/patch_ple.py

# 3. serve, rank 0 on the API head
NODE_RANK=0 MAX_NUM_SEQS=64 MAX_MODEL_LEN=262144 GPU_UTIL=0.85 \
  KV_DTYPE=bfloat16 PORT=8892 ./scripts/serve-qwen38fn-nvfp4-vllm.sh
NODE_RANK=1 ... ./scripts/serve-qwen38fn-nvfp4-vllm.sh   # on the second node

# 4. replay the identical measurement set used for both lanes
./scripts/replay-bench.sh qwen38fn-nvfp4 8892 qwen3.8-flash-next-nvfp4
```

`scripts/serve-qwen38fn-fp8-vllm.sh` is included unchanged so the two launchers can be
diffed. They differ only in model path, container name, port and served model name: every
engine flag is identical, which is what makes the comparison weights-only.

## Notes that cost time

* The image `ENTRYPOINT` is already `vllm serve`, so anything passed after the image name is
  appended to it. Use `--entrypoint bash` and pass a full command string, or unrelated flags
  fail with confusing JSON parse errors.
* `--enforce-eager` is required. The full cudagraph modes wedge GB10.
* `--kv-cache-dtype bfloat16` is required. The stock QSA kernels declare
  `supported_kv_cache_dtypes = ["auto", "bfloat16"]` (`qsa.py:70`) and raise at `:107` and
  `:186`. This costs roughly twice the KV bytes of an fp8 cache and is the largest single
  memory line item in the configuration.
* `NCCL_MIN_NCHANNELS=4` and `NCCL_MAX_NCHANNELS=4`. The default of 64 hangs channel init.
* `vllm serve --help` exits 1 in a container started without `--gpus`, so an empty grep of
  its output is not evidence that a flag is absent.
* Poll `/health`, never `/v1/models`, to decide readiness.
* On GB10 the page cache and the GPU share one pool, so a 135 GB transfer will eat it.
  `scripts/cache-warden.py` calls `posix_fadvise(POSIX_FADV_DONTNEED)` on files that have
  stopped growing. During the transfer it released 84 GB and held `MemAvailable` at about
  115 GB, at 314 MB/s. Stop it before serving so it does not add noise.

## Files

```
patches/patch_ple.py                     the one-line gate fix, applied to a copy
scripts/serve-qwen38fn-nvfp4-vllm.sh     TP=2 launcher, NVFP4
scripts/serve-qwen38fn-fp8-vllm.sh       TP=2 launcher, FP8, for diffing
scripts/replay-bench.sh                  identical ladder + NIAH for either endpoint
scripts/verify-nvfp4.py                  checkpoint vs HuggingFace LFS sha256
scripts/memsample.sh                     MemAvailable curve tagged with engine log lines
scripts/cache-warden.py                  bounds page cache during bulk transfers
results/                                 raw JSON and the two memory curves
docs/JOURNEY.md                          what failed, in order, and why
```

## License

Apache-2.0. See `NOTICE` for upstream attribution.
