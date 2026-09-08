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

## Getting past 262K: YaRN to 1M needs two patches, not the documented flag

The Qwen model card documents two ways to enable YaRN. On this build the command line one
does not work, and the config file one hits a bug in transformers. Both are worth knowing
before you spend an evening on it.

**The documented flag silently reverts.** This is the card's recipe:

```
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve ... \
  --hf-overrides '{"text_config": {"rope_parameters": {... "rope_type": "yarn", "factor": 4.0 ...}}}' \
  --max-model-len 1000000
```

The override is accepted and reaches the engine (it shows up in `non-default args`), and
`max_model_len` resolves to 1,000,000. Then the multimodal processor loads, a second
`ModelConfig` is built that re-reads the raw config, and the window drops back to 262,144:

```
INFO [model.py:1965] Using max model len 1000000
[ERROR] `min_frames` is part of Qwen3VLVideoProcessorInitKwargs ...
INFO [model.py:1965] Using max model len 262144
```

No error is raised. The only symptom is those two lines, about twelve seconds apart, and
transformers continuing to log `rope_type='default'`. Setting
`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` does not change it.

**Patching config.json works, and then exposes a transformers bug.** Writing the card's
rope block into `text_config.rope_parameters` and bind mounting it over the model directory
gets `rope_type='yarn'` recognised, and then the engine dies:

```
AttributeError: 'Qwen4ExpConfig' object has no attribute 'max_position_embeddings'
```

The cause is in `transformers/modeling_rope_utils.py`. On any non-default rope type it does:

```python
self.rope_parameters.setdefault("original_max_position_embeddings", self.max_position_embeddings)
```

`setdefault` evaluates its default eagerly, so `self.max_position_embeddings` is read even
though the key is already present in the config. On this multimodal wrapper config that
attribute lives on `text_config`, not at the top level, so it raises. There are five such
accesses in that file: one in `standardize_rope_params` and the rest in
`_validate_yarn_rope_parameters` and its error path. Fixing only the first moves the failure
to line 928.

This is an upstream bug and it affects any multimodal model using YaRN, not just this one.
Adding `max_position_embeddings` or `original_max_position_embeddings` to config.json at any
level does **not** work around it: those keys are not stored as attributes on this config class.

**What works.** `patches/mk_yarn_config.py` writes the patched config.json, and
`patches/patch_rope_step1.py` plus `patches/patch_rope_step2.py` replace the five accesses
with a helper that falls back to `text_config`. Both are bind mounted over the image, and the
whole thing is behind `YARN=1` so the default path is untouched.

Result at `GPU_UTIL=0.90`, `MAX_NUM_SEQS=16`, factor 4.0:

```
Using max model len 1000000        (capability resolves to 1,048,576 = 262,144 x 4)
Model loading took 64.43 GiB       (vs 64.06 GiB native, so YaRN costs about 0.37 GiB)
GPU KV cache size: 2,852,941 tokens, Maximum concurrency for 1,000,000 tokens per request: 2.85x
```

**Whether you should.** Qwen's own guidance is not to leave this on: "All the notable
open-source frameworks implement static YaRN, which means the scaling factor remains constant
regardless of input length, potentially impacting performance on shorter texts. We advise
modifying the rope_parameters configuration only when processing long contexts is required."
Factor 4.0 applies to a 2K agent turn exactly as it does to an 800K document. If long context
is occasional, serve native and stand a second instance up when needed, or lower the factor to
match actual need (factor 2.0 for about 524K).

**Sizing tip.** Rather than guessing at `--gpu-memory-utilization`, read what the engine tells
you at startup:

```
Free memory on device (111.38/121.69 GiB) on startup.
Desired GPU memory utilization is (0.9, 109.52 GiB).
Actual usage is 68.68 GiB for consumed memory (weights + non-torch), 1.42 GiB for peak activation
Replace gpu_memory_utilization config with --kv-cache-memory=44161560064 (41.13 GiB) to fully utilize gpu memory
```

That names the exact byte figure for the largest KV pool the box will give you.

## Which NVFP4 checkpoint: NVIDIA's, not the community one

This playbook originally used `RadixArk/Qwen3.8-Flash-Next-NVFP4` because it was the first
NVFP4 build available. It has since been replaced with
**`nvidia/Qwen3.8-Flash-Next-NVFP4`**, and the reasoning is worth recording because it is
mostly not about speed.

**Provenance.** A quantization is a full reprocessing of every weight, so it is an
opportunity to alter behaviour, and altered weights cannot practically be audited.
Benchmarks do not catch trigger-conditioned behaviour: a checkpoint can score normally on
nine benchmarks and still carry something that surfaces only on specific inputs. Take the
model author's own build first, and if they do not publish the quant format you need, a
vendor build (NVIDIA, Red Hat). Qwen do not publish NVFP4, so NVIDIA is the right source.

**What else differed, measured rather than assumed:**

| | RadixArk | NVIDIA |
|---|---|---|
| Calibration | cnn_dailymail only, activations captured from live SGLang serving | cnn_dailymail **plus** Nemotron-Post-Training-v2 |
| Published accuracy | none | 9 benchmarks against the FP8 baseline |
| Documented runtime | SGLang only | vLLM |
| Files | 419, 135.3 GiB | 25, 123.6 GiB (one 50 GiB PLE/MTP blob) |
| Load time per node | 534 s | **447 s** |
| Weights per node | 64.43 GiB | **61.93 GiB** |
| KV pool at util 0.85, 1M window | 2,422,600 tok | **2,916,600 tok** |
| Headroom after load | 3.9 GiB | **6.0 GiB** |
| MTP speculative decoding | works | blocked, see below |

The calibration difference is the one that should matter for agentic work: cnn_dailymail is
news prose, so scales derived from it alone are calibrated on text that looks nothing like
code or tool calling. Nemotron-Post-Training-v2 is multi-turn instruction data.

NVIDIA's own published comparison against the FP8 baseline shows parity, NVFP4 ahead on five
and behind on four, all by small margins: GPQA 92.0/91.5, HLE 34.7/35.4, tau2-Telecom
90.8/90.1, MMMU Pro 77.1/78.3, SciCode 16.3/18.8, AA-LCR 71.9/74.1, IFBench 80.5/81.0,
Omniscience 28.1/27.6, Terminal-Bench 2.1 83.3/82.9. That independently corroborates the
36/40 vs 36/40 SWE result measured here on the RadixArk build.

### The PLE gate is not a RadixArk defect

Both builds ship FP8 PLE embedding tables with a single scale tensor, and vLLM's
`isinstance(quant_config, Fp8Config)` check rejects both. NVIDIA's own model card documents
this: serving requires vLLM commit `d4d703c` or later. Our image is older, which is why the
patch in this repository is needed; it reimplements what that commit does.

### MTP: NVIDIA's build cannot speculate on an older vLLM

NVIDIA block-quantized the MTP draft layer to FP8 (1,536 `weight_scale_inv` tensors in
`model-fp8-mtp-ple.safetensors`), and declares it in `quantized_layers` as
`mtp.layers.0.mlp.experts: {"quant_algo": "FP8_PB_WO", "group_size": 128}` under
`quant_algo: MIXED_PRECISION`. Loading fails with:

```
AttributeError: Layer mtp.layers.48.mlp.experts has no parameter 'w2_weight_scale_inv'
```

RadixArk excluded `mtp.*` from quantization entirely, leaving it bf16, which is why theirs
loaded. NVIDIA's card names the fix: vLLM PR #55513, unmerged at time of writing. That PR
adds block-FP8 MoE support to the MIXED_PRECISION path. It is ported to this build in
`patches/` and gated behind `MTP_FIX=1`.

Worth noting before treating this as a loss: on this hardware the NVIDIA build **without**
MTP reached 218.11 tok/s aggregate at c=16, against the RadixArk build **with** MTP=3 at
212.09. At higher concurrency batching dominates and speculation contributes little.

## Thermals and memory pressure on this chassis

Measured while benchmarking, because both affect what the numbers mean.

Cooling here is two Sparks side by side (deliberately not stacked, so neither breathes the
other's exhaust), one 120 mm USB fan per unit at full speed, 20 C room, both on a switched
outlet for remote power cycling.

**Under sustained GPU load** (94% utilization, hours into a two node run): GPU 58 C and
52 C, hottest board sensor 71 C and 68 C. The thermal slowdown counters are zero, lifetime,
on both boxes:

```
SW Thermal Slowdown : 0 us
HW Thermal Slowdown : 0 us
```

The counter that is *not* zero is power capping. So with this much airflow the GPU thermal
path is not the limit and the clocks sit where the power cap puts them. Past the point where
you are out of thermal slowdown entirely, more cooling cannot return clocks, because
temperature was not what was holding them.

**Linux does no thermal management on these boxes.** All seven thermal zones have exactly one
trip point each, and nothing bound to any of them:

```
thermal_zone0..6   trips=1   cdevs=0
trip_point_0_temp = 104C
trip_point_0_type = critical
```

`Processor` cooling devices exist but sit at `cur_state 0` of `max_state 3`, unattached to any
zone, so nothing drives them. There is no fan or pwm entry under hwmon and `nvidia-smi`
reports `fan.speed` as `N/A`. The fan curve is firmware side and not exposed.

**CPU temperature is regulated to a setpoint, not a function of clock.** Three runs, all 20
cores loaded for 180 s, same fan and same 20 C ambient:

| condition | clock | peak |
|---|---|---|
| `performance` governor | 2808 MHz | 92 C |
| `schedutil` governor | 2808 MHz | 92 C |
| `performance`, `scaling_max_freq` capped to 2.0 GHz | 2000 MHz | 92 C |

Idle was 37 C, cooldown 30 s after load ended was 55 to 56 C in every case. Cutting the clock
by 28% moved the peak by **zero**. The clock also sat at exactly `cpuinfo_max_freq` (2808 MHz)
at 37 C and at 92 C alike, so nothing was being scaled back on the way up.

That rules out both throttling and a clock/temperature relationship: the system takes the CPU
to about 92 C under sustained load and holds it there, and given less heat to move it simply
moves less. Practical consequence: **there is no OS side knob that lowers CPU temperature on
this hardware.** Not the governor (an all core load is 100% utilization, so `schedutil` ramps
to maximum exactly like `performance`), not a frequency cap, and the fans are not yours to
control. 92 C under load is normal here; the only trip point is at 104 C.

**Memory pressure shows up as CPU burn, not as swapping.** On GB10 the GPU, the CPU and the
page cache share one pool. Under pressure:

```
pswpout           0            <- nothing ever swapped
pgscan_direct     1,964,122    (node 1)    8,048,973 (node 2)
pgsteal_direct    1,913,054               5,752,670
/proc/pressure/memory total   11.8 s                33.0 s
```

That is direct reclaim: the kernel evicting page cache synchronously inside the allocating
thread, scanning millions of pages. It is pure CPU work and produces no disk writes, which is
why the swap counters stay at zero while load average climbs (about 25 with no user process
running, during one recovery). `vm.swappiness` does not help, because it only governs
anonymous pages; page cache eviction happens regardless. Watch `/proc/pressure/memory`, not
`free`, and bound the cache at the source with `scripts/cache-warden.py`.

## Files

```
patches/patch_ple.py                     the one-line gate fix, applied to a copy
scripts/serve-qwen38fn-nvfp4-vllm.sh     TP=2 launcher, NVFP4
scripts/serve-qwen38fn-fp8-vllm.sh       TP=2 launcher, FP8, for diffing
scripts/replay-bench.sh                  identical ladder + NIAH for either endpoint
scripts/verify-nvfp4.py                  checkpoint vs HuggingFace LFS sha256
scripts/memsample.sh                     MemAvailable curve tagged with engine log lines
scripts/cache-warden.py                  bounds page cache during bulk transfers
scripts/cputherm.sh                      CPU-bound thermal probe used for the table above
results/                                 raw JSON and the two memory curves
docs/JOURNEY.md                          what failed, in order, and why
```

## License

Apache-2.0. See `NOTICE` for upstream attribution.
