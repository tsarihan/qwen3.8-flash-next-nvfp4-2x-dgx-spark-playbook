# Manual model-switch scripts

Run from the orchestration host (OMEN) which has passwordless SSH to both Sparks.
No autoswitch: each brings up exactly one production model and tears down the others.
An accidental model referral can never evict production, because nothing switches
automatically -- you run the `.sh` by hand when you want to change models.

| script | model | image | port | notes |
|---|---|---|---|---|
| `switch-qwen38fn.sh` | Qwen3.8-Flash-Next NVFP4 (NVIDIA) | vLLM qwen38fn | 8892 | 1M YaRN, MTP k=1, bf16 KV, seqs 16, util 0.85 -- the overnight winner |
| `switch-glm53.sh` | GLM-5.3-Flash NVFP4 (RedHat) | tonyd2wild sm121-v8 | 8890 | MTP-4, fp8 KV; `MM_IMAGES=0` for full 262K without vision |
| `switch-deepseekv4.sh` | DeepSeek-V4-Flash-0731 MXFP4 | anemll/dspark-vllm-gx10:0.1.1 | 8891 | b12x MXFP4 kernel, nvfp4_ds_mla KV, 1M; **spec off by default** |

`switch-common.sh` holds the shared teardown (only the managed model containers are
removed -- qdrant/postgres/webui/ollama are never touched), the memory-reclaim wait,
and the health poll (`/health`, never `/v1/models`).

**DeepSeek gotcha, learned the hard way.** The stock vLLM image (`pinned-dev403`) cannot
serve the MXFP4 checkpoint on GB10: `process_weights_after_loading` routes the MoE scales
through DeepGEMM, which asserts `Unknown SF transformation` at `layout.hpp:60` because
DeepGEMM has no `arch_major=12` branch (DeepGEMM #372, vLLM #47436). The anemll image with
`VLLM_USE_B12X_MOE=1` uses the b12x MXFP4 kernel that is native to sm_12x and loads clean.
Also: on that image, DSpark speculation wedges the engine under concurrent streams, so the
switch script defaults `SPEC_DECODE=off` (stable under load; ~14 tok/s single stream).
`SPEC_DECODE=on ./switch-deepseekv4.sh` gets ~41 tok/s for single-user interactive only.
