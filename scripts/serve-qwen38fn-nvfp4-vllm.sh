#!/usr/bin/env bash
# Qwen3.8-Flash-Next-NVFP4 (RadixArk, modelopt W4A4) on 2x DGX Spark (GB10, sm_121), vLLM TP=2.
#   spark-1: NODE_RANK=0 ./serve-qwen38fn-nvfp4-vllm.sh
#   spark-2: NODE_RANK=1 ./serve-qwen38fn-nvfp4-vllm.sh
#
# Sources, and why each flag is here:
#  - vLLM recipe (recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next): image, expert-parallel,
#    reasoning/tool parsers, MTP=3. NOTE its configs are ALL discrete datacenter GPUs
#    (GB300/H200/H100/MI355X) -- none are unified memory, so its memory guidance
#    (max-num-seqs 256, VLLM_PLE_CPU_OFFLOAD for "80GB GPUs") does NOT transfer to GB10.
#  - GB10 field notes: NCCL channel cap, fabric-pinned host IPs. On this box the default
#    NCCL_MIN/MAX_NCHANNELS=64 hangs init.
#  - Our own GLM TP=2 experience on these exact nodes: --enforce-eager (FULL cudagraph
#    families wedge GB10), fabric NCCL env, poll /health never /v1/models.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/data/models/qwen3.8-flash-next-nvfp4}"
IMAGE="${IMAGE:-vllm/vllm-openai:qwen38-flash-next}"
NAME="${NAME:-qwen38fn-nvfp4-vllm}"
PORT="${PORT:-8892}"
NODE_RANK="${NODE_RANK:?set NODE_RANK=0 on spark-1, 1 on spark-2}"
MASTER_ADDR="${MASTER_ADDR:-192.168.101.14}"
MASTER_PORT="${MASTER_PORT:-25400}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"    # native; 1M needs YaRN
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"           # recipe says 256 -- that is sized for 80GB HBM
GPU_UTIL="${GPU_UTIL:-0.85}"                # 0.85 proven stable for GLM on these nodes
KV_DTYPE="${KV_DTYPE:-bfloat16}"            # MUST be bf16: the engine rejects fp8 with
# "NotImplementedError: Qwen3.8-Flash-Next QSA requires a BF16 main KV cache".
# Qwen Sparse Attention keeps its main KV in bf16, so this costs 2x the KV bytes of the
# fp8 cache GLM used -- the main memory consequence of this architecture on a 121GB node.
SPEC="${SPEC:-mtp}"                         # mtp | off
MTP_K="${MTP_K:-3}"                         # recipe value
EP="${EP:-1}"                               # --enable-expert-parallel
PLE_OFFLOAD="${PLE_OFFLOAD:-0}"             # measured both ways: unified memory != HBM
TOOL_PARSER="${TOOL_PARSER:-qwen3_xml}"
MOE_BACKEND="${MOE_BACKEND:-auto}"
YARN="${YARN:-0}"                       # 1 enables YaRN rope scaling for >native context
YARN_FACTOR="${YARN_FACTOR:-4.0}"       # 262144 * 4 = 1048576
NATIVE_LEN="${NATIVE_LEN:-262144}"      # model native window, the YaRN baseline
YARN_CFG="${YARN_CFG:-$HOME/patches/qwen38fn-nvfp4-config-yarn.json}"  # per-checkpoint patched config
MTP_FIX="${MTP_FIX:-0}"                 # 1 mounts the ported vLLM PR 55513 (block FP8 MTP)
QSA_FP8="${QSA_FP8:-0}"                 # 1 widens the QSA kv-dtype gate to allow fp8 (EXPERIMENTAL)
GEN_TEMP="${GEN_TEMP:-1.0}"             # Qwen model card, thinking mode
GEN_TOP_P="${GEN_TOP_P:-0.95}"
GEN_TOP_K="${GEN_TOP_K:-20}"
      # auto lets vLLM pick (FLASHINFER_CUTLASS here);
                                        # set e.g. marlin to test a different NVFP4 MoE lane
     # recipe says qwen3_xml, Qwen repo says qwen3_coder

SPEC_ARGS=""
[ "$SPEC" = "mtp" ] && SPEC_ARGS="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_K}}'"
EP_ARG=""; [ "$EP" = "1" ] && EP_ARG="--enable-expert-parallel"
MOE_ARG=""; [ "$MOE_BACKEND" != "auto" ] && MOE_ARG="--moe-backend $MOE_BACKEND"
GEN_ARG="--override-generation-config '{\"temperature\":${GEN_TEMP},\"top_p\":${GEN_TOP_P},\"top_k\":${GEN_TOP_K}}'"
CFG_MOUNT=""
MTP_MOUNT=""
QSA_MOUNT=""
[ "$QSA_FP8" = "1" ] && QSA_MOUNT="-v $HOME/patches/qsa.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/qsa.py:ro"
V=/usr/local/lib/python3.12/dist-packages/vllm
[ "$MTP_FIX" = "1" ] && MTP_MOUNT="-v $HOME/patches/modelopt.py:$V/model_executor/layers/quantization/modelopt.py:ro -v $HOME/patches/mtp.py:$V/models/qwen3_8_flash_next/nvidia/mtp.py:ro"
# YaRN needs BOTH: a config.json whose text_config.rope_parameters says yarn (the
# --hf-overrides route does not survive this build, the mm path re-reads the raw
# config), and a transformers fix, because modeling_rope_utils reads
# self.max_position_embeddings which lives on text_config for this mm wrapper config.
[ "$YARN" = "1" ] && CFG_MOUNT="-v $YARN_CFG:/model/config.json:ro -v $HOME/patches/modeling_rope_utils.py:/usr/local/lib/python3.12/dist-packages/transformers/modeling_rope_utils.py:ro"
ROPE_ARG=""
# This model uses transformers 5.x rope_parameters (not rope_scaling) and is mRoPE, so the
# override must live under text_config and preserve mrope_section / mrope_interleaved /
# partial_rotary_factor. vLLM here takes it via --hf-overrides; --rope-scaling does not exist.
[ "$YARN" = "1" ] && ROPE_ARG="--hf-overrides '{\"text_config\":{\"rope_parameters\":{\"rope_type\":\"yarn\",\"factor\":${YARN_FACTOR},\"original_max_position_embeddings\":${NATIVE_LEN},\"mrope_interleaved\":true,\"mrope_section\":[11,11,10],\"partial_rotary_factor\":0.25,\"rope_theta\":10000000}}}'"
HEADLESS=""; [ "$NODE_RANK" = "1" ] && HEADLESS="--headless"
case "$NODE_RANK" in
  0) HOST_IP=192.168.101.14 ;;
  1) HOST_IP=192.168.101.15 ;;
esac

docker rm -f "$NAME" >/dev/null 2>&1 || true

CMD="vllm serve /model \
  --served-model-name qwen3.8-flash-next-nvfp4 \
  --host 0.0.0.0 --port $PORT \
  --trust-remote-code \
  --tensor-parallel-size 2 $EP_ARG \
  --gpu-memory-utilization $GPU_UTIL \
  --max-model-len $MAX_MODEL_LEN \
  --max-num-seqs $MAX_NUM_SEQS \
  --kv-cache-dtype $KV_DTYPE \
  --enforce-eager \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser $TOOL_PARSER \
  --distributed-executor-backend mp \
  --nnodes 2 --node-rank $NODE_RANK \
  --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
  $MOE_ARG $ROPE_ARG $GEN_ARG \
  $HEADLESS $SPEC_ARGS"

docker run -d --name "$NAME" --entrypoint bash \
  --gpus all --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_DIR":/model:ro \
  -v "$HOME/patches/ple_layer.py":/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro \
  $CFG_MOUNT $MTP_MOUNT $QSA_MOUNT \
  -e VLLM_HOST_IP=$HOST_IP \
  -e VLLM_PLE_CPU_OFFLOAD=$PLE_OFFLOAD \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=$YARN \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_QWEN38FN_PLE_FP8=1 \
  -e QSA_FP8_KV=$QSA_FP8 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=roceP2p1s0f0 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ADDR_RANGE=192.168.101.0/24 \
  -e NCCL_SOCKET_IFNAME=enP2p1s0f0np0 -e GLOO_SOCKET_IFNAME=enP2p1s0f0np0 \
  -e TP_SOCKET_IFNAME=enP2p1s0f0np0 \
  -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$IMAGE" -c "$CMD"

echo "[$NAME] rank=$NODE_RANK tp=2 ep=$EP moe=$MOE_BACKEND mtpfix=$MTP_FIX yarn=$YARN/$YARN_FACTOR kv=$KV_DTYPE spec=$SPEC/$MTP_K seqs=$MAX_NUM_SEQS ple=$PLE_OFFLOAD parser=$TOOL_PARSER launched."
[ "$NODE_RANK" = "0" ] && echo "Health: http://0.0.0.0:${PORT}/health   (poll /health, NOT /v1/models)"
