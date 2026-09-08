#!/usr/bin/env bash
# Bring up Qwen3.8-Flash-Next NVFP4 (NVIDIA checkpoint) TP=2, tears down any other model first.
# Config = the overnight matrix winner: 1M YaRN, MTP k=1, bf16 KV, seqs 16, util 0.85.
# Port 8892, fabric 192.168.101.x. Run this by hand; no autoswitch.
cd "$(dirname "$0")" && source ./switch-common.sh
ENV="MAX_NUM_SEQS=16 YARN=1 YARN_FACTOR=4.0 MAX_MODEL_LEN=1000000 GPU_UTIL=0.85 \
KV_DTYPE=bfloat16 PORT=8892 SPEC=mtp MTP_K=1 MTP_FIX=1 QSA_FP8=0 \
MODEL_DIR=/data/models/qwen38fn-nvfp4-nvidia \
YARN_CFG=\$HOME/patches/qwen38fn-nvidia-config-yarn.json NAME=qwen38fn-nvidia"
say "########## SWITCH -> Qwen3.8-Flash-Next NVFP4 (NVIDIA, 1M, MTP-1) ##########"
teardown
launch "$ENV" serve-qwen38fn-nvfp4-vllm.sh 8892 && served_note qwen3.8-flash-next-nvfp4 8892
