#!/usr/bin/env bash
# Bring up GLM-5.3-Flash NVFP4 (RedHatAI checkpoint) TP=2, tears down any other model first.
# Serve script defaults: MTP-4, fp8_e4m3 KV, 244224 ctx with vision (MM_IMAGES=4).
# Set MM_IMAGES=0 for the full 262144 window without vision. Port 8890, fabric 192.168.101.x.
cd "$(dirname "$0")" && source ./switch-common.sh
ENV="MASTER_ADDR=192.168.101.14 NAME=glm53-nvfp4-vllm"
say "########## SWITCH -> GLM-5.3-Flash NVFP4 (RedHat, MTP-4, fp8 KV) ##########"
teardown
launch "$ENV" serve-glm53-nvfp4-vllm.sh 8890 && served_note glm-5.3-flash-nvfp4 8890
