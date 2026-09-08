#!/usr/bin/env bash
# Bring up DeepSeek-V4-Flash-0731 (MXFP4 base) TP=2, tears down any other model first.
# Runtime: ghcr.io/anemll/dspark-vllm-gx10:0.1.1 with VLLM_USE_B12X_MOE=1 -- the b12x
# MXFP4 kernel native to sm_12x. (The stock vLLM image falls through to DeepGEMM MXFP4
# scale packing, which asserts "Unknown SF transformation" at layout.hpp -- no arch_major=12
# branch on GB10.) KV nvfp4_ds_mla, 1M ctx, util 0.80. Port 8891, fabric 192.168.100.x.
#
# SPEC_DECODE default OFF here on purpose. DSpark speculation gives ~41 tok/s single stream,
# but on this image spec-on wedges the engine under concurrent streams (>1). Off is stable
# under any load. For single-user interactive speed, run: SPEC_DECODE=on ./switch-deepseekv4.sh
#
# The NVFP4 transcode is not on the nodes; MXFP4 is DeepSeek's own publish.
cd "$(dirname "$0")" && source ./switch-common.sh
ENV="NAME=dsv4-anemll SPEC_DECODE=${SPEC_DECODE:-off} VLLM_USE_B12X_MOE=1"
say "########## SWITCH -> DeepSeek-V4-Flash-0731 MXFP4 (anemll b12x, spec=${SPEC_DECODE:-off}, 1M) ##########"
teardown
launch "$ENV" serve-anemll-mxfp4.sh 8891 && served_note deepseek-v4-flash-0731 8891
