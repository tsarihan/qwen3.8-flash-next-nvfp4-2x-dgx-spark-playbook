#!/usr/bin/env bash
set -u
source ~/matrix-lib.sh
say "########## PHASE 1: MTP ##########"
#        name          spec mtpk fix kv        yarn len      seqs qsafp8
run_cfg  mtp-off-bf16  off  3    0   bfloat16  1    1000000  16   0
run_cfg  mtp1-bf16     mtp  1    1   bfloat16  1    1000000  16   0
run_cfg  mtp3-bf16     mtp  3    1   bfloat16  1    1000000  16   0
say "########## PHASE 1 COMPLETE ##########"
