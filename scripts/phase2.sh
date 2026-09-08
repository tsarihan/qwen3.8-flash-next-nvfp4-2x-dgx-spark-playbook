#!/usr/bin/env bash
set -u
source ~/matrix-lib.sh
BEST=$(python3 - "$R/summary.txt" <<'PY'
import json,sys,os
best=("mtp-off-bf16","off","3","0",-1.0)
p=sys.argv[1]
if os.path.exists(p):
    for line in open(p):
        parts=line.strip().split("|")
        if len(parts)<3 or parts[1]!="OK": continue
        try: row=json.loads(parts[2])
        except Exception: continue
        n=row.get("config",""); c1=row.get("c1_agg",-1) or -1
        if n.startswith("mtp1"): t=("mtp","1","1")
        elif n.startswith("mtp3"): t=("mtp","3","1")
        elif n.startswith("mtp-off"): t=("off","3","0")
        else: continue
        if c1>best[4]: best=(n,t[0],t[1],t[2],c1)
print("%s %s %s %s %.2f"%best)
PY
)
set -- $BEST; BNAME=$1; BSPEC=$2; BK=$3; BFIX=$4; BC1=$5
say "########## PHASE 2 | winner: $BNAME (c1_agg=$BC1) ##########"
#        name           spec    mtpk  fix    kv        yarn len      seqs qsafp8
run_cfg  fp8kv-mtpoff   off     3     0      fp8_e4m3  1    1000000  16   1
run_cfg  fp8kv-best     $BSPEC  $BK   $BFIX  fp8_e4m3  1    1000000  16   1
run_cfg  streams32      $BSPEC  $BK   $BFIX  bfloat16  1    1000000  32   0
run_cfg  streams64      $BSPEC  $BK   $BFIX  bfloat16  1    1000000  64   0
run_cfg  native262k     $BSPEC  $BK   $BFIX  bfloat16  0    262144   16   0
say "########## PHASE 2 COMPLETE ##########"
