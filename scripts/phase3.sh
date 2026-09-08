#!/usr/bin/env bash
# Phase 3: stand up the overall winner and run SWE-bench Pro 40 through litellm,
# which validates real agentic work AND the proxy hop in one go.
set -u
source ~/matrix-lib.sh

WIN=$(python3 - "$R/summary.txt" <<'PY'
import json,sys,os
# rank by c16 aggregate (throughput) but require 5/5 needles at both depths
best=None
p=sys.argv[1]
if os.path.exists(p):
    for line in open(p):
        parts=line.strip().split("|",2)
        if len(parts)<3 or parts[1]!="OK": continue
        try: r=json.loads(parts[2])
        except Exception: continue
        if str(r.get("niah_4096"))!="5/5" or str(r.get("niah_131072"))!="5/5": continue
        # The user needs the 1M window: 262k is too short for full-repo work and they
        # autocompact at 800k. Exclude native262k regardless of how fast it is.
        if r.get("config")=="native262k": continue
        # Rank by SINGLE STREAM, not c16. This rig serves full-repo agentic coding:
        # one long-lived stream at large context, not a fleet of concurrent agents.
        if best is None or (r.get("c1_agg") or 0)>(best.get("c1_agg") or 0): best=r
name=(best or {}).get("config","mtp-off-bf16")
m={"mtp1":("mtp","1","1"),"mtp3":("mtp","3","1"),"mtp-off":("off","3","0"),
   "fp8kv-mtpoff":("off","3","0"),"streams32":("off","3","0"),"streams64":("off","3","0"),
   "native262k":("off","3","0")}
spec,k,fix=("off","3","0")
for pre,t in m.items():
    if name.startswith(pre): spec,k,fix=t; break
kv="fp8_e4m3" if name.startswith("fp8kv") else "bfloat16"
q="1" if name.startswith("fp8kv") else "0"
seqs="32" if name=="streams32" else ("64" if name=="streams64" else "16")
yarn="0" if name=="native262k" else "1"
ln="262144" if name=="native262k" else "1000000"
print("%s %s %s %s %s %s %s %s"%(name,spec,k,fix,kv,q,seqs,ln))
PY
)
set -- $WIN; WN=$1; WSPEC=$2; WK=$3; WFIX=$4; WKV=$5; WQ=$6; WSEQ=$7; WLEN=$8
say "########## PHASE 3 | winner $WN, seqs forced to 64 for the full ladder ##########"

teardown
YARNV=$( [ "$WN" = "native262k" ] && echo 0 || echo 1 )
ENVS="MAX_NUM_SEQS=64 YARN=$YARNV YARN_FACTOR=4.0 MAX_MODEL_LEN=$WLEN GPU_UTIL=0.85 KV_DTYPE=$WKV PORT=8892 SPEC=$WSPEC MTP_K=$WK MTP_FIX=$WFIX QSA_FP8=$WQ MODEL_DIR=/data/models/qwen38fn-nvfp4-nvidia YARN_CFG=\$HOME/patches/qwen38fn-nvidia-config-yarn.json NAME=qwen38fn-nvidia"
ssh -o BatchMode=yes $S1 "cd ~ && NODE_RANK=0 $ENVS ./serve-qwen38fn-nvfp4-vllm.sh" >/dev/null 2>&1
sleep 5
ssh -o BatchMode=yes $S2 "cd ~ && NODE_RANK=1 $ENVS ./serve-qwen38fn-nvfp4-vllm.sh" >/dev/null 2>&1

ok=0
for i in $(seq 1 60); do
  c=$(ssh -o BatchMode=yes -o ConnectTimeout=8 $S1 'curl -s -o /dev/null -w "%{http_code}" --max-time 6 http://localhost:8892/health' 2>/dev/null)
  [ "$c" = "200" ] && { ok=1; break; }
  sleep 20
done
[ "$ok" = "1" ] || { say "PHASE 3 could not start winner, skipping SWE"; exit 0; }
say "winner up, running FULL concurrency ladder first (matrix only tested c<=16)"
python3 ~/sweep.py --base-url "$BASE" --model "$MODEL" --concurrency 1,2,4,8,16,32,64 \
    --max-tokens 512 --tag winner-fullladder --out "$R/sweep-winner-fullladder.json" >/dev/null 2>&1
python3 - "$R/sweep-winner-fullladder.json" <<'PY2' | tee -a "$LOG"
import json,sys,os
p=sys.argv[1]
if os.path.exists(p):
    d=json.load(open(p))
    print("FULL LADDER (winner, max_num_seqs=64)")
    print("  conc   agg tok/s   per-stream   ttft s")
    best=None
    for r in d["rows"]:
        print("  %4d %11.2f %12.2f %8.3f"%(r["concurrency"],r["aggregate_tps"],
              r["per_stream_decode_tps"],r["ttft_mean_s"]))
        if best is None or r["aggregate_tps"]>best["aggregate_tps"]: best=r
    print("  peak aggregate at c=%d: %.2f tok/s"%(best["concurrency"],best["aggregate_tps"]))
PY2
say "relaunching winner at max_num_seqs=16 (the recommended setting) for SWE"
teardown
ENVS16=$(echo "$ENVS" | sed "s/MAX_NUM_SEQS=64/MAX_NUM_SEQS=16/")
ssh -o BatchMode=yes $S1 "cd ~ && NODE_RANK=0 $ENVS16 ./serve-qwen38fn-nvfp4-vllm.sh" >/dev/null 2>&1
sleep 5
ssh -o BatchMode=yes $S2 "cd ~ && NODE_RANK=1 $ENVS16 ./serve-qwen38fn-nvfp4-vllm.sh" >/dev/null 2>&1
for i in $(seq 1 60); do
  c=$(ssh -o BatchMode=yes -o ConnectTimeout=8 $S1 'curl -s -o /dev/null -w "%{http_code}" --max-time 6 http://localhost:8892/health' 2>/dev/null)
  [ "$c" = "200" ] && break; sleep 20
done
say "starting SWE 40 through litellm at seqs=16"

cd ~/swe-enterprise40 && OPENAI_API_KEY="${LITELLM_KEY:?set this to your litellm master key}" ./run-suite.sh "nvidia-$WN" "openai/qwen-3.8-flash-next[1m]" http://127.0.0.1:4000/v1 5 10 >> ~/overnight.log 2>&1
say "SWE run finished, grading"
python3 ~/conv_preds.py "nvidia-$WN" >> ~/overnight.log 2>&1
cd ~/swebench-pro/SWE-bench_Pro-os && ~/swebench-pro/.venv/bin/python swe_bench_pro_eval.py \
  --raw_sample_path=$HOME/swe-enterprise40/dataset/test.jsonl \
  --patch_path=$HOME/swe-enterprise40/patches-nvidia-$WN.json \
  --output_dir=$HOME/swe-enterprise40/eval-nvidia-$WN \
  --scripts_dir=run_scripts --num_workers=3 --dockerhub_username=jefzda --use_local_docker >> ~/overnight.log 2>&1
python3 - "$HOME/swe-enterprise40/eval-nvidia-$WN/eval_results.json" <<'PY' | tee -a "$LOG"
import json,sys,os
p=sys.argv[1]
if os.path.exists(p):
    d=json.load(open(p)); r=sum(1 for v in d.values() if v)
    print("SWE RESULT: %d/%d = %.1f%%"%(r,len(d),100*r/len(d)))
else:
    print("SWE RESULT: eval_results.json not written")
PY
say "########## PHASE 3 COMPLETE ##########"
