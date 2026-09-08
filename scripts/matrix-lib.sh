#!/usr/bin/env bash
# Shared driver for the config matrix. Sparks only serve; this runs on OMEN.
# Hard rules: GPU_UTIL never above 0.85, abort any config that drives MemAvailable
# below MEM_FLOOR_GIB, so an unattended run cannot wedge a node.
S1=tsarihan@192.168.8.10
S2=tsarihan@192.168.8.11
BASE=http://192.168.8.10:8892/v1
MODEL=qwen3.8-flash-next-nvfp4
R=$HOME/matrix-results
LOG=$R/matrix.log
MEM_FLOOR_GIB=3
mkdir -p "$R"

say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
mem_gib(){ ssh -o BatchMode=yes -o ConnectTimeout=8 "$1" 'awk "/MemAvailable/{printf \"%d\", \$2/1024/1024}" /proc/meminfo' 2>/dev/null; }
teardown(){
  ssh -o BatchMode=yes -o ConnectTimeout=10 $S1 'docker rm -f qwen38fn-nvidia >/dev/null 2>&1; pkill -f "[m]emsample.sh"' 2>/dev/null
  ssh -o BatchMode=yes -o ConnectTimeout=10 $S2 'docker rm -f qwen38fn-nvidia >/dev/null 2>&1; pkill -f "[m]emsample.sh"' 2>/dev/null
  sleep 8
}

# run_cfg <name> <spec> <mtpk> <mtpfix> <kv> <yarn> <maxlen> <seqs> <qsafp8>
run_cfg(){
  local name="$1" spec="$2" mtpk="$3" mtpfix="$4" kv="$5" yarn="$6" maxlen="$7" seqs="$8" qsafp8="${9:-0}"
  say "=== $name | spec=$spec/$mtpk mtpfix=$mtpfix kv=$kv qsafp8=$qsafp8 yarn=$yarn len=$maxlen seqs=$seqs ==="
  teardown
  local E="MAX_NUM_SEQS=$seqs YARN=$yarn YARN_FACTOR=4.0 MAX_MODEL_LEN=$maxlen GPU_UTIL=0.85 KV_DTYPE=$kv PORT=8892 SPEC=$spec MTP_K=$mtpk MTP_FIX=$mtpfix QSA_FP8=$qsafp8 MODEL_DIR=/data/models/qwen38fn-nvfp4-nvidia YARN_CFG=\$HOME/patches/qwen38fn-nvidia-config-yarn.json NAME=qwen38fn-nvidia"
  ssh -o BatchMode=yes $S1 "cd ~ && NODE_RANK=0 $E ./serve-qwen38fn-nvfp4-vllm.sh" >/dev/null 2>&1
  sleep 5
  ssh -o BatchMode=yes $S2 "cd ~ && NODE_RANK=1 $E ./serve-qwen38fn-nvfp4-vllm.sh" >/dev/null 2>&1
  ssh -o BatchMode=yes $S1 'setsid nohup ~/memsample.sh qwen38fn-nvidia ~/memcurve-matrix.log >/dev/null 2>&1 </dev/null & disown' 2>/dev/null

  local ok=0
  for i in $(seq 1 60); do
    local c=$(ssh -o BatchMode=yes -o ConnectTimeout=8 $S1 'curl -s -o /dev/null -w "%{http_code}" --max-time 6 http://localhost:8892/health' 2>/dev/null)
    [ "$c" = "200" ] && { ok=1; break; }
    local m1=$(mem_gib $S1) m2=$(mem_gib $S2)
    if [ -n "$m1" ] && [ "$m1" -lt "$MEM_FLOOR_GIB" ] 2>/dev/null; then
      say "$name ABORT low mem spark-1 ${m1}GiB"; echo "$name|ABORT_LOWMEM|spark1=${m1}" >> "$R/summary.txt"; teardown; return; fi
    if [ -n "$m2" ] && [ "$m2" -lt "$MEM_FLOOR_GIB" ] 2>/dev/null; then
      say "$name ABORT low mem spark-2 ${m2}GiB"; echo "$name|ABORT_LOWMEM|spark2=${m2}" >> "$R/summary.txt"; teardown; return; fi
    local alive=$(ssh -o BatchMode=yes -o ConnectTimeout=8 $S1 'docker ps --format "{{.Names}}"|grep -qx qwen38fn-nvidia && echo Y || echo N' 2>/dev/null)
    if [ "$alive" = "N" ]; then
      local err=$(ssh -o BatchMode=yes $S1 'docker logs qwen38fn-nvidia 2>&1|grep -iE "Error|Exception"|grep -viE "min_frames|max_frames|use_fast|deprecated"|tail -1|cut -c1-150' 2>/dev/null)
      say "$name FAIL_START: $err"; echo "$name|FAIL_START|$err" >> "$R/summary.txt"; teardown; return; fi
    sleep 20
  done
  [ "$ok" = "1" ] || { say "$name TIMEOUT"; echo "$name|TIMEOUT|" >> "$R/summary.txt"; teardown; return; }

  local kvline=$(ssh -o BatchMode=yes $S1 'docker logs qwen38fn-nvidia 2>&1|grep -oE "GPU KV cache size: [0-9,]+ tokens, Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x"|tail -1' 2>/dev/null)
  local loadline=$(ssh -o BatchMode=yes $S1 'docker logs qwen38fn-nvidia 2>&1|grep -oE "Model loading took [0-9.]+ GiB memory and [0-9.]+ seconds"|tail -1' 2>/dev/null)
  say "$name UP | $kvline | $loadline"

  python3 ~/sweep.py --base-url "$BASE" --model "$MODEL" --concurrency 1,4,16 --max-tokens 512 \
      --tag "$name" --out "$R/sweep-$name.json" >/dev/null 2>&1
  # NIAH doubles as the correctness check, which matters most for the fp8 KV runs
  python3 ~/needles.py --base-url "$BASE" --model "$MODEL" --contexts 4096,131072 --needles 5 \
      --tag "$name" --out "$R/niah-$name.json" >/dev/null 2>&1

  python3 - "$R" "$name" "$kvline" "$loadline" <<'PYEOF' >> "$R/summary.txt"
import json,sys,os
R,name,kvline,loadline=sys.argv[1:5]
def g(f):
    p=os.path.join(R,f); return json.load(open(p)) if os.path.exists(p) else None
sw=g("sweep-%s.json"%name); ni=g("niah-%s.json"%name)
row={"config":name,"kv":kvline,"load":loadline}
if sw:
    for r in sw["rows"]:
        row["c%d_agg"%r["concurrency"]]=round(r["aggregate_tps"],2)
        row["c%d_str"%r["concurrency"]]=round(r["per_stream_decode_tps"],2)
        row["c%d_ttft"%r["concurrency"]]=round(r["ttft_mean_s"],3)
if ni:
    for r in ni["results"]:
        row["niah_%d"%r["target"]]=r.get("needles_found", r.get("error","ERR"))
print("%s|OK|%s"%(name,json.dumps(row)))
PYEOF
  say "$name DONE"
  teardown
}
