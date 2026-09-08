#!/usr/bin/env bash
# Shared helpers for the manual model-switch scripts. Sourced, not run directly.
# These orchestrate BOTH DGX Sparks from this host (OMEN). No autoswitch: you run
# one switch-<model>.sh by hand when you want that model up.
set -uo pipefail
S1=tsarihan@192.168.8.10   # node-rank 0, the API head
S2=tsarihan@192.168.8.11   # node-rank 1
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"

# Only these container names are ever torn down. Everything else on the nodes
# (qdrant, postgres, open-webui, ollama, etc.) is left untouched.
MANAGED="qwen38fn-nvidia glm53-nvfp4-vllm dsv4-anemll dsv4-0731-official deepseek-v4-flash-vllm-dspark-1"

say(){ echo "[$(date +%H:%M:%S)] $*"; }

teardown(){
  say "tearing down managed model containers on both nodes"
  for H in "$S1" "$S2"; do
    $SSH "$H" "for c in $MANAGED; do sudo docker rm -f \$c >/dev/null 2>&1 || true; done"
  done
  # wait for unified memory to actually come back before launching the next model
  for i in $(seq 1 60); do
    A=$($SSH "$S1" "free -g | awk '/Mem:/{print \$7}'" 2>/dev/null)
    B=$($SSH "$S2" "free -g | awk '/Mem:/{print \$7}'" 2>/dev/null)
    A=${A:-0}; B=${B:-0}
    say "free: spark-1 ${A}G  spark-2 ${B}G"
    [ "$A" -ge 60 ] && [ "$B" -ge 60 ] && { say "memory reclaimed"; return 0; }
    sleep 10
  done
  say "WARNING: memory did not fully reclaim after 10 min; launching anyway"
}

launch(){  # $1 = env string (shared by both ranks), $2 = serve script, $3 = port
  local ENV="$1" SCRIPT="$2" PORT="$3"
  say "launching rank 0 on spark-1"
  $SSH "$S1" "cd ~ && NODE_RANK=0 $ENV ./$SCRIPT" >/dev/null 2>&1
  sleep 5
  say "launching rank 1 on spark-2"
  $SSH "$S2" "cd ~ && NODE_RANK=1 $ENV ./$SCRIPT" >/dev/null 2>&1
  say "waiting for /health on spark-1:$PORT (poll /health, never /v1/models)"
  for i in $(seq 1 90); do
    c=$($SSH "$S1" "curl -s -o /dev/null -w '%{http_code}' --max-time 6 http://localhost:$PORT/health" 2>/dev/null)
    [ "$c" = "200" ] && { say "UP and healthy on :$PORT"; return 0; }
    sleep 20
  done
  say "ERROR: did not reach healthy in 30 min. Check: ssh $S1 'sudo docker logs -f <name>'"
  return 1
}

served_note(){  # $1 = served-model-name, $2 = port
  echo
  say "served-model-name: $1   endpoint: http://192.168.8.10:$2/v1"
  say "litellm on OMEN routes to this; nothing else is up (single production model)."
}
