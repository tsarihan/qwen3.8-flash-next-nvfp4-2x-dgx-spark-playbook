#!/usr/bin/env bash
# Replay the EXACT measurement set the FP8 artifacts used, against any endpoint.
# usage: replay-bench.sh <tag> <port> <served-model-name>
set -u
TAG="${1:?tag}"; PORT="${2:?port}"; MODEL="${3:?model}"
HOST="${HOST:-192.168.8.10}"   # API head (rank 0); override for your rig
BASE="http://${HOST}:${PORT}/v1"
R="${RESULTS_DIR:-$HOME/glm53bench/results}"
PB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # scripts live beside this file
mkdir -p "$R"

echo "=== [1/3] concurrency ladder 1,2,4,8,16,32,64 @512 tok ==="
python3 "$PB/sweep.py" --base-url "$BASE" --model "$MODEL" \
  --concurrency 1,2,4,8,16,32,64 --max-tokens 512 --tag "${TAG}-ladder" \
  --out "$R/sweep-${TAG}-ladder.json" || echo "LADDER FAILED"

echo "=== [2/3] NIAH short 4096,32768,131072 (5 needles) ==="
python3 "$PB/needles.py" --base-url "$BASE" --model "$MODEL" \
  --contexts 4096,32768,131072 --needles 5 --tag "$TAG" \
  --out "$R/niah-${TAG}.json" || echo "NIAH SHORT FAILED"

echo "=== [3/3] NIAH long 200000,245000 (5 needles) ==="
python3 "$PB/needles.py" --base-url "$BASE" --model "$MODEL" \
  --contexts 200000,245000 --needles 5 --tag "$TAG" \
  --out "$R/niah-${TAG}-long.json" || echo "NIAH LONG FAILED"

echo "=== replay $TAG complete ==="
ls -la "$R"/*${TAG}*
