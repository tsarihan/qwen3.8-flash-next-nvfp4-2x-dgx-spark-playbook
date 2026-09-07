#!/usr/bin/env bash
# Sample MemAvailable every 2s and tag each sample with the newest vLLM log line,
# so the memory curve can be correlated against kernel selection vs weight loading.
CONT="${1:?container name}"
OUT="${2:-$HOME/memcurve.log}"
: > "$OUT"
start=$(date +%s)
while true; do
  avail=$(awk '/MemAvailable/{printf "%.1f", $2/1024/1024}' /proc/meminfo)
  t=$(( $(date +%s) - start ))
  line=$(docker logs --tail 1 "$CONT" 2>&1 | tr -d '\r' | tail -1 | cut -c1-160)
  printf 'T+%04ds available=%sGiB | %s\n' "$t" "$avail" "$line" >> "$OUT"
  docker ps --format '{{.Names}}' | grep -qx "$CONT" || { echo "container gone at T+${t}s" >> "$OUT"; break; }
  sleep 2
done
