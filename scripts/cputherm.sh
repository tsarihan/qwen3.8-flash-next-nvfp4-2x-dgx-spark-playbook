#!/usr/bin/env bash
# CPU-bound thermal probe. Loads every core, samples temps, reports peak and steady state.
# usage: cputherm.sh [seconds]   (default 180)
SECS="${1:-180}"
N=$(nproc)
GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)
hottest() { for z in /sys/class/thermal/thermal_zone*/; do cat "$z/temp" 2>/dev/null; done | sort -rn | head -1; }
echo "host=$(hostname) cores=$N governor=$GOV duration=${SECS}s"
echo "idle_baseline_C=$(( $(hottest)/1000 ))"
for i in $(seq 1 "$N"); do (while :; do :; done) & done
LOADPIDS=$(jobs -p)
peak=0
start=$(date +%s)
while [ $(( $(date +%s) - start )) -lt "$SECS" ]; do
  t=$(( $(hottest)/1000 ))
  f=$(( $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo 0) / 1000 ))
  [ "$t" -gt "$peak" ] && peak=$t
  printf "t+%03ds hottest=%sC cpu0=%sMHz load=%s\n" "$(( $(date +%s) - start ))" "$t" "$f" "$(cut -d' ' -f1 /proc/loadavg)"
  sleep 10
done
kill $LOADPIDS 2>/dev/null
sleep 5
echo "peak_C=$peak"
echo "cooldown_30s_C=$(( $(hottest)/1000 ))"
echo "thermal_slowdown_counters:"
nvidia-smi -q -d PERFORMANCE 2>/dev/null | grep -iE "SW Thermal Slowdown|HW Thermal Slowdown" | head -4
