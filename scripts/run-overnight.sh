#!/usr/bin/env bash
set -u
while pgrep -f "[r]evalidate-nvidia.sh" >/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] baseline done, phase 1 starting" >> ~/overnight.log
~/phase1.sh >> ~/overnight.log 2>&1
echo "[$(date +%H:%M:%S)] phase 1 done, phase 2 starting" >> ~/overnight.log
~/phase2.sh >> ~/overnight.log 2>&1
echo "[$(date +%H:%M:%S)] ALL PHASES DONE" >> ~/overnight.log
