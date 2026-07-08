#!/usr/bin/env bash
# Stop the run WITHOUT purging any cache (so a relaunch reuses all good rollouts).
cd "/mnt/c/Users/occhi/Desktop/life/code/self-prediction-capabilities/experiments/2026-05-16_self_prediction" || exit 1
pkill -9 -f 'run.py --real' 2>/dev/null
pkill -9 -f 'mini-extra' 2>/dev/null
pkill -9 -f 'swebench.harness' 2>/dev/null
pkill -9 -f 'run_full_laptop' 2>/dev/null
sleep 3
docker ps -q | xargs -r docker rm -f >/dev/null 2>&1
echo "run.py still running: $(pgrep -cf 'run.py --real')"
echo "rollouts kept: $(find data_real/rollouts -name 'r*.json' 2>/dev/null | wc -l)"
