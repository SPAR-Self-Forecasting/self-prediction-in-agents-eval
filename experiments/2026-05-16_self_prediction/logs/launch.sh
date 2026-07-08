#!/usr/bin/env bash
# Detach + launch the full eval, then verify it started.
cd "/mnt/c/Users/occhi/Desktop/life/code/self-prediction-capabilities/experiments/2026-05-16_self_prediction" || exit 1
: > logs/full_run.log
setsid nohup bash run_full_laptop.sh >logs/full_run.log 2>&1 </dev/null &
sleep 12
echo "--- run.py proc ---"
pgrep -af 'run.py --real' | grep -v pgrep || echo "NOT RUNNING"
echo "--- log head ---"
head -25 logs/full_run.log
