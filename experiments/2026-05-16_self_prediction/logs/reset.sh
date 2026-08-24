#!/usr/bin/env bash
# Stop the run and purge the caches poisoned by the 'mini-extra not found' bug:
# all rollouts + their derived outcomes + run-level aggregates. Keeps forecasts/
# (elicited predictions, independent of rollout success) to avoid re-spending.
cd "/mnt/c/Users/occhi/Desktop/life/code/self-prediction-capabilities/experiments/2026-05-16_self_prediction" || exit 1
echo "=== stopping run ==="
pkill -9 -f 'run.py --real' 2>/dev/null
pkill -9 -f 'mini-extra' 2>/dev/null
pkill -9 -f 'swebench.harness' 2>/dev/null
pkill -9 -f 'run_full_laptop' 2>/dev/null
sleep 3
docker ps -q | xargs -r docker rm -f >/dev/null 2>&1
echo "run.py still running: $(pgrep -cf 'run.py --real')"
echo "=== purging poisoned caches ==="
rm -rf data_real/rollouts data_real/outcomes data_real/tasks.json results_real
echo "=== remaining in data_real ==="
ls -1 data_real 2>/dev/null
echo "kept forecasts: $(find data_real/forecasts -name '*.json' 2>/dev/null | wc -l) files"
