#!/usr/bin/env bash
# Stop any running eval, clear stray containers, and remove stale run-level
# caches (tasks.json) so a relaunch reselects all 30 tasks.
cd "/mnt/c/Users/occhi/Desktop/life/code/self-prediction-capabilities/experiments/2026-05-16_self_prediction" || exit 1
echo "=== before: run.py procs ==="
pgrep -af 'run.py --real' || echo none
pkill -9 -f 'run.py --real' 2>/dev/null
pkill -9 -f 'mini-extra' 2>/dev/null
pkill -9 -f 'swebench.harness' 2>/dev/null
pkill -9 -f 'run_full_laptop' 2>/dev/null
sleep 3
echo "=== stray containers (removing) ==="
docker ps -q | xargs -r docker rm -f >/dev/null 2>&1
docker ps
echo "=== after: run.py proc count ==="
pgrep -cf 'run.py --real'
echo "=== stale tasks.json ==="
/root/exp1venv/bin/python -c 'import json;print("task count:",len(json.load(open("data_real/tasks.json"))))' 2>/dev/null || echo "no tasks.json"
rm -fv data_real/tasks.json
echo "=== data_real contents ==="
ls -1 data_real 2>/dev/null
