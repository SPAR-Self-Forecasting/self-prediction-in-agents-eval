#!/usr/bin/env bash
# Quick health/progress check for the running full eval.
cd "/mnt/c/Users/occhi/Desktop/life/code/self-prediction-capabilities/experiments/2026-05-16_self_prediction" || exit 1
echo "=== run.py processes (pid -> ppid) ==="
for p in $(pgrep -f 'run.py --real'); do
  ppid=$(awk '{print $4}' /proc/$p/stat 2>/dev/null)
  echo "  pid=$p  ppid=$ppid"
done
echo "=== rollouts on disk (of 3360 target) ==="
find data_real/rollouts -name 'r*.json' 2>/dev/null | wc -l
echo "=== forecasts on disk ==="
find data_real/forecasts -name '*.json' 2>/dev/null | wc -l
echo "=== running containers ==="
docker ps --format '{{.Image}}' 2>/dev/null | sed 's#swebench/sweb.eval.x86_64.##' | head
echo "=== free RAM in WSL ==="
free -g | awk '/Mem/{print "  total="$2"G used="$3"G free="$4"G"}'
echo "=== log tail ==="
tail -10 logs/full_run.log
