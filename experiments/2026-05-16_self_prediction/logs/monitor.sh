#!/usr/bin/env bash
# One-shot monitor snapshot for the self-tuning loop. Run via:
#   wsl.exe -d Ubuntu -- bash -lc 'bash "<EXP>/logs/monitor.sh"'
cd "/mnt/c/Users/occhi/Desktop/life/code/self-prediction-capabilities/experiments/2026-05-16_self_prediction" || exit 1
echo "run_alive: $(pgrep -cf 'run.py --real')"
echo "rollouts: $(find data_real/rollouts -name 'r*.json' 2>/dev/null | wc -l) / 3360"
echo "forecasts: $(find data_real/forecasts -name '*.json' 2>/dev/null | wc -l)"
echo "loadavg(1/5/15, 16 cores): $(cut -d' ' -f1-3 /proc/loadavg)"
free -g | awk '/Mem/{print "ram_gb: used="$3" free="$4" total="$2}'
echo "containers: $(docker ps -q | wc -l)"
echo "oldest_agent_proc_sec: $(ps -eo etimes,cmd 2>/dev/null | grep 'mini-extra swebench' | grep -v grep | awk '{print $1}' | sort -rn | head -1)"
echo "summary_exists: $(test -f results_real/run_summary.json && echo YES || echo no)"
echo "--- errored by model ---"
/root/exp1venv/bin/python - <<'PY'
import glob, json, collections
bm = collections.Counter(); tot = err = 0
for f in glob.glob('data_real/rollouts/**/r*.json', recursive=True):
    try: d = json.load(open(f))
    except Exception: continue
    tot += 1
    if d.get('errored'):
        err += 1; bm[d.get('model', '?')] += 1
print(f'total={tot} errored={err}')
for m, n in bm.most_common():
    print(f'  {n}  {m}')
PY
echo "--- log tail ---"
tail -3 logs/full_run.log 2>/dev/null
