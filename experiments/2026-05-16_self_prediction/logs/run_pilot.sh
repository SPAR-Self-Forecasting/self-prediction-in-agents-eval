cd /mnt/c/Users/occhi/Desktop/life/code/self-prediction-capabilities
export PATH=$HOME/exp1venv/bin:$PATH
export MSWEA_COST_TRACKING=ignore_errors
python experiments/2026-05-16_self_prediction/run.py --real \
  --models qwen/qwen3-8b qwen/qwen3-14b google/gemini-2.5-flash-lite \
  --n-tasks 3 --step-limit 250 \
  --data-dir experiments/2026-05-16_self_prediction/data_pilot \
  --results-dir experiments/2026-05-16_self_prediction/results_pilot \
  --rollout-workers 2 --max-workers 6
echo "PILOT_EXIT=$?"
