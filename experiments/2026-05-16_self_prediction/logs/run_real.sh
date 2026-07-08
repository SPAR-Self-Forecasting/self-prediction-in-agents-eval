cd /mnt/c/Users/occhi/Desktop/life/code/self-prediction-capabilities
export PATH=$HOME/exp1venv/bin:$PATH
# mini-swe-agent's cost tracking calls litellm to price the model; litellm has no
# price entry for some openrouter/* ids and raises, aborting the agent after 1
# call. Tell mini-swe-agent to ignore cost-tracking errors.
export MSWEA_COST_TRACKING=ignore_errors
python experiments/2026-05-16_self_prediction/run.py --real \
  --models qwen/qwen3-8b qwen/qwen3-14b --n-tasks 1 --step-limit 40 \
  --data-dir experiments/2026-05-16_self_prediction/data_real \
  --results-dir experiments/2026-05-16_self_prediction/results_real \
  --rollout-workers 2 --max-workers 4
echo "RUN_EXIT=$?"
