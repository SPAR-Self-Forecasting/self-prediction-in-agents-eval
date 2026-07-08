#!/usr/bin/env bash
# Launch the FULL real eval (12 models x 30 tasks x 8 rollouts) on this laptop's
# WSL2 Ubuntu, using the existing /root/exp1venv environment.
#
# Detached + resumable: the per-(model,task,rollout) cache in data_real/ means
# you can re-run this script to continue after any stop (crash, reboot, etc.).
set -euo pipefail
REPO="/mnt/c/Users/occhi/Desktop/life/code/self-prediction-capabilities"
EXP="$REPO/experiments/2026-05-16_self_prediction"
PY=/root/exp1venv/bin/python
# Put the venv's bin on PATH so the rollout subprocess can find `mini-extra`
# (run.py shells out to `mini-extra swebench ...`; without this every rollout
# fails with "No such file or directory: 'mini-extra'").
export PATH="/root/exp1venv/bin:$PATH"
cd "$EXP"

# --- load secrets from .env (robust to Windows CRLF / surrounding quotes) ---
if [ -f "$REPO/.env" ]; then
  while IFS='=' read -r k v; do
    k="$(printf '%s' "$k" | tr -d '[:space:]')"
    case "$k" in ''|'#'*|export) continue;; esac
    v="$(printf '%s' "$v" | tr -d '\r')"
    v="${v#\"}"; v="${v%\"}"; v="${v#\'}"; v="${v%\'}"
    eval ": \"\${$k:=\$v}\""; export "$k"
  done < "$REPO/.env"
fi
# litellm lacks price entries for some openrouter ids; don't let that abort agents.
export MSWEA_COST_TRACKING=ignore_errors

# Rollouts are API-wait-bound, not CPU/RAM-bound: real agent containers use only
# tens of MB at rest, so we run many in parallel. 20 workers keeps the machine
# busy; the practical ceiling is OpenRouter rate limits (watched during the run).
# All 12 models, all 30 tasks (no --models / --n-tasks overrides).
exec "$PY" run.py --real \
  --rollout-workers 20 \
  --max-workers 12 \
  --step-limit 250 \
  --data-dir "$EXP/data_real" \
  --results-dir "$EXP/results_real"
