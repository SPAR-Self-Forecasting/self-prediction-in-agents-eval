"""mini-swe-agent rollout driver for Experiment 1 (EVAL_SPEC_V1.md sec 6, sec 12).

Two modes:

- **Real** -- drives the ``mini-extra swebench`` CLI as a subprocess (clarification
  Q5): it sets up the SWE-bench Docker environment, runs the ``DefaultAgent``, and
  writes a ``mini-swe-agent-1.1`` trajectory + a ``preds.json``. ``p_solve`` ground
  truth then comes from the official ``swebench`` harness. The 250-step limit is
  the sole budget -- ``cost_limit`` is disabled (finding F3 / Q9).
- **Mock** -- generates deterministic synthetic trajectories so the test suite and
  the mocked dry-run never need Docker, the network, or money. Mock trajectories
  are structurally faithful to ``mini-swe-agent-1.1`` and are extractable by every
  ``trajectory.py`` extractor.

``mini-swe-agent`` / ``swebench`` are optional dependencies, imported only on the
real path; nothing here imports them at module load.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import config
from dataset import Task
from src.types import Rollout

_UNCERTAIN_LINES = [
    "I'm not certain this is the root cause, but it is the most likely one.",
    "This might be tricky; the failing test could depend on other modules.",
    "Let me double-check the surrounding code before editing.",
]
_CONFIDENT_LINES = [
    "The bug is clear. I will fix it directly.",
    "This is a straightforward off-by-one error.",
    "I have located the defect and will apply the fix.",
]


# --------------------------------------------------------------------------
# Real-run command construction
# --------------------------------------------------------------------------
def litellm_model_string(model_id: str) -> str:
    """OpenRouter model id -> the litellm routing string mini-swe-agent expects."""
    return f"openrouter/{model_id}"


def _instance_filter(instance_id: str) -> str:
    """An exact-match ``--filter`` regex for one instance id.

    SWE-bench instance ids use only ``-`` / ``_`` / alphanumerics plus the
    occasional ``.`` -- only ``.`` is a regex metacharacter, so escape just that
    (and ``\\``); over-escaping with ``re.escape`` would mangle the readable id.
    """
    escaped = instance_id.replace("\\", "\\\\").replace(".", r"\.")
    return f"^{escaped}$"


def build_swebench_command(
    model_id: str, instance_id: str, output_dir: str | Path, step_limit: int
) -> list[str]:
    """The ``mini-extra swebench`` CLI invocation for one (model, instance) rollout."""
    return [
        "mini-extra", "swebench",
        "--subset", config.SWEBENCH_DATASET,
        "--split", config.SWEBENCH_SPLIT,
        "--filter", _instance_filter(instance_id),
        "--model", litellm_model_string(model_id),
        "-o", str(output_dir),
        "-c", "swebench.yaml",
        "-c", f"agent.step_limit={step_limit}",
        "-c", f"agent.cost_limit={config.ROLLOUT_COST_LIMIT}",
    ]


def build_evaluation_command(
    predictions_path: str | Path, run_id: str, workers: int = 4
) -> list[str]:
    """The ``swebench.harness.run_evaluation`` invocation that yields ``p_solve``."""
    return [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", config.SWEBENCH_DATASET,
        "--split", config.SWEBENCH_SPLIT,
        "--predictions_path", str(predictions_path),
        "--run_id", run_id,
        "--max_workers", str(workers),
    ]


def load_trajectory_file(path: str | Path) -> dict:
    """Load a ``*.traj.json`` trajectory file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Real rollout
# --------------------------------------------------------------------------
def run_real_rollout(
    model_id: str, task: Task, rollout_idx: int, output_dir: str | Path, step_limit: int
) -> Rollout:
    """Run one real rollout via the ``mini-extra swebench`` CLI.

    ``solved`` is left None -- it is filled by the SWE-bench evaluation harness in
    pipeline Stage 3. A subprocess or parse failure yields an errored Rollout.
    """
    cell_dir = Path(output_dir) / model_id.replace("/", "__") / f"rollout_{rollout_idx}" / task.instance_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_swebench_command(model_id, task.instance_id, cell_dir, step_limit)
    # mini-swe-agent prices each call via litellm; litellm lacks a price entry for
    # some openrouter/* ids and raises, which aborts the agent after one call.
    # ``ignore_errors`` skips cost accounting (real billing is unaffected).
    env = {**os.environ, "MSWEA_COST_TRACKING": "ignore_errors"}
    try:
        # Cap each rollout: a hung model API call (no inner timeout) would otherwise
        # run forever and permanently occupy a worker slot, starving throughput.
        # On timeout the subprocess is killed and TimeoutExpired -> errored rollout
        # (recoverable; retried in the end cleanup pass). 6000s comfortably exceeds
        # legit slow rollouts (~50 min) while catching multi-hour hangs.
        subprocess.run(cmd, check=True, capture_output=True, text=True, env=env, timeout=6000)
        traj_files = sorted(cell_dir.rglob("*.traj.json"))
        if not traj_files:
            raise FileNotFoundError(f"no .traj.json produced under {cell_dir}")
        trajectory = load_trajectory_file(traj_files[0])
    except Exception as exc:  # subprocess / parse failure -> errored rollout
        return Rollout(model=model_id, task_id=task.instance_id, rollout_idx=rollout_idx,
                       errored=True, error=str(exc))
    return Rollout(model=model_id, task_id=task.instance_id, rollout_idx=rollout_idx,
                   trajectory=trajectory, solved=None, errored=False)


def run_swebench_evaluation(predictions_path: str | Path, run_id: str, eval_cwd: str | Path,
                            workers: int = 4) -> None:
    """Run the official SWE-bench harness on a ``preds.json`` (real run only)."""
    subprocess.run(build_evaluation_command(predictions_path, run_id, workers),
                   check=True, cwd=str(eval_cwd))


def read_resolved(
    log_root: str | Path, run_id: str, prediction_name: str, instance_id: str
) -> bool:
    """Read the ``resolved`` boolean from a SWE-bench harness per-instance report.

    ``prediction_name`` is the ``model_name_or_path`` written into ``preds.json``;
    the harness escapes ``/`` to ``__`` in its log directory layout.
    """
    escaped = prediction_name.replace("/", "__")
    report = (Path(log_root) / "logs" / "run_evaluation" / run_id / escaped
              / instance_id / "report.json")
    if not report.is_file():
        return False
    data = json.loads(report.read_text(encoding="utf-8"))
    return bool(data.get(instance_id, {}).get("resolved", False))


# --------------------------------------------------------------------------
# Mock rollout (dry-run / tests)
# --------------------------------------------------------------------------
def _mock_seed(model_id: str, task_id: str, rollout_idx: int) -> int:
    """A stable (cross-process) RNG seed for one mock rollout."""
    digest = hashlib.sha256(f"{model_id}|{task_id}|{rollout_idx}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _mock_patch_file(task: Task) -> str:
    return f"src/{task.repo.split('/')[-1]}/module.py"


def _mock_diff(patch_file: str, rng: random.Random) -> str:
    delta = rng.randint(1, 9)
    return (
        f"diff --git a/{patch_file} b/{patch_file}\n"
        f"--- a/{patch_file}\n+++ b/{patch_file}\n"
        f"@@ -1,3 +1,4 @@\n def compute(x):\n"
        f"-    return x\n+    # adjusted for the reported edge case\n+    return x + {delta}\n"
    )


def _mock_action_message(command: str, content: str, completion_tokens: int) -> dict:
    return {
        "role": "assistant",
        "content": content,
        "extra": {
            "actions": [{"command": command, "tool_call_id": "tc"}],
            "response": {"usage": {
                "prompt_tokens": 200,
                "completion_tokens": completion_tokens,
                "total_tokens": 200 + completion_tokens,
            }},
            "cost": round(completion_tokens * 1e-6, 8),
        },
    }


def _mock_trajectory(task: Task, rng: random.Random, solved: bool, hit_limit: bool) -> dict:
    patch_file = _mock_patch_file(task)
    explore_cmds = ["ls -la", "grep -rn compute .", f"cat {patch_file}", "find . -name '*.py'",
                    "git log --oneline -5"]

    steps: list[tuple[str, str]] = []  # (command, kind)
    n_explore = rng.randint(2, 9)
    for _ in range(n_explore):
        steps.append((rng.choice(explore_cmds), "explore"))
    if not hit_limit:
        for _ in range(rng.randint(1, 3)):
            steps.append((f"sed -i 's/return x/return x + 1/' {patch_file}", "edit"))
        for _ in range(rng.randint(1, 4)):
            steps.append((rng.choice(["pytest tests/", "python -m pytest -q"]), "test"))

    messages: list[dict] = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": task.forecast_description()},
    ]
    for command, _kind in steps:
        line = rng.choice(_UNCERTAIN_LINES if rng.random() < 0.4 else _CONFIDENT_LINES)
        messages.append(_mock_action_message(command, line, rng.randint(90, 620)))
        messages.append({"role": "tool", "tool_call_id": "tc", "content": "output...",
                         "extra": {"raw_output": "output...", "returncode": 0}})

    exit_status = "LimitsExceeded" if hit_limit else "Submitted"
    submission = "" if hit_limit else _mock_diff(patch_file, rng)
    messages.append({"role": "exit", "content": exit_status,
                     "extra": {"exit_status": exit_status, "submission": submission}})

    return {
        "info": {
            "exit_status": exit_status,
            "submission": submission,
            "model_stats": {"api_calls": len(steps)},
        },
        "messages": messages,
        "trajectory_format": "mini-swe-agent-1.1",
    }


def mock_rollout(model_id: str, task: Task, rollout_idx: int) -> Rollout:
    """A deterministic synthetic rollout for the mocked dry-run / tests.

    ``solved`` is drawn so the mock honours the sec 7 capability ladder
    (P(solve) ~ the model's estimated SWE-bench capability), which makes the
    Stage-4 cross-predictor designation behave sensibly in a dry-run.
    """
    spec = config.model_by_id(model_id)
    rng = random.Random(_mock_seed(model_id, task.instance_id, rollout_idx))
    solved = rng.random() < spec.est_swebench
    hit_limit = rng.random() < 0.05
    trajectory = _mock_trajectory(task, rng, solved, hit_limit)
    return Rollout(
        model=model_id,
        task_id=task.instance_id,
        rollout_idx=rollout_idx,
        trajectory=trajectory,
        solved=solved,
        errored=False,
    )
