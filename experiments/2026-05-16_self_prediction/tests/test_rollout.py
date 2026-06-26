"""Tests for rollout.py -- the mini-swe-agent driver and its mock mode."""

import config
import rollout
import trajectory as T
from dataset import mock_tasks


TASKS = mock_tasks()


# --------------------------------------------------------------------------
# mock_rollout: structure
# --------------------------------------------------------------------------
def test_mock_rollout_returns_valid_rollout():
    r = rollout.mock_rollout("qwen/qwen3-8b", TASKS[0], rollout_idx=0)
    assert r.model == "qwen/qwen3-8b"
    assert r.task_id == TASKS[0].instance_id
    assert r.rollout_idx == 0
    assert r.errored is False
    assert isinstance(r.solved, bool)
    assert r.trajectory["trajectory_format"] == "mini-swe-agent-1.1"


def test_mock_trajectory_has_canonical_structure():
    r = rollout.mock_rollout("z-ai/glm-5", TASKS[3], rollout_idx=2)
    traj = r.trajectory
    assert "info" in traj and "messages" in traj
    assert "submission" in traj["info"] and "exit_status" in traj["info"]
    # at least one assistant action message carrying extra.actions + usage
    actions = T.action_messages(traj)
    assert len(actions) >= 1
    assert actions[0]["extra"]["actions"][0]["command"]
    assert "completion_tokens" in actions[0]["extra"]["response"]["usage"]


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------
def test_mock_rollout_is_deterministic():
    a = rollout.mock_rollout("openai/gpt-4.1-mini", TASKS[5], 1)
    b = rollout.mock_rollout("openai/gpt-4.1-mini", TASKS[5], 1)
    assert a.trajectory == b.trajectory
    assert a.solved == b.solved


def test_mock_rollouts_vary_across_index():
    step_counts = {
        T.count_steps(rollout.mock_rollout("z-ai/glm-4.5", TASKS[0], i).trajectory)
        for i in range(8)
    }
    assert len(step_counts) > 1  # the 8 rollouts of a cell are not all identical


# --------------------------------------------------------------------------
# Mock trajectories are extractable by trajectory.py
# --------------------------------------------------------------------------
def test_mock_trajectory_is_extractable():
    r = rollout.mock_rollout("deepseek/deepseek-v3.2", TASKS[7], 3)
    vals = T.extract_deterministic(r.trajectory, solved=r.solved)
    assert vals["total_expected_steps"] >= 1
    assert vals["p_solve"] in (0.0, 1.0)
    # avg_tokens_step present (mock always emits usage metadata)
    assert vals["avg_tokens_step"] is not None


def test_mock_capability_ladder_holds_on_average():
    """A high-capability model solves more mock tasks than a low-capability one."""
    def solve_rate(model_id):
        hits = sum(
            rollout.mock_rollout(model_id, t, i).solved
            for t in TASKS for i in range(config.N_ROLLOUTS)
        )
        return hits / (len(TASKS) * config.N_ROLLOUTS)

    assert solve_rate("z-ai/glm-5") > solve_rate("qwen/qwen3-next-80b-a3b-instruct")


# --------------------------------------------------------------------------
# Real-run command construction (subprocess not executed here)
# --------------------------------------------------------------------------
def test_build_swebench_command():
    cmd = rollout.build_swebench_command(
        "qwen/qwen3-8b", instance_id="repo__x-1", output_dir="/tmp/out", step_limit=250
    )
    assert "mini-extra" in cmd and "swebench" in cmd
    assert "openrouter/qwen/qwen3-8b" in cmd
    # step limit and the disabled cost limit are both passed (finding F3 / Q9)
    assert any("agent.step_limit=250" in c for c in cmd)
    assert any("agent.cost_limit=" in c for c in cmd)
    assert any("repo__x-1" in c for c in cmd)


def test_build_evaluation_command():
    cmd = rollout.build_evaluation_command("/tmp/preds.json", run_id="exp1_run")
    assert "swebench.harness.run_evaluation" in cmd
    assert "/tmp/preds.json" in cmd
    assert "exp1_run" in cmd
