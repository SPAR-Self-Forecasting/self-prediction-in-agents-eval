"""Tests for metrics.py -- self-forecast loss + per-dim native-unit transparency (sec 15.6 + 15.7)."""

import pytest

import metrics
from src.types import Dimension, Forecast, Outcome

PSOLVE = Dimension("p_solve", "binary")        # Brier loss for binary
STEPS = Dimension("total_expected_steps", "continuous")  # MAE for continuous
N_ROLL = 8


# --------------------------------------------------------------------------
# Fixture builder
# --------------------------------------------------------------------------
def _cell(model, task, value, *, dim_name="p_solve", n_valid=N_ROLL):
    """A cell where every (valid) rollout has ``dim_name == value``."""
    outs = []
    for i in range(N_ROLL):
        if i < n_valid:
            outs.append(Outcome(model, task, i, {dim_name: float(value)}))
        else:
            outs.append(Outcome(model, task, i, {dim_name: None}, errored=True))
    return outs


def _fc(model, task, condition, kind, predictions, broken=False):
    """Returns a one-element list to match the multi-sample ScoringData shape (sec 9.6).

    Tests are sample-naive; one sample per cell is sufficient and the
    aggregation in metrics is the identity on a single sample.
    """
    return [Forecast(model, task, condition, kind,
                     predictions={} if broken else dict(predictions),
                     broken=broken, sample_idx=0)]


def build_data(self_p, *, outcome_value=1.0, n_valid=N_ROLL, broken_self_task=None):
    """A 3-model / 3-task / 1-binary-dimension ScoringData (t2 unsolved to avoid degeneracy)."""
    models = ["m1", "m2", "m3"]
    families = {"A": ["m1", "m2"], "B": ["m3"]}
    tasks = ["t1", "t2", "t3"]
    outcomes, self_fc = {}, {}
    for m in models:
        outcomes[m] = {}
        self_fc[m] = {}
        for t in tasks:
            nv = n_valid if (m == "m1" and t == "t1") else N_ROLL
            # t2 unsolved so the dimension is non-degenerate across the pooled data.
            v = 0.0 if t == "t2" else outcome_value
            outcomes[m][t] = _cell(m, t, v, n_valid=nv)
            self_fc[m][t] = {}
            for cond in ("no_thinking", "thinking"):
                brk = (broken_self_task == t and m == "m1")
                # self matches t2's truth so the "perfect-self" branch can be exercised
                p = 0.0 if (t == "t2" and self_p == 1.0) else self_p
                self_fc[m][t][cond] = _fc(m, t, cond, "self", {"p_solve": p}, broken=brk)
    return metrics.ScoringData(
        dimensions=(PSOLVE,), models=models, families=families, tasks=tasks,
        outcomes=outcomes, self_forecasts=self_fc,
    )


# --------------------------------------------------------------------------
# Per-cell loss dispatch (sec 15.7)
# --------------------------------------------------------------------------
def test_binary_uses_brier_and_continuous_uses_mae():
    from src.scoring import per_forecast_loss

    # Binary -> squared loss
    assert per_forecast_loss(0.5, [1.0, 0.0], PSOLVE) == pytest.approx(0.25)
    # Continuous -> MAE on raw values
    assert per_forecast_loss(100.0, [50.0, 200.0], STEPS) == pytest.approx(75.0)


# --------------------------------------------------------------------------
# model_scores headline + per-dimension transparency
# --------------------------------------------------------------------------
def test_model_scores_perfect_self_forecast_has_zero_loss():
    data = build_data(self_p=1.0)
    s = metrics.model_scores(data, "m1", "no_thinking")
    assert s["mean_loss_per_dim"]["p_solve"] == pytest.approx(0.0)
    assert s["mean_loss"] == pytest.approx(0.0)


def test_model_scores_constant_miss_has_uniform_brier():
    # self at 0.5 against outcomes 1, 0, 1 -> Brier 0.25 on each cell.
    data = build_data(self_p=0.5)
    s = metrics.model_scores(data, "m1", "no_thinking")
    assert s["mean_loss_per_dim"]["p_solve"] == pytest.approx(0.25)
    assert s["mean_loss"] == pytest.approx(0.25)


def test_model_scores_has_no_retired_cross_aggregates():
    data = build_data(self_p=0.8)
    s = metrics.model_scores(data, "m1", "no_thinking")
    for k in ("IAI", "SFS_self", "SFS_cross", "skill_advantage_per_dim",
              "iai_per_dim", "sfs_self_per_dim", "sfs_cross_per_dim", "floored_dims",
              "wins", "wins_per_dim", "raw_iai_per_dim",
              "mean_loss_self_per_dim", "mean_loss_cross_per_dim"):
        assert k not in s
    assert "mean_loss_per_dim" in s
    assert "mean_loss" in s


# --------------------------------------------------------------------------
# Missing data (sec 9.6)
# --------------------------------------------------------------------------
def test_cell_below_valid_threshold_is_dropped():
    # m1/t1 has only 4 valid rollouts (< 5) -> that task drops for m1.
    data = build_data(self_p=0.8, n_valid=4)
    s = metrics.model_scores(data, "m1", "no_thinking")
    assert s["n_tasks_used"]["p_solve"] == 2  # t1 dropped, t2 + t3 remain


def test_broken_forecast_drops_the_task():
    data = build_data(self_p=0.8, broken_self_task="t3")
    s = metrics.model_scores(data, "m1", "no_thinking")
    assert s["n_tasks_used"]["p_solve"] == 2  # t3 self forecast broken -> dropped


# --------------------------------------------------------------------------
# Bootstrap-compatible recompute
# --------------------------------------------------------------------------
def test_model_scores_accepts_resample_indices():
    data = build_data(self_p=0.8)
    point = metrics.model_scores(data, "m1", "no_thinking")["mean_loss"]
    explicit = metrics.model_scores(
        data, "m1", "no_thinking",
        task_indices=[0, 1, 2],
        rollout_indices=[list(range(N_ROLL))] * 3,
    )["mean_loss"]
    assert point == pytest.approx(explicit)


def test_all_model_scores_covers_models_with_self_forecasts():
    data = build_data(self_p=0.8)
    scores = metrics.all_model_scores(data, "no_thinking")
    assert set(scores) == {"m1", "m2", "m3"}
    assert all("mean_loss" in s for s in scores.values())


def test_equal_dimension_mean_loss_prevents_one_dim_from_domination():
    mlpds = {
        "m1": {"p_solve": 0.5, "avg_tokens_step": 500.0},
        "m2": {"p_solve": 0.5, "avg_tokens_step": 100.0},
    }
    dim_names = ["p_solve", "avg_tokens_step"]
    scales = metrics.dimension_loss_scales(mlpds, dim_names)
    native_m1 = (0.5 + 500.0) / 2
    equal_m1 = metrics.equal_dimension_mean_loss(mlpds["m1"], scales, dim_names)
    assert native_m1 > 200
    assert equal_m1 == pytest.approx((0.5 / 0.5 + 500.0 / 500.0) / 2)
    assert equal_m1 <= 1.0
