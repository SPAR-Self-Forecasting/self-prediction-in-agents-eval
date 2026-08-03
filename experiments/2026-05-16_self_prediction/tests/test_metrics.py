"""Tests for metrics.py -- wins-fraction + per-dim native-unit transparency (sec 15.6 + 15.7)."""

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


def build_data(self_p, cross_p, *, outcome_value=1.0, n_valid=N_ROLL, broken_self_task=None):
    """A 3-model / 3-task / 1-binary-dimension ScoringData (t2 unsolved to avoid degeneracy)."""
    models = ["m1", "m2", "m3"]
    families = {"A": ["m1", "m2"], "B": ["m3"]}
    tasks = ["t1", "t2", "t3"]
    outcomes, self_fc, cross_fc = {}, {}, {}
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
    for t in tasks:
        cross_fc[t] = {c: _fc("m3", t, c, "anonymous", {"p_solve": cross_p}) for c in ("no_thinking", "thinking")}
    return metrics.ScoringData(
        dimensions=(PSOLVE,), models=models, families=families, tasks=tasks,
        cross_model="m3", outcomes=outcomes, self_forecasts=self_fc, cross_forecasts=cross_fc,
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
# wins_fraction headline
# --------------------------------------------------------------------------
def test_wins_one_when_self_is_perfect():
    # self predicts each cell exactly; cross misses -> self wins every cell -> 1.0
    data = build_data(self_p=1.0, cross_p=0.5)
    assert metrics.wins_fraction(data, "m1", "no_thinking") == pytest.approx(1.0)


def test_wins_half_on_ties():
    # self == cross -> every cell is a tie -> wins == 0.5
    data = build_data(self_p=0.5, cross_p=0.5)
    assert metrics.wins_fraction(data, "m1", "no_thinking") == pytest.approx(0.5)


def test_wins_zero_when_cross_is_perfect():
    # cross matches truth, self misses
    data = build_data(self_p=0.5, cross_p=1.0)
    # patch cross t2 to also match its truth (0.0)
    for cond in ("no_thinking", "thinking"):
        data.cross_forecasts["t2"][cond] = _fc("m3", "t2", cond, "anonymous", {"p_solve": 0.0})
    assert metrics.wins_fraction(data, "m1", "no_thinking") == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Per-dimension transparency (native units, sec 15.7)
# --------------------------------------------------------------------------
def test_model_scores_returns_raw_iai_in_native_units():
    # self predicts every truth; cross at 0.5; outcomes are 1, 0, 1 across t1/t2/t3.
    data = build_data(self_p=1.0, cross_p=0.5)
    s = metrics.model_scores(data, "m1", "no_thinking")
    # mean self loss = 0; mean cross loss = (0.5-1)^2 = 0.25 each -> 0.25
    assert s["mean_loss_self_per_dim"]["p_solve"] == pytest.approx(0.0)
    assert s["mean_loss_cross_per_dim"]["p_solve"] == pytest.approx(0.25)
    # raw_iai = mean_cross - mean_self = 0.25 (in Brier units for this binary dim)
    assert s["raw_iai_per_dim"]["p_solve"] == pytest.approx(0.25)


def test_model_scores_returns_wins_per_dim():
    # Self perfect on the single dim -> per-dim wins = 1.0 too.
    data = build_data(self_p=1.0, cross_p=0.5)
    s = metrics.model_scores(data, "m1", "no_thinking")
    assert s["wins_per_dim"]["p_solve"] == pytest.approx(1.0)
    # Tied losses across every cell -> per-dim wins = 0.5 (B2 heatmap mid-tone).
    tied = build_data(self_p=0.5, cross_p=0.5)
    assert metrics.model_scores(tied, "m1", "no_thinking")["wins_per_dim"]["p_solve"] == pytest.approx(0.5)


def test_model_scores_has_no_normalised_aggregates():
    # §15.7 retired skill_advantage / s_d² / floored_dims entirely.
    data = build_data(self_p=0.8, cross_p=0.3)
    s = metrics.model_scores(data, "m1", "no_thinking")
    for k in ("IAI", "SFS_self", "SFS_cross", "skill_advantage_per_dim",
              "iai_per_dim", "sfs_self_per_dim", "sfs_cross_per_dim", "floored_dims"):
        assert k not in s
    # but wins + native-unit per-dim breakdowns are present
    assert "wins" in s
    assert "raw_iai_per_dim" in s
    assert "mean_loss_self_per_dim" in s and "mean_loss_cross_per_dim" in s


# --------------------------------------------------------------------------
# Missing data (sec 9.6)
# --------------------------------------------------------------------------
def test_cell_below_valid_threshold_is_dropped():
    # m1/t1 has only 4 valid rollouts (< 5) -> that task drops for m1.
    data = build_data(self_p=0.8, cross_p=0.3, n_valid=4)
    s = metrics.model_scores(data, "m1", "no_thinking")
    assert s["n_tasks_used"]["p_solve"] == 2  # t1 dropped, t2 + t3 remain


def test_broken_forecast_drops_the_task():
    data = build_data(self_p=0.8, cross_p=0.3, broken_self_task="t3")
    s = metrics.model_scores(data, "m1", "no_thinking")
    assert s["n_tasks_used"]["p_solve"] == 2  # t3 self forecast broken -> dropped


# --------------------------------------------------------------------------
# Cross-predictor designation (sec 8) -- unchanged
# --------------------------------------------------------------------------
def test_designate_cross_predictor_picks_highest_solve_rate():
    outcomes = {}
    rates = {"m1": 0.2, "m2": 0.9, "m3": 0.5}
    for m, rate in rates.items():
        outcomes[m] = {}
        for t in ("t1", "t2"):
            solved = [1.0] * round(rate * N_ROLL) + [0.0] * (N_ROLL - round(rate * N_ROLL))
            outcomes[m][t] = [
                Outcome(m, t, i, {"p_solve": solved[i], "p_first_attempt_correct": 0.0,
                                  "total_expected_steps": 10.0})
                for i in range(N_ROLL)
            ]
    assert metrics.designate_cross_predictor(outcomes, ["m1", "m2", "m3"]) == "m2"


def test_designate_cross_predictor_tiebreak_on_first_attempt():
    outcomes = {}
    for m, fac in (("m1", 0.1), ("m2", 0.8)):
        outcomes[m] = {}
        for t in ("t1",):
            outcomes[m][t] = [
                Outcome(m, t, i, {"p_solve": 1.0, "p_first_attempt_correct": fac,
                                  "total_expected_steps": 10.0})
                for i in range(N_ROLL)
            ]
    assert metrics.designate_cross_predictor(outcomes, ["m1", "m2"]) == "m2"


# --------------------------------------------------------------------------
# Bootstrap-compatible recompute
# --------------------------------------------------------------------------
def test_wins_fraction_accepts_resample_indices():
    data = build_data(self_p=0.8, cross_p=0.3)
    point = metrics.wins_fraction(data, "m1", "no_thinking")
    explicit = metrics.wins_fraction(
        data, "m1", "no_thinking",
        task_indices=[0, 1, 2],
        rollout_indices=[list(range(N_ROLL))] * 3,
    )
    assert point == pytest.approx(explicit)


def test_all_model_wins_excludes_cross_and_models_without_self_forecasts():
    data = build_data(self_p=0.8, cross_p=0.3)
    wins = metrics.all_model_wins(data, "no_thinking")
    assert "m3" not in wins                 # cross-predictor receives no wins score
    assert set(wins) == {"m1", "m2"}        # both have self-forecasts
