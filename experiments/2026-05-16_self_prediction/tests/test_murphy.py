"""Tests for murphy.py -- CORP/binned Murphy decomposition of the binary dims.

Three layers:
  1. the math (PAV, CORP identity + non-negativity, binned identity-with-residual);
  2. event assembly (mirrors metrics._dim_losses gating; cell-equal weighting);
  3. the report invariant: decomposed brier == the pipeline's per-dim mean loss.
"""

import numpy as np
import pytest

import config
import metrics
import murphy
from src.types import Dimension, Forecast, Outcome

RNG = np.random.default_rng(42)


# --------------------------------------------------------------------------
# 1a. PAV
# --------------------------------------------------------------------------
def test_pav_fit_is_monotone_in_forecast_order():
    f = RNG.random(200)
    o = (RNG.random(200) < f).astype(float)
    w = np.ones_like(f)
    fitted = murphy.pav_calibrate(f, o, w)
    order = np.argsort(f, kind="stable")
    assert np.all(np.diff(fitted[order]) >= -1e-12)


def test_pav_ties_in_forecast_get_one_fitted_value():
    f = np.array([0.3, 0.3, 0.3, 0.7])
    o = np.array([0.0, 1.0, 1.0, 1.0])
    w = np.ones(4)
    fitted = murphy.pav_calibrate(f, o, w)
    assert fitted[0] == fitted[1] == fitted[2]


def test_pav_on_already_monotone_data_is_blockwise_mean():
    # Two clean groups, no violation: fit = group means.
    f = np.array([0.2, 0.2, 0.8, 0.8])
    o = np.array([0.0, 0.0, 1.0, 1.0])
    fitted = murphy.pav_calibrate(f, o, np.ones(4))
    assert np.allclose(fitted, [0.0, 0.0, 1.0, 1.0])


def test_pav_pools_violating_blocks_by_weighted_mean():
    # Descending outcomes force one pooled block: weighted mean = (2*1 + 1*0)/3.
    f = np.array([0.2, 0.8])
    o = np.array([1.0, 0.0])
    w = np.array([2.0, 1.0])
    fitted = murphy.pav_calibrate(f, o, w)
    assert np.allclose(fitted, [2.0 / 3.0, 2.0 / 3.0])


def test_pav_never_increases_brier():
    for seed in range(5):
        rng = np.random.default_rng(seed)
        f = rng.random(300)
        o = (rng.random(300) < 0.4).astype(float)
        w = rng.uniform(0.5, 2.0, 300)
        fitted = murphy.pav_calibrate(f, o, w)
        bs_f = np.sum(w * (f - o) ** 2) / np.sum(w)
        bs_r = np.sum(w * (fitted - o) ** 2) / np.sum(w)
        assert bs_r <= bs_f + 1e-12


# --------------------------------------------------------------------------
# 1b. CORP decomposition
# --------------------------------------------------------------------------
def test_corp_identity_exact_random_data():
    for seed in range(10):
        rng = np.random.default_rng(seed)
        f = rng.random(250)
        o = (rng.random(250) < f).astype(float)
        w = rng.uniform(0.2, 3.0, 250)
        c = murphy.corp_decomposition(f, o, w)
        assert c["brier"] == pytest.approx(
            c["reliability"] - c["resolution"] + c["uncertainty"], abs=1e-12)


def test_corp_components_nonnegative():
    for seed in range(10):
        rng = np.random.default_rng(seed)
        f = rng.random(250)
        o = (rng.random(250) < 0.3).astype(float)
        c = murphy.corp_decomposition(f, o)
        assert c["reliability"] >= -1e-12
        assert c["resolution"] >= -1e-12
        assert c["uncertainty"] >= 0.0


def test_corp_constant_forecast_has_zero_resolution():
    o = (RNG.random(200) < 0.4).astype(float)
    f = np.full_like(o, 0.7)
    c = murphy.corp_decomposition(f, o)
    assert c["resolution"] == pytest.approx(0.0, abs=1e-12)
    # ...and all error above the UNC floor is miscalibration.
    assert c["reliability"] == pytest.approx(c["brier"] - c["uncertainty"], abs=1e-12)


def test_corp_perfect_forecaster_is_pure_resolution():
    o = (RNG.random(200) < 0.5).astype(float)
    c = murphy.corp_decomposition(o.copy(), o)
    assert c["brier"] == pytest.approx(0.0, abs=1e-12)
    assert c["reliability"] == pytest.approx(0.0, abs=1e-12)
    assert c["resolution"] == pytest.approx(c["uncertainty"], abs=1e-12)


def test_corp_uncertainty_is_base_rate_variance():
    o = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    f = np.full(5, 0.5)
    c = murphy.corp_decomposition(f, o)
    p = 2.0 / 5.0
    assert c["base_rate"] == pytest.approx(p)
    assert c["uncertainty"] == pytest.approx(p * (1 - p))


def test_corp_degenerate_outcomes_zero_uncertainty():
    o = np.ones(50)
    f = RNG.random(50)
    c = murphy.corp_decomposition(f, o)
    assert c["uncertainty"] == 0.0
    assert c["resolution"] == pytest.approx(0.0, abs=1e-12)
    assert c["reliability"] == pytest.approx(c["brier"], abs=1e-12)


def test_corp_empty_events_gives_nan_components():
    c = murphy.corp_decomposition([], [])
    assert c["n_events"] == 0
    assert np.isnan(c["brier"]) and np.isnan(c["reliability"])


def test_corp_rejects_non_binary_outcomes():
    with pytest.raises(ValueError, match="binary"):
        murphy.corp_decomposition([0.5, 0.5], [0.0, 0.4])


def test_corp_weights_change_the_answer():
    # Weighting must actually be applied, not ignored.
    f = np.array([0.1, 0.9])
    o = np.array([1.0, 0.0])
    c1 = murphy.corp_decomposition(f, o, np.array([1.0, 1.0]))
    c2 = murphy.corp_decomposition(f, o, np.array([10.0, 1.0]))
    assert c1["base_rate"] != c2["base_rate"]


# --------------------------------------------------------------------------
# 1c. Binned decomposition
# --------------------------------------------------------------------------
def test_binned_identity_holds_with_residual():
    rng = np.random.default_rng(7)
    f = rng.random(400)
    o = (rng.random(400) < f).astype(float)
    w = rng.uniform(0.5, 2.0, 400)
    b = murphy.binned_decomposition(f, o, w, n_bins=10)
    brier = np.sum(w * (f - o) ** 2) / np.sum(w)
    assert brier == pytest.approx(
        b["reliability"] - b["resolution"] + b["uncertainty"] + b["residual"], abs=1e-12)


def test_binned_bin_table_covers_all_events():
    f = RNG.random(100)
    o = (RNG.random(100) < 0.5).astype(float)
    b = murphy.binned_decomposition(f, o, n_bins=10)
    assert sum(bn["n_events"] for bn in b["bins"]) == 100
    for bn in b["bins"]:
        assert bn["lo"] <= bn["mean_forecast"] <= bn["hi"] + 1e-12
        assert 0.0 <= bn["observed_rate"] <= 1.0


def test_binned_quantile_strategy_handles_heavy_ties():
    f = np.array([0.5] * 50 + [0.9] * 5)
    o = (RNG.random(55) < 0.5).astype(float)
    b = murphy.binned_decomposition(f, o, n_bins=10, strategy="quantile")
    assert sum(bn["n_events"] for bn in b["bins"]) == 55


def test_binned_forecast_exactly_one_is_kept():
    # f == 1.0 must land in the last bin, not be clipped out.
    f = np.array([1.0, 1.0, 0.05])
    o = np.array([1.0, 1.0, 0.0])
    b = murphy.binned_decomposition(f, o, n_bins=10)
    assert sum(bn["n_events"] for bn in b["bins"]) == 3


# --------------------------------------------------------------------------
# 2. Event assembly -- a small hand-built ScoringData exercising every gate
# --------------------------------------------------------------------------
DIMS = (Dimension("p_solve", "binary"), Dimension("avg_tokens_step", "continuous"))
M_SELF, M_CROSS = "m/self", "m/cross"
TASKS = ["t0", "t1", "t2", "t3"]


def _outcome_cell(model, task, solves, n=8, missing_from=None):
    """8 rollouts; `solves[i]` gives p_solve; indices >= missing_from are invalid."""
    cell = []
    for i in range(n):
        v = None if (missing_from is not None and i >= missing_from) else float(solves[i % len(solves)])
        cell.append(Outcome(model, task, i,
                            {"p_solve": v, "avg_tokens_step": 100.0}))
    return cell


def _fc(model, task, cond, kind, p_solve, broken=False):
    preds = {} if broken else {"p_solve": p_solve, "avg_tokens_step": 100.0}
    return Forecast(model, task, cond, kind, preds, broken=broken)


@pytest.fixture
def tiny_data() -> metrics.ScoringData:
    """4 tasks: t0 normal, t1 fails retention (<5 valid), t2 broken self
    forecast, t3 normal with two self samples (aggregation check)."""
    outcomes = {
        M_SELF: {
            "t0": _outcome_cell(M_SELF, "t0", [1, 0, 1, 0, 1, 0, 1, 0]),
            "t1": _outcome_cell(M_SELF, "t1", [1] * 8, missing_from=4),  # 4 valid < 5
            "t2": _outcome_cell(M_SELF, "t2", [1] * 8),
            "t3": _outcome_cell(M_SELF, "t3", [1, 1, 1, 1, 1, 1, 0, 0], missing_from=6),  # 6 valid
        },
        M_CROSS: {t: _outcome_cell(M_CROSS, t, [1] * 8) for t in TASKS},
    }
    self_fc = {M_SELF: {
        "t0": {c: [_fc(M_SELF, "t0", c, "self", 0.6)] for c in config.CONDITIONS},
        "t1": {c: [_fc(M_SELF, "t1", c, "self", 0.6)] for c in config.CONDITIONS},
        "t2": {c: [_fc(M_SELF, "t2", c, "self", 0.6, broken=True)] for c in config.CONDITIONS},
        "t3": {c: [_fc(M_SELF, "t3", c, "self", 0.4),
                   _fc(M_SELF, "t3", c, "self", 0.8)] for c in config.CONDITIONS},
    }}
    cross_fc = {t: {c: [_fc(M_CROSS, t, c, "cross", 0.5)] for c in config.CONDITIONS}
                for t in TASKS}
    return metrics.ScoringData(
        dimensions=DIMS, models=[M_SELF, M_CROSS], families={"F": [M_SELF, M_CROSS]},
        tasks=TASKS, cross_model=M_CROSS, outcomes=outcomes,
        self_forecasts=self_fc, cross_forecasts=cross_fc,
    )


def test_events_apply_retention_and_broken_forecast_gates(tiny_data):
    dim = DIMS[0]
    f, o, w, n_cells = murphy.binary_events(tiny_data, M_SELF, dim, "no_thinking", "self")
    # t1 dropped (retention), t2 dropped (broken self forecast) -> t0 + t3.
    assert n_cells == 2
    assert len(f) == 8 + 6  # t0: 8 valid rollouts, t3: 6 valid rollouts


def test_events_weighting_makes_cells_equal(tiny_data):
    dim = DIMS[0]
    f, o, w, n_cells = murphy.binary_events(tiny_data, M_SELF, dim, "no_thinking", "self")
    assert np.sum(w) == pytest.approx(n_cells)  # each cell contributes weight 1
    uniq = np.unique(w)
    assert len(uniq) == 2
    assert np.allclose(sorted(uniq), [1 / 8, 1 / 6])


def test_events_forecast_is_mean_over_samples(tiny_data):
    dim = DIMS[0]
    f, o, w, _ = murphy.binary_events(tiny_data, M_SELF, dim, "no_thinking", "self")
    # t3's two samples (0.4, 0.8) aggregate to 0.6 (Option A mean).
    t3_forecasts = f[-6:]
    assert np.allclose(t3_forecasts, 0.6)


def test_events_same_cells_for_self_and_cross(tiny_data):
    dim = DIMS[0]
    fs, os_, ws, ns = murphy.binary_events(tiny_data, M_SELF, dim, "no_thinking", "self")
    fc, oc, wc, nc = murphy.binary_events(tiny_data, M_SELF, dim, "no_thinking", "cross")
    assert ns == nc
    assert np.array_equal(os_, oc)   # identical outcomes...
    assert np.array_equal(ws, wc)    # ...and weights; only the forecast differs
    assert np.allclose(fc, 0.5)


def test_events_brier_matches_metrics_mean_loss(tiny_data):
    """The invariant: cell-equal weighting reproduces the pipeline's per-dim loss."""
    dim = DIMS[0]
    for kind, key in (("self", "mean_loss_self_per_dim"), ("cross", "mean_loss_cross_per_dim")):
        f, o, w, _ = murphy.binary_events(tiny_data, M_SELF, dim, "no_thinking", kind)
        c = murphy.corp_decomposition(f, o, w)
        expected = metrics.model_scores(tiny_data, M_SELF, "no_thinking")[key]["p_solve"]
        assert c["brier"] == pytest.approx(expected, abs=1e-12)


def test_events_rejects_unknown_kind(tiny_data):
    with pytest.raises(ValueError, match="kind"):
        murphy.binary_events(tiny_data, M_SELF, DIMS[0], "no_thinking", "other")


# --------------------------------------------------------------------------
# 3. Report structure + invariants on a fuller synthetic ScoringData
# --------------------------------------------------------------------------
def test_report_structure_and_uncertainty_agreement(tiny_data):
    report = murphy.murphy_report(tiny_data, n_bins=5)
    assert report["cross_model"] == M_CROSS
    assert report["models"] == [M_SELF]
    for cond in config.CONDITIONS:
        block = report["conditions"][cond]["p_solve"]
        entry = block["per_model"][M_SELF]
        # self and cross decompose over identical events -> identical UNC/base rate.
        assert entry["self"]["uncertainty"] == pytest.approx(entry["cross"]["uncertainty"], abs=1e-12)
        assert entry["self"]["base_rate"] == pytest.approx(entry["cross"]["base_rate"], abs=1e-12)
        # CORP identity survives the report plumbing.
        for kind in ("self", "cross"):
            comp = entry[kind]
            assert comp["brier"] == pytest.approx(
                comp["reliability"] - comp["resolution"] + comp["uncertainty"], abs=1e-12)
        assert block["mean_across_models"]["n_models"] == 1
        assert "caveat" in block["pooled"]


def test_report_covers_only_binary_dimensions(tiny_data):
    report = murphy.murphy_report(tiny_data, n_bins=5, dimensions=murphy.BINARY_DIMENSIONS[:1])
    assert list(report["conditions"]["no_thinking"].keys()) == ["p_solve"]


def test_report_is_json_serialisable(tiny_data):
    import json
    report = murphy.murphy_report(tiny_data, n_bins=5)
    json.dumps(report)  # must not raise (all floats/ints/strs, no numpy types)
