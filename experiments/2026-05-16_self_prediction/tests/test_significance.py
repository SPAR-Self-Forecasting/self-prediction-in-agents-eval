"""Tests for significance.py -- the task-bootstrap inference tier.

Four layers:
  1. cell assembly (same retention / broken-forecast gates as metrics + murphy);
  2. the constant baseline, including the identity that ties the binary version
     to Murphy's UNC term;
  3. the interval and multiplicity machinery (coverage direction, BH, floors);
  4. report structure and the invariants the paper quotes off it.
"""

import json

import numpy as np
import pytest

import config
import metrics
import murphy
import significance
from src.types import Dimension, Forecast, Outcome

RNG = np.random.default_rng(42)

DIMS = (Dimension("p_solve", "binary"), Dimension("avg_tokens_step", "continuous"))
M = "qwen/qwen3-8b"
M2 = "z-ai/glm-4.5"
TASKS = ["t0", "t1", "t2", "t3"]


def _outcome_cell(model, task, solves, tokens=100.0, n=8, missing_from=None):
    cell = []
    for i in range(n):
        invalid = missing_from is not None and i >= missing_from
        cell.append(Outcome(
            model, task, i,
            {"p_solve": None if invalid else float(solves[i % len(solves)]),
             "avg_tokens_step": None if invalid else float(tokens)},
        ))
    return cell


def _fc(model, task, cond, p_solve, tokens=100.0, broken=False):
    preds = {} if broken else {"p_solve": p_solve, "avg_tokens_step": tokens}
    return Forecast(model, task, cond, "self", preds, broken=broken)


@pytest.fixture
def tiny_data() -> metrics.ScoringData:
    """4 tasks: t0 normal, t1 fails retention (<5 valid), t2 broken self
    forecast, t3 normal with two self samples (aggregation check)."""
    outcomes = {M: {
        "t0": _outcome_cell(M, "t0", [1, 0, 1, 0, 1, 0, 1, 0]),
        "t1": _outcome_cell(M, "t1", [1] * 8, missing_from=4),   # 4 valid < 5
        "t2": _outcome_cell(M, "t2", [1] * 8),
        "t3": _outcome_cell(M, "t3", [1, 1, 1, 1, 1, 1, 0, 0], missing_from=6),
    }}
    self_fc = {M: {
        "t0": {c: [_fc(M, "t0", c, 0.6)] for c in config.CONDITIONS},
        "t1": {c: [_fc(M, "t1", c, 0.6)] for c in config.CONDITIONS},
        "t2": {c: [_fc(M, "t2", c, 0.6, broken=True)] for c in config.CONDITIONS},
        "t3": {c: [_fc(M, "t3", c, 0.4), _fc(M, "t3", c, 0.8)]
               for c in config.CONDITIONS},
    }}
    return metrics.ScoringData(
        dimensions=DIMS, models=[M], families={"Qwen": [M]},
        tasks=TASKS, outcomes=outcomes, self_forecasts=self_fc,
    )


def _grid_data(n_tasks=20, bias=0.4, seed=0) -> metrics.ScoringData:
    """Two models over ``n_tasks`` clean tasks, each over-predicting p_solve by
    roughly ``bias``. Big enough for the bootstrap to have something to say."""
    rng = np.random.default_rng(seed)
    tasks = [f"t{i}" for i in range(n_tasks)]
    outcomes, forecasts = {}, {}
    for m in (M, M2):
        outcomes[m], forecasts[m] = {}, {}
        for t in tasks:
            rate = float(rng.random() * 0.4)
            solves = (rng.random(8) < rate).astype(float)
            outcomes[m][t] = [
                Outcome(m, t, i, {"p_solve": float(solves[i]),
                                  "avg_tokens_step": 100.0 + 10.0 * i})
                for i in range(8)
            ]
            f = min(1.0, solves.mean() + bias)
            forecasts[m][t] = {c: [_fc(m, t, c, f, tokens=120.0)]
                               for c in config.CONDITIONS}
    return metrics.ScoringData(
        dimensions=DIMS, models=[M, M2],
        families={"Qwen": [M], "Zhipu": [M2]},
        tasks=tasks, outcomes=outcomes, self_forecasts=forecasts,
    )


# --------------------------------------------------------------------------
# 1. Cell assembly
# --------------------------------------------------------------------------
def test_collect_applies_retention_and_broken_forecast_gates(tiny_data):
    table = significance.collect(tiny_data, M, DIMS[0], "no_thinking")
    # t1 dropped (retention), t2 dropped (broken self forecast) -> t0 + t3.
    assert table.tasks == ["t0", "t3"]
    assert table.n == 2


def test_collect_forecast_is_the_mean_over_samples(tiny_data):
    table = significance.collect(tiny_data, M, DIMS[0], "no_thinking")
    # t3's two samples (0.4, 0.8) aggregate to 0.6.
    assert table.forecast[table.tasks.index("t3")] == pytest.approx(0.6)


def test_collect_masks_padding_so_moments_use_only_valid_rollouts(tiny_data):
    table = significance.collect(tiny_data, M, DIMS[0], "no_thinking")
    t3 = table.tasks.index("t3")
    assert table.valid[t3].sum() == 6           # t3 keeps 6 of its 8 rollouts
    assert table.out_mean[t3] == pytest.approx(1.0)   # all six valid ones solved
    assert table.out_sq_mean[t3] == pytest.approx(1.0)


def test_collect_loss_matches_the_pipeline_per_cell_loss(tiny_data):
    """The whole point of the module is to put intervals on the *same* numbers
    analysis.py reports, so the per-cell losses must agree exactly."""
    for dim in DIMS:
        table = significance.collect(tiny_data, M, dim, "no_thinking")
        expected = metrics.model_scores(
            tiny_data, M, "no_thinking", dimensions=(dim,))["mean_loss_per_dim"][dim.name]
        assert float(table.loss.mean()) == pytest.approx(expected, abs=1e-12)


def test_collect_returns_none_when_nothing_is_scored(tiny_data):
    empty = metrics.ScoringData(
        dimensions=DIMS, models=[M], families={"Qwen": [M]}, tasks=TASKS,
        outcomes=tiny_data.outcomes,
        self_forecasts={M: {t: {c: [] for c in config.CONDITIONS} for t in TASKS}},
    )
    assert significance.collect(empty, M, DIMS[0], "no_thinking") is None


# --------------------------------------------------------------------------
# 2. The constant baseline
# --------------------------------------------------------------------------
def test_binary_constant_baseline_equals_murphy_uncertainty():
    """A binary constant baseline is Murphy's UNC term, so skill_vs_constant is
    REL - RES. If this identity breaks the paper's wording breaks with it."""
    data = _grid_data(seed=3)
    dim = DIMS[0]
    for m in (M, M2):
        table = significance.collect(data, m, dim, "no_thinking")
        base = significance.constant_baseline_loss(table, dim)
        f, o, w, _ = murphy.binary_events(data, m, dim, "no_thinking")
        comp = murphy.corp_decomposition(f, o, w)
        assert base == pytest.approx(comp["uncertainty"], abs=1e-9)

        skill = float(table.loss.mean()) - base
        assert skill == pytest.approx(comp["reliability"] - comp["resolution"], abs=1e-9)


def test_continuous_constant_baseline_uses_the_forecast_free_constant():
    data = _grid_data(seed=5)
    dim = DIMS[1]
    table = significance.collect(data, M, dim, "no_thinking")
    const = float(table.out_mean.mean())
    manual = float(np.mean([
        np.abs(const - table.outcomes[i][table.valid[i]]).mean()
        for i in range(table.n)
    ]))
    assert significance.constant_baseline_loss(table, dim) == pytest.approx(manual)


def test_constant_baseline_is_vectorised_consistently_over_draws():
    """The bootstrap path (rows given) and the observed path must agree when the
    rows are the identity -- otherwise every interval is centred on the wrong
    statistic."""
    data = _grid_data(seed=11)
    for dim in DIMS:
        table = significance.collect(data, M, dim, "no_thinking")
        rows = np.tile(np.arange(table.n), (3, 1))
        vec = significance.constant_baseline_loss(table, dim, rows)
        assert np.allclose(vec, significance.constant_baseline_loss(table, dim))


# --------------------------------------------------------------------------
# 3. Intervals and multiplicity
# --------------------------------------------------------------------------
def test_bias_interval_recovers_a_planted_overprediction():
    data = _grid_data(bias=0.4, seed=7)
    stats = significance.model_dimension_stats(data, M, DIMS[0], "no_thinking", b=2000)
    lo, hi = stats["bias"]["ci95"]
    assert stats["bias"]["estimate"] > 0
    assert lo > 0 and stats["bias"]["excludes_zero"]
    assert lo <= stats["bias"]["estimate"] <= hi


def test_no_bias_gives_an_interval_that_contains_zero():
    data = _grid_data(bias=0.0, seed=9)
    stats = significance.model_dimension_stats(data, M, DIMS[0], "no_thinking", b=2000)
    lo, hi = stats["bias"]["ci95"]
    assert lo <= 0.0 <= hi
    assert not stats["bias"]["excludes_zero"]


def test_p_value_is_floored_at_two_over_b():
    """A percentile bootstrap cannot resolve past 1/B; reporting p = 0 would
    claim precision the method does not have."""
    boot = np.full(500, 5.0)  # every resample far from the null
    assert significance._interval(boot, 5.0)["p"] == pytest.approx(2.0 / 500)


def test_bh_counts_split_by_direction():
    per_model = {
        "a": {"bias": {"estimate": +1.0, "p": 0.001}},
        "b": {"bias": {"estimate": -1.0, "p": 0.002}},
        "c": {"bias": {"estimate": +1.0, "p": 0.90}},
    }
    counts = significance._apply_bh(per_model, ("bias",))
    assert counts == {"n_significant": 2,
                      "n_significant_positive": 1,
                      "n_significant_negative": 1}
    assert per_model["a"]["bias"]["significant"] is True
    assert per_model["c"]["bias"]["significant"] is False
    # BH only ever inflates a p-value
    assert per_model["a"]["bias"]["p_bh"] >= per_model["a"]["bias"]["p"]


def test_bh_tolerates_a_model_with_nothing_to_test():
    assert significance._apply_bh({"a": None}, ("bias",))["n_significant"] == 0


def test_thinking_effect_is_paired_on_shared_tasks_only(tiny_data):
    stats = significance.thinking_effect_stats(tiny_data, M, DIMS[0], b=200)
    # only t0 and t3 survive both conditions, which is under the n>=5 floor
    assert stats is None


def test_thinking_effect_reports_zero_when_conditions_are_identical():
    data = _grid_data(seed=13)  # both conditions get the same forecasts
    stats = significance.thinking_effect_stats(data, M, DIMS[0], b=500)
    assert stats["estimate"] == pytest.approx(0.0)
    assert stats["loss_thinking"] == pytest.approx(stats["loss_no_thinking"])


# --------------------------------------------------------------------------
# 4. Report structure and the invariants the paper quotes
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def report() -> dict:
    return significance.build_report(_grid_data(seed=17), b=500)


def test_report_is_json_serialisable(report):
    json.dumps(report)  # no numpy scalars may leak into the artefact


def test_report_records_how_the_inference_was_done(report):
    method = report["method"]
    assert "task" in method["resampling_unit"]
    assert "never resampled" in method["resampling_unit"]
    assert "Benjamini-Hochberg" in method["multiplicity"]


def test_report_never_resamples_models(report):
    """Models are fixed effects here; the pooled row is the only place model
    identity enters the resampling, and there only as a cluster label."""
    assert set(report["families"]) == set(report["models"])
    pooled = report["conditions"]["no_thinking"]["p_solve"]["pooled_bias"]
    assert "ci95_family_clustered" in pooled and "ci95_task_clustered" in pooled


def test_pooled_reports_the_wider_of_the_two_clusterings(report):
    for cond in config.CONDITIONS:
        for name in ("pooled_bias", "pooled_skill_vs_constant"):
            pooled = report["conditions"][cond]["p_solve"][name]
            widths = [pooled["ci95_task_clustered"], pooled["ci95_family_clustered"]]
            reported = pooled["ci95_reported"]
            assert reported in widths
            assert reported[1] - reported[0] == max(hi - lo for lo, hi in widths)


def test_direction_counts_never_exceed_the_model_count(report):
    for cond in config.CONDITIONS:
        for dim in config.DIMENSION_NAMES:
            blk = report["conditions"][cond].get(dim)
            if blk is None:
                continue
            n = blk["n_models"]
            assert blk["n_models_significantly_over"] + \
                   blk["n_models_significantly_under"] == blk["n_models_bias_significant"]
            assert blk["n_models_bias_significant"] <= n
            assert blk["n_models_rho_defined"] <= n


def test_scaling_permutation_p_is_bounded_below_by_its_own_floor(report):
    fam = report["scaling"]["family_level"]
    assert fam["exact_permutation_p"] >= fam["smallest_attainable_p"] - 1e-12
    assert 0.0 < report["scaling"]["permutation_p_models"] <= 1.0


def test_scaling_intervals_bracket_their_point_estimates(report):
    for entry in report["scaling"]["per_model"].values():
        for key in ("headline_loss", "solve_rate"):
            lo, hi = entry[key]["ci95"]
            assert lo <= entry[key]["estimate"] <= hi


def test_report_is_reproducible_under_a_fixed_seed():
    data = _grid_data(seed=23)
    a = significance.model_dimension_stats(data, M, DIMS[0], "no_thinking", b=400)
    b = significance.model_dimension_stats(data, M, DIMS[0], "no_thinking", b=400)
    assert a["bias"]["ci95"] == b["bias"]["ci95"]
