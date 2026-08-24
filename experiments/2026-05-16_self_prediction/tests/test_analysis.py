"""Tests for analysis.py -- descriptive self-forecast loss readout (sec 15.6).

The inferential statistics tier (bootstrap / CIs / RQ2 fit / robustness) is
pruned for the MVP, so these tests assert the point-estimate contract only.
"""

import hashlib
import random
from functools import lru_cache

import analysis
import config
import metrics
from src.types import Forecast, Outcome


# --------------------------------------------------------------------------
# Synthetic ScoringData
# --------------------------------------------------------------------------
def _rng(*parts) -> random.Random:
    key = "|".join(str(p) for p in parts)
    return random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big"))


def _cell_truth(dim, skill, model, task) -> float:
    r = _rng("truth", dim.name, model, task)
    if dim.name == "p_solve":
        return float(r.random() < skill)                # solve rate tracks capability
    if dim.is_binary:
        return float(r.random() < 0.5)
    if dim.name in ("avg_tokens_step", "total_expected_steps"):
        return float(r.uniform(20.0, 400.0))             # heavy-tailed counts
    return float(r.uniform(0.05, 0.95))                  # explore_edit_ratio


def _clamp(dim, value) -> float:
    if dim.is_binary or dim.name == "explore_edit_ratio":
        return min(1.0, max(0.0, value))
    return max(0.0, value)


@lru_cache(maxsize=None)
def synth_data(n_tasks: int = 8) -> metrics.ScoringData:
    """A ScoringData where self forecasts track each cell's truth with error ~ (1 - skill).

    Lower loss for more capable models (RQ2), and the thinking condition (smaller
    error) lowers loss (RQ3).
    """
    dims = config.DIMENSIONS
    models = list(config.MODEL_IDS)
    families = config.families()
    tasks = [f"task{i}" for i in range(n_tasks)]

    outcomes: dict = {}
    self_fc: dict = {}
    for m in models:
        skill = config.model_by_id(m).est_swebench
        outcomes[m] = {}
        self_fc[m] = {}
        for t in tasks:
            cell = []
            truths = {d.name: _cell_truth(d, skill, m, t) for d in dims}
            for i in range(config.N_ROLLOUTS):
                values = {}
                for d in dims:
                    if d.is_binary:
                        values[d.name] = truths[d.name]
                    else:
                        jitter = _rng("roll", d.name, m, t, i).uniform(-0.04, 0.04)
                        values[d.name] = max(0.0, truths[d.name] * (1.0 + jitter))
                cell.append(Outcome(m, t, i, values))
            outcomes[m][t] = cell
            self_fc[m][t] = {}
            for cond in config.CONDITIONS:
                err_scale = (1.0 - skill) * (0.5 if cond == "no_thinking" else 0.28)
                preds = {}
                for d in dims:
                    e = _rng("self", d.name, m, t, cond).uniform(-1, 1) * err_scale
                    base = truths[d.name]
                    preds[d.name] = _clamp(d, base + e if d.is_binary else base * (1.0 + e))
                self_fc[m][t][cond] = [Forecast(m, t, cond, "self", preds, sample_idx=0)]

    return metrics.ScoringData(
        dimensions=dims, models=models, families=families, tasks=tasks,
        outcomes=outcomes, self_forecasts=self_fc,
    )


# --------------------------------------------------------------------------
# RQ1 -- mean self loss (point estimate)
# --------------------------------------------------------------------------
def test_rq1_reports_mean_loss():
    res = analysis.rq1_accuracy(synth_data())
    assert res["n_models"] == len(config.MODEL_IDS)
    assert res["point_estimate"] > 0.0
    assert len(res["per_model_loss"]) == len(config.MODEL_IDS)


def test_rq1_is_deterministic():
    data = synth_data()
    a = analysis.rq1_accuracy(data)
    b = analysis.rq1_accuracy(data)
    assert a["point_estimate"] == b["point_estimate"]


# --------------------------------------------------------------------------
# RQ2 -- loss ~ solve_rate scaling (descriptive Spearman rho)
# --------------------------------------------------------------------------
def test_rq2_scaling_descriptive_spearman():
    res = analysis.rq2_scaling(synth_data())
    assert res["n_models"] == len(config.MODEL_IDS)
    assert -1.0 <= res["spearman_rho"] <= 1.0
    assert res["spearman_rho"] < 0                     # higher capability -> lower loss
    assert len(res["solve_rates"]) == len(config.MODEL_IDS)
    assert len(res["per_model_loss"]) == len(config.MODEL_IDS)
    # the statistics tier stays pruned: no fitted slope, CI, or p-value
    for absent in ("linear_coef", "ci_lo", "ci_hi", "bootstrap_p"):
        assert absent not in res


# --------------------------------------------------------------------------
# RQ3 -- Δloss (point estimate)
# --------------------------------------------------------------------------
def test_rq3_detects_thinking_benefit():
    res = analysis.rq3_elicitation(synth_data())
    assert res["null"] == 0.0
    assert res["point_estimate"] < 0                   # thinking lowers self error -> negative Δloss
    assert len(res["per_model_difference"]) == len(config.MODEL_IDS)


def test_rq3_excludes_models():
    res = analysis.rq3_elicitation(
        synth_data(), exclude_models=config.THINKING_SENSITIVITY_EXCLUDE,
    )
    assert res["n_models"] == len(config.MODEL_IDS) - len(config.THINKING_SENSITIVITY_EXCLUDE)


# --------------------------------------------------------------------------
# Solve rates + missingness
# --------------------------------------------------------------------------
def test_solve_rates_cover_all_models():
    rates = analysis.solve_rates(synth_data())
    assert len(rates) == len(config.MODEL_IDS)
    assert all(0.0 <= r <= 1.0 for r in rates.values())


def test_missingness_report_clean_data():
    rep = analysis.missingness_report(synth_data())
    assert rep["flagged_models_dropped_cells"] == []      # synth data is fully valid
    assert rep["flagged_models_broken_forecasts"] == []
    assert len(rep["dropped_cell_rate"]) == len(config.MODEL_IDS)


# --------------------------------------------------------------------------
# Full analysis (end-to-end shape)
# --------------------------------------------------------------------------
def test_run_full_analysis_runs_end_to_end():
    out = analysis.run_full_analysis(synth_data())
    for key in ("rq1_accuracy", "rq2_scaling", "rq3_elicitation", "exploratory",
                "missingness", "solve_rates"):
        assert key in out
    # rq2 is descriptive (Spearman rho); the inferential robustness battery stays pruned
    assert "spearman_rho" in out["rq2_scaling"]
    assert "robustness" not in out
    # per-dim breakdown covers all 10 dimensions in native units (A6)
    assert len(out["exploratory"]["per_dimension_mean_loss"]) == 10
