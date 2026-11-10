"""Descriptive readout of self-forecast accuracy (EVAL_SPEC_V1.md sec 15.6).

This module reports **point estimates only**. The inferential tier lives in
``significance.py``, which puts task-bootstrap CIs and BH-corrected tests on the
same quantities; keeping them apart means the descriptive readout stays cheap
and cannot silently acquire a test.

What remains here, all descriptive:

- **RQ1 Accuracy**    -- mean self-forecast loss across models (lower = better).
- **RQ2 Scaling**     -- Spearman rho of self loss vs empirical solve rate.
- **RQ3 Elicitation** -- mean paired Δloss(M) = loss(thinking) − loss(no_thinking).
- **Exploratory**     -- per-dimension mean self loss in native units.
- **Missingness**     -- dropped-cell / broken-forecast rates with the sec 9.6 flags.
- **solve_rates**     -- empirical solve rate per model (capability context / plots).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

import config
import metrics
from src.types import Dimension

#: Null reference values (0.0 = "no thinking effect" for paired diffs).
_DIFF_NULL = 0.0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _eligible_models(data: metrics.ScoringData, models: Sequence[str] | None = None) -> list[str]:
    """Models with self-forecasts available for scoring."""
    pool = list(models) if models is not None else list(data.models)
    return [m for m in pool if m in data.self_forecasts and data.self_forecasts[m]]


def _nanmean(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _per_model_mean_loss_per_dim(
    data: metrics.ScoringData, condition: str, dims, models: Sequence[str]
) -> dict[str, dict[str, float]]:
    return {
        m: metrics.model_scores(data, m, condition, dimensions=dims)["mean_loss_per_dim"]
        for m in models
    }


def _per_model_mean_loss(
    data: metrics.ScoringData, condition: str, dims, models: Sequence[str]
) -> dict[str, float]:
    return {
        m: metrics.model_scores(data, m, condition, dimensions=dims)["mean_loss"]
        for m in models
    }


def _per_model_equal_dimension_loss(
    mlpds: dict[str, dict[str, float]], dim_names: Sequence[str]
) -> dict[str, float]:
    scales = metrics.dimension_loss_scales(mlpds, dim_names)
    return {
        m: metrics.equal_dimension_mean_loss(mlpd, scales, dim_names)
        for m, mlpd in mlpds.items()
    }


def _per_model_paired_diff(
    data: metrics.ScoringData, dims, models: Sequence[str]
) -> dict[str, float]:
    """Δloss(M) = loss(thinking) - loss(no_thinking) per model -- the RQ3 estimand."""
    out: dict[str, float] = {}
    for m in models:
        thinking = metrics.model_scores(data, m, "thinking", dimensions=dims)["mean_loss"]
        no_thinking = metrics.model_scores(data, m, "no_thinking", dimensions=dims)["mean_loss"]
        out[m] = thinking - no_thinking
    return out


def solve_rates(data: metrics.ScoringData, models: Sequence[str] | None = None) -> dict[str, float]:
    """Empirical solve rate per model (sec 7) -- descriptive context."""
    model_list = _eligible_models(data, models)
    return {m: metrics.empirical_solve_rate(data.outcomes, m) for m in model_list}


def _spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank-correlation coefficient (descriptive); NaN if undefined."""
    from scipy.stats import spearmanr

    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    mask = ~(np.isnan(xa) | np.isnan(ya))
    if mask.sum() < 3 or np.unique(xa[mask]).size < 2 or np.unique(ya[mask]).size < 2:
        return float("nan")
    return float(spearmanr(xa[mask], ya[mask])[0])


def rq2_scaling(
    data: metrics.ScoringData,
    *,
    condition: str = "no_thinking",
    dimensions: tuple[Dimension, ...] | None = None,
    models: Sequence[str] | None = None,
) -> dict:
    """RQ2: does self-forecast loss scale with model capability?

    Descriptive Spearman rho between empirical solve rate and equal-dimension
    mean self loss (each dimension scaled by its cross-model peak so token/step
    MAE does not dominate the pooled headline).
    """
    dims = dimensions if dimensions is not None else data.dimensions
    dim_names = [d.name for d in dims]
    model_list = _eligible_models(data, models)
    mlpds = _per_model_mean_loss_per_dim(data, condition, dims, model_list)
    per_model_loss_native = _per_model_mean_loss(data, condition, dims, model_list)
    per_model_loss = _per_model_equal_dimension_loss(mlpds, dim_names)
    scales = metrics.dimension_loss_scales(mlpds, dim_names)
    rates = {m: metrics.empirical_solve_rate(data.outcomes, m) for m in model_list}
    rho = _spearman_rho([rates[m] for m in model_list],
                        [per_model_loss[m] for m in model_list])
    return {
        "estimand": "Spearman rho of equal-dimension mean self loss vs empirical solve rate",
        "condition": condition,
        "n_models": len(model_list),
        "spearman_rho": rho,
        "per_model_loss": per_model_loss,
        "per_model_loss_native": per_model_loss_native,
        "loss_scales": scales,
        "solve_rates": rates,
        "caveat": "descriptive monotonic association; not a significance test",
    }


# --------------------------------------------------------------------------
# RQ1 -- Self-forecast accuracy (mean loss, point estimate)
# --------------------------------------------------------------------------
def rq1_accuracy(
    data: metrics.ScoringData,
    *,
    condition: str = "no_thinking",
    dimensions: tuple[Dimension, ...] | None = None,
    models: Sequence[str] | None = None,
) -> dict:
    """RQ1: mean self-forecast loss across models (point estimate)."""
    dims = dimensions if dimensions is not None else data.dimensions
    model_list = _eligible_models(data, models)
    per_model = _per_model_mean_loss(data, condition, dims, model_list)
    return {
        "estimand": "mean self-forecast loss across models",
        "condition": condition,
        "n_models": len(model_list),
        "point_estimate": _nanmean(per_model.values()),
        "per_model_loss": per_model,
    }


# --------------------------------------------------------------------------
# RQ3 -- Elicitation (mean paired Δloss, point estimate)
# --------------------------------------------------------------------------
def rq3_elicitation(
    data: metrics.ScoringData,
    *,
    dimensions: tuple[Dimension, ...] | None = None,
    models: Sequence[str] | None = None,
    exclude_models: Sequence[str] = (),
) -> dict:
    """RQ3: mean paired Δloss(M) = loss(thinking) - loss(no_thinking) (point estimate)."""
    dims = dimensions if dimensions is not None else data.dimensions
    model_list = [m for m in _eligible_models(data, models) if m not in set(exclude_models)]
    per_model = _per_model_paired_diff(data, dims, model_list)
    return {
        "estimand": "mean paired Δloss(M) = loss(thinking) - loss(no_thinking)",
        "null": _DIFF_NULL,
        "n_models": len(model_list),
        "excluded_models": list(exclude_models),
        "point_estimate": _nanmean(per_model.values()),
        "per_model_difference": per_model,
    }


# --------------------------------------------------------------------------
# Exploratory: per-dimension mean self loss in native units (sec 15.7)
# --------------------------------------------------------------------------
def exploratory(data: metrics.ScoringData, *, condition: str = "no_thinking") -> dict:
    """Per-dimension mean self loss (native units) across models."""
    models = _eligible_models(data)
    scores = {m: metrics.model_scores(data, m, condition) for m in models}

    per_dim_mean: dict[str, float] = {}
    for dim in data.dimensions:
        losses = np.array(
            [scores[m]["mean_loss_per_dim"][dim.name] for m in models], dtype=float,
        )
        losses = losses[~np.isnan(losses)]
        per_dim_mean[dim.name] = float(np.mean(losses)) if losses.size else float("nan")

    def _partition_mean(names) -> float:
        return _nanmean(per_dim_mean.get(n, float("nan")) for n in names)

    return {
        "per_dimension_mean_loss": per_dim_mean,
        "partition_capability_mean_loss": _partition_mean(config.CAPABILITY_DIMENSIONS),
        "partition_behavioral_mean_loss": _partition_mean(config.BEHAVIORAL_DIMENSIONS),
    }


def missingness_report(data: metrics.ScoringData) -> dict:
    """Dropped-cell and broken-forecast rates, with the sec 9.6 >20% flags."""
    models = _eligible_models(data)
    n_tasks = len(data.tasks)

    dropped_cell_rate: dict[str, dict[str, float]] = {}
    flagged_dropped: list[str] = []
    for m in models:
        dropped_cell_rate[m] = {}
        for dim in data.dimensions:
            n_dropped = sum(not data._retained[m][t][dim.name] for t in data.tasks)
            rate = n_dropped / n_tasks
            dropped_cell_rate[m][dim.name] = rate
            if rate > config.FLAG_FRACTION and m not in flagged_dropped:
                flagged_dropped.append(m)

    broken_rate: dict[str, dict[str, float]] = {}
    flagged_broken: list[str] = []
    for m in models:
        broken_rate[m] = {}
        for cond in config.CONDITIONS:
            n_broken = 0
            for t in data.tasks:
                samples = data.self_forecasts.get(m, {}).get(t, {}).get(cond, [])
                if samples and all(s.broken for s in samples):
                    n_broken += 1
                elif not samples:
                    n_broken += 1
            rate = n_broken / n_tasks
            broken_rate[m][cond] = rate
            if rate > config.FLAG_FRACTION and m not in flagged_broken:
                flagged_broken.append(m)

    return {
        "dropped_cell_rate": dropped_cell_rate,
        "broken_forecast_rate": broken_rate,
        "flagged_models_dropped_cells": flagged_dropped,
        "flagged_models_broken_forecasts": flagged_broken,
    }


# --------------------------------------------------------------------------
# Full analysis (descriptive)
# --------------------------------------------------------------------------
def run_full_analysis(data: metrics.ScoringData) -> dict:
    """Assemble the descriptive readout: RQ1 + RQ3 point estimates, exploratory,
    missingness, and per-model solve rates."""
    return {
        "rq1_accuracy": rq1_accuracy(data),
        "rq2_scaling": rq2_scaling(data),
        "rq3_elicitation": rq3_elicitation(data),
        "exploratory": exploratory(data),
        "missingness": missingness_report(data),
        "solve_rates": solve_rates(data),
    }
