"""Descriptive readout of the wins-fraction metric (EVAL_SPEC_V1.md sec 15.6).

MVP scope: this module reports **point estimates only**. The inferential
statistics tier -- the hierarchical bootstrap (sec 9.3), BCa/jackknife CIs,
the secondary mean tests, the RQ2 OLS scaling fit, and the full robustness /
sensitivity battery (sec 9.5) -- has been pruned and will be reintroduced once
the primary signal is validated.

What remains, all descriptive:

- **RQ1 Existence**   -- mean ``wins(M, no_thinking)`` across IAI-eligible models
                         (the headline "is it > 0.5?" number, no CI).
- **RQ3 Elicitation** -- mean paired ``Δwins(M) = wins(thinking) − wins(no_thinking)``.
- **Exploratory**     -- per-dimension mean ``raw_iai`` in native units.
- **Missingness**     -- dropped-cell / broken-forecast rates with the sec 9.6 flags.
- **solve_rates**     -- empirical solve rate per model (capability context / plots).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

import config
import metrics
from src.types import Dimension

#: Null reference values, carried through for readability of the point estimate
#: (0.5 = "no advantage" for wins; 0.0 = "no thinking effect" for paired diffs).
_WINS_NULL = 0.5
_DIFF_NULL = 0.0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _iai_eligible_models(data: metrics.ScoringData, models: Sequence[str] | None = None) -> list[str]:
    """Models that receive a score: not the cross-predictor AND have self-forecasts."""
    pool = list(models) if models is not None else list(data.models)
    return [m for m in pool
            if m != data.cross_model and m in data.self_forecasts and data.self_forecasts[m]]


def _nanmean(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _per_model_wins(
    data: metrics.ScoringData, condition: str, dims, models: Sequence[str]
) -> dict[str, float]:
    return {
        m: metrics.wins_fraction(data, m, condition, dimensions=dims)
        for m in models
    }


def _per_model_paired_diff(
    data: metrics.ScoringData, dims, models: Sequence[str]
) -> dict[str, float]:
    """Δwins(M) = wins(thinking) - wins(no_thinking) per model -- the RQ3 estimand."""
    out: dict[str, float] = {}
    for m in models:
        thinking = metrics.wins_fraction(data, m, "thinking", dimensions=dims)
        no_thinking = metrics.wins_fraction(data, m, "no_thinking", dimensions=dims)
        out[m] = thinking - no_thinking
    return out


def solve_rates(data: metrics.ScoringData, models: Sequence[str] | None = None) -> dict[str, float]:
    """Empirical solve rate per IAI-eligible model (sec 7) -- descriptive context."""
    model_list = _iai_eligible_models(data, models)
    return {m: metrics.empirical_solve_rate(data.outcomes, m) for m in model_list}


# --------------------------------------------------------------------------
# RQ1 -- Existence (mean wins, point estimate)
# --------------------------------------------------------------------------
def rq1_existence(
    data: metrics.ScoringData,
    *,
    condition: str = "no_thinking",
    dimensions: tuple[Dimension, ...] | None = None,
    models: Sequence[str] | None = None,
) -> dict:
    """RQ1 (sec 15.6): mean ``wins(M)`` across IAI-eligible models (point estimate)."""
    dims = dimensions if dimensions is not None else data.dimensions
    model_list = _iai_eligible_models(data, models)
    per_model = _per_model_wins(data, condition, dims, model_list)
    return {
        "estimand": "mean wins(M) across IAI-eligible models",
        "null": _WINS_NULL,
        "condition": condition,
        "n_models": len(model_list),
        "point_estimate": _nanmean(per_model.values()),
        "per_model_wins": per_model,
    }


# --------------------------------------------------------------------------
# RQ3 -- Elicitation (mean paired Δwins, point estimate)
# --------------------------------------------------------------------------
def rq3_elicitation(
    data: metrics.ScoringData,
    *,
    dimensions: tuple[Dimension, ...] | None = None,
    models: Sequence[str] | None = None,
    exclude_models: Sequence[str] = (),
) -> dict:
    """RQ3 (sec 15.6): mean paired ``Δwins(M)`` (point estimate)."""
    dims = dimensions if dimensions is not None else data.dimensions
    model_list = [m for m in _iai_eligible_models(data, models) if m not in set(exclude_models)]
    per_model = _per_model_paired_diff(data, dims, model_list)
    return {
        "estimand": "mean paired Δwins(M) = wins(thinking) - wins(no_thinking)",
        "null": _DIFF_NULL,
        "n_models": len(model_list),
        "excluded_models": list(exclude_models),
        "point_estimate": _nanmean(per_model.values()),
        "per_model_difference": per_model,
    }


# --------------------------------------------------------------------------
# Exploratory: per-dimension raw_IAI in native units (sec 15.7)
# --------------------------------------------------------------------------
def exploratory(data: metrics.ScoringData, *, condition: str = "no_thinking") -> dict:
    """Per-dimension mean ``raw_iai`` (native units) across IAI-eligible models.

    Brier units for binary dims, raw distance units for continuous dims. The
    per-dimension significance testing (one-sample t + BH-FDR, sec 9.4) is part
    of the pruned statistics tier and is not computed here.
    """
    models = _iai_eligible_models(data)
    scores = {m: metrics.model_scores(data, m, condition) for m in models}

    per_dim_raw_mean: dict[str, float] = {}
    for dim in data.dimensions:
        raws = np.array([scores[m]["raw_iai_per_dim"][dim.name] for m in models], dtype=float)
        raws = raws[~np.isnan(raws)]
        per_dim_raw_mean[dim.name] = float(np.mean(raws)) if raws.size else float("nan")

    def _partition_mean(names) -> float:
        return _nanmean(per_dim_raw_mean.get(n, float("nan")) for n in names)

    return {
        "per_dimension_raw_iai_mean": per_dim_raw_mean,
        "partition_capability_mean_raw_iai": _partition_mean(config.CAPABILITY_DIMENSIONS),
        "partition_behavioral_mean_raw_iai": _partition_mean(config.BEHAVIORAL_DIMENSIONS),
    }


def missingness_report(data: metrics.ScoringData) -> dict:
    """Dropped-cell and broken-forecast rates, with the sec 9.6 >20% flags."""
    models = _iai_eligible_models(data)
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

    # Multi-sample (sec 9.6): a cell is "broken" only when *every* one of its
    # N samples is broken -- as long as one sample is usable, the cell is in.
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
                elif not samples:  # no samples on disk -> treated as broken
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
        "rq1_existence": rq1_existence(data),
        "rq3_elicitation": rq3_elicitation(data),
        "exploratory": exploratory(data),
        "missingness": missingness_report(data),
        "solve_rates": solve_rates(data),
    }
