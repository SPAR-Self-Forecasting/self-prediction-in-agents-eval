"""Scoring aggregates for Experiment 1 (EVAL_SPEC_V1.md sec 15.6 + sec 15.7).

The metric is layered, each layer reported in its own units (no cross-dim
normalisation):

- **Headline:** ``wins(M, condition)`` -- the fraction of ``(task, dimension)``
  cells where ``loss_self < loss_cross``. Bounded [0, 1]; ties = 0.5.
  Equivalent to HELM's Mean Win Rate (Liang et al. 2022); answers RQ1/RQ2/RQ3.
- **Per-dimension transparency:** ``mean_loss_self``, ``mean_loss_cross``,
  ``raw_iai = mean_loss_cross − mean_loss_self`` -- all in the dimension's
  native units (Brier units for binary, raw distance units for continuous).

Per-cell loss is selected by ``Dimension.kind`` (sec 15.7): Brier for binary,
MAE for continuous. No log transform, no ``s_d²`` normalisation, no W3 floor --
the entire normalisation machinery exists only when one needs to *average*
heterogeneous per-dim scores, and the wins-fraction handles that by binarising
per cell.

**Known caveat (sec 15.4 W2, documented-only).** Each forecast is one sample
so ``loss = bias² + Var(forecast)``; self and cross forecasters have different
variances and the ``Var`` term does not fully cancel. Weaker / more-erratic
forecasters carry an inflated ``loss(self)`` and win fewer cells. See
``analysis.rq2_scaling`` for the reporting caveat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from src.scoring import per_forecast_loss
from src.types import Dimension, Forecast, Outcome


# --------------------------------------------------------------------------
# The scoring data container
# --------------------------------------------------------------------------
@dataclass
class ScoringData:
    """Everything the metrics need, indexed for fast (re)computation.

    Holds the raw data only -- no per-dim normalisation constants live here
    (sec 15.7 retired ``s_d`` and ``floored_dims``).
    """

    dimensions: tuple[Dimension, ...]
    models: list[str]
    families: dict[str, list[str]]
    tasks: list[str]
    cross_model: str
    outcomes: dict[str, dict[str, list[Outcome]]]
    # Multi-sample (sec 9.6): inner condition value is a *list* of N Forecast
    # samples (config.N_FORECAST_SAMPLES). Broken samples are kept in the list
    # and skipped by _aggregate_predictions / _sample_variance.
    self_forecasts: dict[str, dict[str, dict[str, list[Forecast]]]]
    cross_forecasts: dict[str, dict[str, list[Forecast]]]

    def __post_init__(self) -> None:
        first = self.outcomes[self.models[0]][self.tasks[0]]
        self.n_rollouts = len(first)
        # retention is a fixed data-quality decision on the original data (sec 9.6).
        self._retained: dict[str, dict[str, dict[str, bool]]] = {}
        for model in self.models:
            self._retained[model] = {}
            for task in self.tasks:
                cell = self.outcomes[model][task]
                self._retained[model][task] = {
                    d.name: sum(o.valid_for(d.name) for o in cell) >= config.MIN_VALID_ROLLOUTS
                    for d in self.dimensions
                }


# --------------------------------------------------------------------------
# Per-cell helpers
# --------------------------------------------------------------------------
def _valid_subset(cell: list[Outcome], dim_name: str, rollout_idx_subset) -> list[float]:
    """The ground-truth values of the (resampled) rollouts valid for a dimension."""
    return [
        cell[i].values[dim_name]
        for i in rollout_idx_subset
        if cell[i].valid_for(dim_name)
    ]


def _usable_samples(samples: list[Forecast] | None, dim_name: str) -> list[Forecast]:
    """Non-broken samples that carry a numeric value for ``dim_name`` (multi-sample, sec 9.6)."""
    if not samples:
        return []
    return [s for s in samples if (s is not None and not s.broken and dim_name in s.predictions)]


def _aggregate_prediction(samples: list[Forecast] | None, dim_name: str) -> float | None:
    """Option-A aggregation (sec 15.6): mean prediction across non-broken samples.

    Returns None when every sample for the cell is broken / missing the dim --
    in which case the caller treats the cell as missing on that dim.
    """
    usable = _usable_samples(samples, dim_name)
    if not usable:
        return None
    return float(np.mean([s.predictions[dim_name] for s in usable]))


def _sample_variance(samples: list[Forecast] | None, dim_name: str) -> float | None:
    """Within-cell sample variance for ``dim_name`` (sec 15.7 multi-sample diagnostic).

    Sample (Bessel-corrected) variance over the non-broken samples that carry
    the dim. None when fewer than 2 samples are usable.
    """
    usable = _usable_samples(samples, dim_name)
    if len(usable) < 2:
        return None
    return float(np.var([s.predictions[dim_name] for s in usable], ddof=1))


def _has_self_forecasts(data: ScoringData, model: str) -> bool:
    """Whether self-forecasts were collected for ``model`` (the cross-predictor has none)."""
    return model in data.self_forecasts and bool(data.self_forecasts[model])


# --------------------------------------------------------------------------
# Per-dimension losses (used by both wins and per-dim reporting)
# --------------------------------------------------------------------------
def _dim_losses(
    data: ScoringData,
    model: str,
    dim: Dimension,
    condition: str,
    task_indices,
    rollout_indices,
) -> list[tuple[float, float]]:
    """List of ``(loss_self, loss_cross)`` pairs over the usable task positions.

    Multi-sample aggregation (Option A, sec 15.6): self and cross predictions
    are each replaced by the mean over their N non-broken samples, then
    per_forecast_loss is computed once on the means. Cells where either side
    has no usable sample for the dim are dropped (missing-data, sec 9.6).
    """
    pairs: list[tuple[float, float]] = []
    for position, task_idx in enumerate(task_indices):
        task = data.tasks[task_idx]
        if not data._retained[model][task][dim.name]:
            continue
        self_samples = data.self_forecasts.get(model, {}).get(task, {}).get(condition)
        cross_samples = data.cross_forecasts.get(task, {}).get(condition)
        self_pred = _aggregate_prediction(self_samples, dim.name)
        cross_pred = _aggregate_prediction(cross_samples, dim.name)
        if self_pred is None or cross_pred is None:
            continue
        cell = data.outcomes[model][task]
        valid = _valid_subset(cell, dim.name, rollout_indices[position])
        if not valid:
            continue
        loss_self = per_forecast_loss(self_pred, valid, dim)
        loss_cross = per_forecast_loss(cross_pred, valid, dim)
        pairs.append((loss_self, loss_cross))
    return pairs


# --------------------------------------------------------------------------
# Headline: wins-fraction (sec 15.6)
# --------------------------------------------------------------------------
def wins_fraction(
    data: ScoringData,
    model: str,
    condition: str,
    *,
    task_indices=None,
    rollout_indices=None,
    dimensions: tuple[Dimension, ...] | None = None,
) -> float:
    """``wins(M, condition)`` -- fraction of cells where self beat cross (sec 15.6).

    A cell is a (task, dimension) pair. ``loss_self < loss_cross`` counts as 1
    win; an exact tie counts as 0.5; ``loss_self > loss_cross`` counts as 0.
    NaN if no cell is usable.
    """
    dims = dimensions if dimensions is not None else data.dimensions
    task_indices = list(range(len(data.tasks))) if task_indices is None else list(task_indices)
    if rollout_indices is None:
        full = list(range(data.n_rollouts))
        rollout_indices = [full] * len(task_indices)

    n_cells = 0
    score = 0.0
    for dim in dims:
        for loss_self, loss_cross in _dim_losses(
            data, model, dim, condition, task_indices, rollout_indices,
        ):
            n_cells += 1
            if loss_self < loss_cross:
                score += 1.0
            elif loss_self == loss_cross:
                score += 0.5
    return score / n_cells if n_cells else float("nan")


# --------------------------------------------------------------------------
# Per-dimension transparency (sec 15.7 supporting)
# --------------------------------------------------------------------------
def model_scores(
    data: ScoringData,
    model: str,
    condition: str,
    *,
    task_indices=None,
    rollout_indices=None,
    dimensions: tuple[Dimension, ...] | None = None,
) -> dict:
    """Per-dimension transparency for ``model`` in ``condition`` (sec 15.7).

    Returns a dict with:
      ``wins``                     : the headline wins-fraction over all dims
      ``wins_per_dim``             : per-dim wins-fraction (cells collapsed only
                                     within that dim) -- feeds the model×dim
                                     heatmap (plot B2) without needing to
                                     reconstruct ``ScoringData`` downstream.
      ``mean_loss_self_per_dim``   : per-dim mean self loss (native units)
      ``mean_loss_cross_per_dim``  : per-dim mean cross loss (native units)
      ``raw_iai_per_dim``          : per-dim ``mean_loss_cross − mean_loss_self``
                                     (positive = self does better in native units)
      ``n_tasks_used``             : per-dim count of usable task positions
    """
    dims = dimensions if dimensions is not None else data.dimensions
    task_indices = list(range(len(data.tasks))) if task_indices is None else list(task_indices)
    if rollout_indices is None:
        full = list(range(data.n_rollouts))
        rollout_indices = [full] * len(task_indices)

    raw_iai_per_dim: dict[str, float] = {}
    mean_loss_self_per_dim: dict[str, float] = {}
    mean_loss_cross_per_dim: dict[str, float] = {}
    wins_per_dim: dict[str, float] = {}
    n_used: dict[str, int] = {}
    # Multi-sample diagnostics (sec 15.7 addendum): mean within-cell variance
    # of self and cross sample predictions, per dim. Low variance = confident
    # forecaster; high variance = noisy forecaster. Reported in native units.
    mean_self_variance_per_dim: dict[str, float] = {}
    mean_cross_variance_per_dim: dict[str, float] = {}
    for dim in dims:
        pairs = _dim_losses(data, model, dim, condition, task_indices, rollout_indices)
        n_used[dim.name] = len(pairs)
        if not pairs:
            raw_iai_per_dim[dim.name] = float("nan")
            mean_loss_self_per_dim[dim.name] = float("nan")
            mean_loss_cross_per_dim[dim.name] = float("nan")
            wins_per_dim[dim.name] = float("nan")
            mean_self_variance_per_dim[dim.name] = float("nan")
            mean_cross_variance_per_dim[dim.name] = float("nan")
            continue
        losses_self = np.array([p[0] for p in pairs])
        losses_cross = np.array([p[1] for p in pairs])
        m_self = float(losses_self.mean())
        m_cross = float(losses_cross.mean())
        mean_loss_self_per_dim[dim.name] = m_self
        mean_loss_cross_per_dim[dim.name] = m_cross
        raw_iai_per_dim[dim.name] = m_cross - m_self
        # Per-dim wins: <self beats cross> + 0.5 * <tie>, same convention as wins_fraction.
        diffs = losses_self - losses_cross
        n_wins = float(np.sum(diffs < 0))
        n_ties = float(np.sum(diffs == 0))
        wins_per_dim[dim.name] = (n_wins + 0.5 * n_ties) / len(pairs)
        # Per-dim sample variance (multi-sample diagnostic). Computed over the
        # same retained tasks as the losses; missing variances (cells with <2
        # usable samples) are skipped, not NaN-propagated.
        self_vars = [
            v for tidx in task_indices
            if (v := _sample_variance(
                data.self_forecasts.get(model, {}).get(data.tasks[tidx], {}).get(condition),
                dim.name,
            )) is not None
        ]
        cross_vars = [
            v for tidx in task_indices
            if (v := _sample_variance(
                data.cross_forecasts.get(data.tasks[tidx], {}).get(condition),
                dim.name,
            )) is not None
        ]
        mean_self_variance_per_dim[dim.name] = float(np.mean(self_vars)) if self_vars else float("nan")
        mean_cross_variance_per_dim[dim.name] = float(np.mean(cross_vars)) if cross_vars else float("nan")

    return {
        "wins": wins_fraction(
            data, model, condition,
            task_indices=task_indices, rollout_indices=rollout_indices, dimensions=dims,
        ),
        "wins_per_dim": wins_per_dim,
        "mean_loss_self_per_dim": mean_loss_self_per_dim,
        "mean_loss_cross_per_dim": mean_loss_cross_per_dim,
        "raw_iai_per_dim": raw_iai_per_dim,
        "n_tasks_used": n_used,
        "mean_self_variance_per_dim": mean_self_variance_per_dim,
        "mean_cross_variance_per_dim": mean_cross_variance_per_dim,
    }


def all_model_wins(
    data: ScoringData, condition: str, dimensions: tuple[Dimension, ...] | None = None
) -> dict[str, float]:
    """Point-estimate ``wins(M, condition)`` for every IAI-eligible model (sec 15.6)."""
    return {
        m: wins_fraction(data, m, condition, dimensions=dimensions)
        for m in data.models
        if m != data.cross_model and _has_self_forecasts(data, m)
    }


def all_model_scores(
    data: ScoringData, condition: str, dimensions: tuple[Dimension, ...] | None = None
) -> dict[str, dict]:
    """Full per-model scoring dict (wins + per-dim transparency) for every IAI-eligible model."""
    return {
        m: model_scores(data, m, condition, dimensions=dimensions)
        for m in data.models
        if m != data.cross_model and _has_self_forecasts(data, m)
    }


# --------------------------------------------------------------------------
# Empirical solve rate (sec 7) & cross-predictor designation (sec 8)
# --------------------------------------------------------------------------
def _mean_over_valid(cells: list[list[Outcome]], dim_name: str) -> float:
    values = [o.values[dim_name] for cell in cells for o in cell if o.valid_for(dim_name)]
    return float(np.mean(values)) if values else 0.0


def empirical_solve_rate(outcomes: dict[str, dict[str, list[Outcome]]], model: str) -> float:
    """A model's empirical solve rate -- mean p_solve over its rollouts (sec 7)."""
    return _mean_over_valid(list(outcomes[model].values()), "p_solve")


def rank_models(
    outcomes: dict[str, dict[str, list[Outcome]]], models: list[str]
) -> list[str]:
    """Rank models by empirical capability, best first (sec 8).

    Sort key: descending mean ``p_solve``, then -- as tie-breaks -- descending
    mean ``p_first_attempt_correct``, then ascending mean ``total_expected_steps``.
    """
    ranked = []
    for model in models:
        cells = list(outcomes[model].values())
        ranked.append((
            model,
            _mean_over_valid(cells, "p_solve"),
            _mean_over_valid(cells, "p_first_attempt_correct"),
            _mean_over_valid(cells, "total_expected_steps"),
        ))
    ranked.sort(key=lambda r: (-r[1], -r[2], r[3]))
    return [r[0] for r in ranked]


def designate_cross_predictor(
    outcomes: dict[str, dict[str, list[Outcome]]], models: list[str]
) -> str:
    """Designate the post-hoc cross-predictor: the highest-ranked model (sec 8)."""
    return rank_models(outcomes, models)[0]
