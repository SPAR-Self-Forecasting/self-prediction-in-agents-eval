"""Scoring aggregates for Experiment 1 (EVAL_SPEC_V1.md sec 15.6 + sec 15.7).

Self-forecast accuracy only — how well each model predicts its own behaviour:

- **Headline:** ``mean_loss(M, condition)`` -- mean per-cell loss averaged across
  dimensions and tasks, in each dimension's native units when reported per-dim.
- **Per-dimension transparency:** ``mean_loss_per_dim`` -- Brier for binary
  dims, MAE for continuous dims.

Per-cell loss is selected by ``Dimension.kind`` (sec 15.7): Brier for binary,
MAE for continuous.
"""

from __future__ import annotations

from collections.abc import Sequence
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
    """Everything the metrics need, indexed for fast (re)computation."""

    dimensions: tuple[Dimension, ...]
    models: list[str]
    families: dict[str, list[str]]
    tasks: list[str]
    outcomes: dict[str, dict[str, list[Outcome]]]
    self_forecasts: dict[str, dict[str, dict[str, list[Forecast]]]]

    def __post_init__(self) -> None:
        first = self.outcomes[self.models[0]][self.tasks[0]]
        self.n_rollouts = len(first)
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
    return [
        cell[i].values[dim_name]
        for i in rollout_idx_subset
        if cell[i].valid_for(dim_name)
    ]


def _usable_samples(samples: list[Forecast] | None, dim_name: str) -> list[Forecast]:
    if not samples:
        return []
    return [s for s in samples if (s is not None and not s.broken and dim_name in s.predictions)]


def _aggregate_prediction(samples: list[Forecast] | None, dim_name: str) -> float | None:
    usable = _usable_samples(samples, dim_name)
    if not usable:
        return None
    return float(np.mean([s.predictions[dim_name] for s in usable]))


def _sample_variance(samples: list[Forecast] | None, dim_name: str) -> float | None:
    usable = _usable_samples(samples, dim_name)
    if len(usable) < 2:
        return None
    return float(np.var([s.predictions[dim_name] for s in usable], ddof=1))


def _has_self_forecasts(data: ScoringData, model: str) -> bool:
    cells = data.self_forecasts.get(model)
    if not cells:
        return False
    return any(
        samples
        for per_cond in cells.values()
        for samples in per_cond.values()
        if samples
    )


# --------------------------------------------------------------------------
# Per-dimension self losses
# --------------------------------------------------------------------------
def _dim_self_losses(
    data: ScoringData,
    model: str,
    dim: Dimension,
    condition: str,
    task_indices,
    rollout_indices,
) -> list[float]:
    """Self-forecast losses over usable (task, dim) cells."""
    losses: list[float] = []
    for position, task_idx in enumerate(task_indices):
        task = data.tasks[task_idx]
        if not data._retained[model][task][dim.name]:
            continue
        self_samples = data.self_forecasts.get(model, {}).get(task, {}).get(condition)
        self_pred = _aggregate_prediction(self_samples, dim.name)
        if self_pred is None:
            continue
        cell = data.outcomes[model][task]
        valid = _valid_subset(cell, dim.name, rollout_indices[position])
        if not valid:
            continue
        losses.append(per_forecast_loss(self_pred, valid, dim))
    return losses


# --------------------------------------------------------------------------
# Per-model scores
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
    """Per-dimension self-forecast accuracy for ``model`` in ``condition``."""
    dims = dimensions if dimensions is not None else data.dimensions
    task_indices = list(range(len(data.tasks))) if task_indices is None else list(task_indices)
    if rollout_indices is None:
        full = list(range(data.n_rollouts))
        rollout_indices = [full] * len(task_indices)

    mean_loss_per_dim: dict[str, float] = {}
    n_used: dict[str, int] = {}
    mean_self_variance_per_dim: dict[str, float] = {}
    for dim in dims:
        losses = _dim_self_losses(
            data, model, dim, condition, task_indices, rollout_indices,
        )
        n_used[dim.name] = len(losses)
        if not losses:
            mean_loss_per_dim[dim.name] = float("nan")
            mean_self_variance_per_dim[dim.name] = float("nan")
            continue
        mean_loss_per_dim[dim.name] = float(np.mean(losses))
        self_vars = [
            v for tidx in task_indices
            if (v := _sample_variance(
                data.self_forecasts.get(model, {}).get(data.tasks[tidx], {}).get(condition),
                dim.name,
            )) is not None
        ]
        mean_self_variance_per_dim[dim.name] = (
            float(np.mean(self_vars)) if self_vars else float("nan"))

    usable = [v for v in mean_loss_per_dim.values() if not np.isnan(v)]
    return {
        "mean_loss_per_dim": mean_loss_per_dim,
        "mean_loss": float(np.mean(usable)) if usable else float("nan"),
        "n_tasks_used": n_used,
        "mean_self_variance_per_dim": mean_self_variance_per_dim,
    }


def dimension_loss_scales(
    mean_loss_per_dim_by_model: dict[str, dict[str, float]],
    dim_names: Sequence[str],
) -> dict[str, float]:
    """Per-dimension max loss across models -- divisors for equal-weight pooling.

    Each dimension's native-unit loss is divided by its scale so every dimension
    contributes equally to the pooled headline (Brier dims already live in
    ``[0, 1]``; continuous MAE dims can reach hundreds without scaling).
    """
    scales: dict[str, float] = {}
    for name in dim_names:
        vals = [
            mlpd[name]
            for mlpd in mean_loss_per_dim_by_model.values()
            if name in mlpd and not np.isnan(mlpd[name])
        ]
        peak = max(vals) if vals else 1.0
        scales[name] = peak if peak > 0 else 1.0
    return scales


def equal_dimension_mean_loss(
    mean_loss_per_dim: dict[str, float],
    scales: dict[str, float],
    dim_names: Sequence[str] | None = None,
) -> float:
    """Mean of per-dimension losses after dividing each dim by its scale."""
    names = list(dim_names) if dim_names is not None else list(mean_loss_per_dim.keys())
    normed: list[float] = []
    for name in names:
        raw = mean_loss_per_dim.get(name)
        if raw is None or np.isnan(raw):
            continue
        normed.append(float(raw) / scales[name])
    return float(np.mean(normed)) if normed else float("nan")


def per_model_equal_dimension_loss(
    scores_by_model: dict[str, dict],
    dim_names: Sequence[str],
) -> dict[str, float]:
    """Equal-dimension headline loss for every model in one condition."""
    mlpds = {
        m: s["mean_loss_per_dim"]
        for m, s in scores_by_model.items()
        if "mean_loss_per_dim" in s
    }
    scales = dimension_loss_scales(mlpds, dim_names)
    return {
        m: equal_dimension_mean_loss(mlpd, scales, dim_names)
        for m, mlpd in mlpds.items()
    }


def all_model_scores(
    data: ScoringData, condition: str, dimensions: tuple[Dimension, ...] | None = None
) -> dict[str, dict]:
    return {
        m: model_scores(data, m, condition, dimensions=dimensions)
        for m in data.models
        if _has_self_forecasts(data, m)
    }


# --------------------------------------------------------------------------
# Empirical solve rate (sec 7)
# --------------------------------------------------------------------------
def _mean_over_valid(cells: list[list[Outcome]], dim_name: str) -> float:
    values = [o.values[dim_name] for cell in cells for o in cell if o.valid_for(dim_name)]
    return float(np.mean(values)) if values else 0.0


def empirical_solve_rate(outcomes: dict[str, dict[str, list[Outcome]]], model: str) -> float:
    return _mean_over_valid(list(outcomes[model].values()), "p_solve")


def rank_models(
    outcomes: dict[str, dict[str, list[Outcome]]], models: list[str]
) -> list[str]:
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
