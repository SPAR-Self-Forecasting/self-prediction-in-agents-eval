"""Generic scoring primitives for self-prediction experiments.

Three responsibilities (EVAL_SPEC_V1.md sec 3, sec 9.3):

1. **Per-forecast loss** (sec 3.1) -- Brier loss for binary dimensions, absolute
   error for continuous dimensions, the latter optionally on a log1p scale.
2. **Per-dimension standardisation** (sec 3.3) -- the fixed scale ``s_d`` and the
   ``raw / s_d`` unit conversion.
3. **The hierarchical bootstrap** (sec 9.3) -- the primary inference engine. This
   module supplies the *generic resampling engine*; the experiment's
   ``metrics.py`` supplies the statistic closure that closes over the data.

Everything here is experiment-agnostic: dimension semantics enter only through
the generic ``Dimension`` descriptor.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from src.types import Dimension


# --------------------------------------------------------------------------
# Scale transforms
# --------------------------------------------------------------------------
def transform(values, scale: str):
    """Apply a dimension's loss-scale transform.

    ``raw``   -> identity.
    ``log1p`` -> natural log(1 + x). log1p(0) == 0 cleanly handles a zero-step
                 or empty rollout (sec 3.1). Negative input is a bug (counts are
                 non-negative) and raises loudly.
    """
    arr = np.asarray(values, dtype=float)
    if scale == "raw":
        return arr
    if scale == "log1p":
        if np.any(arr < 0.0):
            raise ValueError(f"log1p scale requires non-negative values; got min {arr.min()}")
        return np.log1p(arr)
    raise ValueError(f"Unknown scale {scale!r}")


# --------------------------------------------------------------------------
# Per-forecast loss (sec 3.1)
# --------------------------------------------------------------------------
def per_forecast_loss(forecast: float, outcomes: Sequence[float], dimension: Dimension) -> float:
    """Per-cell loss of a single forecast against its rollout outcomes (sec 15.7).

    Loss function is selected by dimension kind, computed on the raw scale:
      - **binary** dims     -> Brier: ``mean over rollouts of (f - o)**2``
      - **continuous** dims -> MAE:   ``mean over rollouts of |f - o|``
    """
    outs = np.asarray(outcomes, dtype=float)
    if outs.size == 0:
        raise ValueError("per_forecast_loss requires at least one rollout outcome")
    f = float(forecast)
    if dimension.kind == "binary":
        return float(np.mean((f - outs) ** 2))
    if dimension.kind == "continuous":
        return float(np.mean(np.abs(f - outs)))
    raise ValueError(f"Unknown Dimension.kind {dimension.kind!r}")


# --------------------------------------------------------------------------
# Per-dimension standardisation (sec 3.3)
# --------------------------------------------------------------------------
def pooled_sd(values: Sequence[float], scale: str) -> float:
    """The fixed per-dimension scale ``s_d``.

    Pooled (sample, ddof=1) standard deviation of a dimension's ground-truth
    outcomes -- computed once over all 14 models x 30 tasks x 8 rollouts -- on
    the same scale the loss uses (sec 3.3).
    """
    arr = transform(values, scale)
    if arr.size < 2:
        raise ValueError("pooled_sd requires at least two values")
    return float(np.std(arr, ddof=1))


def standardize(raw: float, s_d: float) -> float:
    """Convert a raw_IAI / mean-loss to dimensionless skill-score units: ``raw / s_d**2`` (W1).

    For squared loss, dividing by the *variance* of outcomes makes the ratio a
    dimensionless fraction-of-baseline-error quantity -- equivalently a Brier
    Skill Score (binary) or R² (continuous). ``s_d <= 0`` means the dimension
    is globally degenerate; per §15.6 the caller is expected to honour the W3
    floor and not invoke standardize() on a floored dimension.
    """
    if s_d <= 0.0:
        raise ValueError(f"s_d must be > 0 (degenerate dimension); got {s_d}")
    s_d = float(s_d)
    return float(raw) / (s_d * s_d)


# --------------------------------------------------------------------------
# Hierarchical bootstrap (sec 9.3)
# --------------------------------------------------------------------------
@dataclass
class Resample:
    """One draw of the 4-level hierarchical bootstrap (sec 9.3).

    ``models``  -- drawn model keys (families then models, both with
                   replacement); may contain duplicates. Position = draw id.
    ``families``-- parallel list: the family each drawn model came from.
    ``task_indices``    -- length n_tasks; the resampled task index vector,
                   shared across all drawn models (sec 9.3 step 3).
    ``rollout_indices`` -- shape (n_draws, n_tasks, n_rollouts); resampled
                   rollout indices for each drawn (model, task) cell. Indexed
                   by draw id so a duplicated model gets independent rollout
                   resamples (sec 9.3 step 4).
    """

    models: list
    families: list
    task_indices: np.ndarray
    rollout_indices: np.ndarray


def identity_resample(families: dict[str, list], n_tasks: int, n_rollouts: int) -> Resample:
    """The non-resampled design -- used to compute the point estimate."""
    models: list = []
    fam_of: list = []
    for fam, fam_models in families.items():
        for m in fam_models:
            models.append(m)
            fam_of.append(fam)
    n_draws = len(models)
    task_indices = np.arange(n_tasks)
    rollout_indices = np.broadcast_to(
        np.arange(n_rollouts), (n_draws, n_tasks, n_rollouts)
    ).copy()
    return Resample(models, fam_of, task_indices, rollout_indices)


def hierarchical_bootstrap(
    families: dict[str, list],
    n_tasks: int,
    n_rollouts: int,
    statistic: Callable[[Resample], float],
    B: int,
    seed: int,
) -> np.ndarray:
    """Run the 4-level hierarchical bootstrap and return the B statistic values.

    Each of B resamples (sec 9.3):
      1. resample the families with replacement (the cluster level);
      2. within each drawn family, resample its models with replacement;
      3. resample the n_tasks tasks with replacement (one vector, shared);
      4. within each drawn (model, task) cell, resample its rollouts.

    ``statistic`` receives the :class:`Resample` and recomputes the full metric.
    """
    if B <= 0:
        raise ValueError(f"B must be positive, got {B}")
    rng = np.random.default_rng(seed)
    fam_names = list(families)
    out = np.empty(B, dtype=float)
    for b in range(B):
        # 1. resample families
        drawn_fams = rng.choice(fam_names, size=len(fam_names), replace=True)
        models: list = []
        fam_of: list = []
        for fam in drawn_fams:
            fam_models = families[fam]
            # 2. resample models within the drawn family
            idx = rng.integers(0, len(fam_models), size=len(fam_models))
            for i in idx:
                models.append(fam_models[i])
                fam_of.append(fam)
        n_draws = len(models)
        # 3. resample tasks (shared vector)
        task_indices = rng.integers(0, n_tasks, size=n_tasks)
        # 4. resample rollouts per drawn cell
        rollout_indices = rng.integers(0, n_rollouts, size=(n_draws, n_tasks, n_rollouts))
        out[b] = statistic(Resample(models, fam_of, task_indices, rollout_indices))
    return out
