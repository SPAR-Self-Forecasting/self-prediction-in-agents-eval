"""Inferential tier for Experiment 1 -- the CIs and tests behind the paper's
SWE-bench claims (EVAL_SPEC_V1.md sec 9, restored; ``analysis.py`` stays the
point-estimate readout).

The resampling unit is the **task**. The 30 SWE-bench Verified instances are an
exchangeable sample (stratified 10/10/10, then drawn under a fixed seed --
``config.SEED``) from a population of tasks we could have drawn instead, so a
task bootstrap answers the question the paper actually asks: would this hold on
other SWE-bench tasks? The 11 models are *not* exchangeable -- five are Qwen
variants and the rest arrive in same-family pairs -- so they are never resampled
as independent replicates. Every primary interval below is per model, computed
across that model's own tasks with the model held fixed, and fleet-level claims
are reported as counts of models whose interval clears the null after a
Benjamini-Hochberg correction across the 11 models. The pooled row is a summary
only and carries the wider of two one-way cluster bootstraps (by task, by model
family), which is the convention the ODCV analysis already uses.

Three estimands, all defined per (model, dimension, condition) over the same
retained cells ``analysis.py`` scores:

- **bias** -- mean over tasks of ``f_c - mean(o_c)``. Positive = the model
  over-predicts. This is the "overconfident about p_solve" claim.
- **skill vs the constant baseline** -- mean per-cell loss minus the loss of
  the constant forecast "this model's own mean outcome, on every task",
  recomputed inside each resample because it is a function of the data.
  Positive = the forecasts are *worse* than a forecaster that ignores the task
  entirely. This is the "no task-level signal" claim. For a binary dimension
  the constant baseline's Brier score is exactly Murphy's UNC term, so this
  statistic is identically ``REL - RES``: a positive interval is a
  distribution-free version of "reliability exceeds resolution".
- **thinking effect** -- mean over tasks of
  ``loss(thinking) - loss(no_thinking)``, paired within task on the cells
  retained under both conditions. Negative = thinking helps.

Spearman rho keeps scipy's own p-value rather than a bootstrap one: resampling
tasks with replacement manufactures ties, which is exactly what a rank statistic
is most sensitive to.

Multiplicity is corrected within each family of tests -- the 11 models inside a
(dimension, statistic) pair, and the 10 dimensions for the fleet-level thinking
contrast -- and both the raw and adjusted p-values are written out.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# Make src/ + this experiment importable when called directly.
_EXP_DIR = Path(__file__).parent.resolve()
_ROOT = _EXP_DIR.parents[1]
for _p in (str(_ROOT), str(_EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: E402
import metrics  # noqa: E402
from murphy import load_scoring_data  # noqa: E402
from src.stats import (  # noqa: E402
    benjamini_hochberg,
    bootstrap_p_value,
    percentile_ci,
    wild_cluster_bootstrap_mean,
)
from src.types import Dimension  # noqa: E402

#: Bootstrap resamples. 10k puts the Monte-Carlo error on a 95% percentile
#: endpoint well below the third decimal, which is the precision we quote.
B = 10_000
SEED = 42
ALPHA = 0.05

#: Chunk size for the MAE constant-baseline bootstrap, which materialises a
#: (chunk, n_tasks, n_rollouts) array.
_CHUNK = 1_000

#: GLM-5 has outcomes but no self-forecasts in this run; the paper reports 11
#: models. Kept identical to ``analyze_continuous.EXCLUDE_MODELS``.
EXCLUDE_MODELS = ("z-ai/glm-5",)


def scored_models() -> list[str]:
    return [m for m in config.MODEL_IDS if m not in EXCLUDE_MODELS]


# --------------------------------------------------------------------------
# Cell assembly -- one row per retained task
# --------------------------------------------------------------------------
@dataclass
class CellTable:
    """One (model, dimension, condition) slice, indexed by task.

    ``outcomes`` is padded to the widest cell and masked, because cells keep
    5-8 valid rollouts; ``out_mean`` / ``out_sq_mean`` are the masked moments,
    which is all the Brier arithmetic needs.
    """

    tasks: list[str]
    forecast: np.ndarray      # (n,)   the cell's mean usable forecast sample
    out_mean: np.ndarray      # (n,)   mean valid rollout outcome
    out_sq_mean: np.ndarray   # (n,)   mean squared valid rollout outcome
    outcomes: np.ndarray      # (n, r) padded valid outcomes
    valid: np.ndarray         # (n, r) mask over ``outcomes``
    loss: np.ndarray          # (n,)   per-cell Brier / MAE

    @property
    def n(self) -> int:
        return len(self.tasks)


def collect(
    data: metrics.ScoringData, model: str, dim: Dimension, condition: str,
) -> CellTable | None:
    """Assemble the retained cells for one slice, applying exactly the gates of
    ``metrics._dim_self_losses`` so every number here scores the same cells the
    point estimates do."""
    tasks: list[str] = []
    forecasts: list[float] = []
    per_cell: list[np.ndarray] = []
    for task in data.tasks:
        if not data._retained[model][task][dim.name]:
            continue
        samples = data.self_forecasts.get(model, {}).get(task, {}).get(condition)
        pred = metrics._aggregate_prediction(samples, dim.name)
        if pred is None:
            continue
        valid = [o.values[dim.name] for o in data.outcomes[model][task] if o.valid_for(dim.name)]
        if not valid:
            continue
        tasks.append(task)
        forecasts.append(float(pred))
        per_cell.append(np.asarray(valid, dtype=float))
    if not tasks:
        return None

    width = max(len(v) for v in per_cell)
    outcomes = np.zeros((len(tasks), width), dtype=float)
    mask = np.zeros((len(tasks), width), dtype=bool)
    for i, v in enumerate(per_cell):
        outcomes[i, : v.size] = v
        mask[i, : v.size] = True

    f = np.asarray(forecasts, dtype=float)
    counts = mask.sum(axis=1)
    out_mean = (outcomes * mask).sum(axis=1) / counts
    out_sq_mean = (outcomes**2 * mask).sum(axis=1) / counts
    if dim.is_binary:
        loss = ((f[:, None] - outcomes) ** 2 * mask).sum(axis=1) / counts
    else:
        loss = (np.abs(f[:, None] - outcomes) * mask).sum(axis=1) / counts
    return CellTable(tasks, f, out_mean, out_sq_mean, outcomes, mask, loss)


# --------------------------------------------------------------------------
# The constant baseline
# --------------------------------------------------------------------------
def constant_baseline_loss(table: CellTable, dim: Dimension, rows=None, const=None):
    """Loss of the constant forecast "this model's own mean outcome".

    ``rows`` selects cells (shape ``(draws, n)`` of task indices, or None for
    all of them) and ``const`` supplies the per-draw constant; both default to
    the observed sample. Returns a scalar for the observed sample and a
    ``(draws,)`` vector otherwise.
    """
    if rows is None:
        m1 = float(table.out_mean.mean())
        c = np.array([m1 if const is None else float(const)])
        sel = np.arange(table.n)[None, :]
    else:
        sel = rows
        c = np.asarray(table.out_mean[sel].mean(axis=1) if const is None else const, dtype=float)

    if dim.is_binary:
        # mean_o (c - o)^2 = c^2 - 2 c E[o] + E[o^2], so the padded rollouts
        # never need to be touched.
        m1 = table.out_mean[sel].mean(axis=1)
        m2 = table.out_sq_mean[sel].mean(axis=1)
        out = c**2 - 2.0 * c * m1 + m2
    else:
        counts = table.valid.sum(axis=1)
        out = np.empty(sel.shape[0], dtype=float)
        for lo in range(0, sel.shape[0], _CHUNK):
            hi = min(lo + _CHUNK, sel.shape[0])
            idx = sel[lo:hi]                                  # (k, n)
            o = table.outcomes[idx]                           # (k, n, r)
            mk = table.valid[idx]
            per_cell = (np.abs(c[lo:hi, None, None] - o) * mk).sum(axis=2) / counts[idx]
            out[lo:hi] = per_cell.mean(axis=1)
    return float(out[0]) if rows is None else out


# --------------------------------------------------------------------------
# Per-model, per-dimension inference
# --------------------------------------------------------------------------
def _interval(boot: np.ndarray, point: float) -> dict:
    lo, hi = percentile_ci(boot, alpha=ALPHA)
    # A percentile bootstrap cannot resolve a tail thinner than one resample, so
    # the p-value is floored at 2/B rather than reported as an exact zero.
    p = max(bootstrap_p_value(boot, null=0.0), 2.0 / boot.size)
    return {
        "estimate": float(point),
        "ci95": [lo, hi],
        "p": p,
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }


def model_dimension_stats(
    data: metrics.ScoringData, model: str, dim: Dimension, condition: str, *, b: int = B,
) -> dict | None:
    """Bias, constant-baseline skill and Spearman rho for one slice, each with a
    95% task-bootstrap interval."""
    table = collect(data, model, dim, condition)
    if table is None or table.n < 5:
        return None

    rng = np.random.default_rng(SEED)
    rows = rng.integers(0, table.n, size=(b, table.n))

    bias_cells = table.forecast - table.out_mean
    bias_boot = bias_cells[rows].mean(axis=1)
    skill_boot = table.loss[rows].mean(axis=1) - constant_baseline_loss(table, dim, rows)

    bias = float(bias_cells.mean())
    skill = float(table.loss.mean()) - constant_baseline_loss(table, dim)

    rho, rho_p = float("nan"), float("nan")
    if np.unique(table.forecast).size > 1 and np.unique(table.out_mean).size > 1:
        res = spearmanr(table.forecast, table.out_mean)
        rho, rho_p = float(res.statistic), float(res.pvalue)

    return {
        "n_tasks": table.n,
        "mean_forecast": float(table.forecast.mean()),
        "mean_outcome": float(table.out_mean.mean()),
        "mean_loss": float(table.loss.mean()),
        "constant_baseline_loss": constant_baseline_loss(table, dim),
        "bias": _interval(bias_boot, bias),
        # for binary dims this is exactly Murphy's REL - RES
        "skill_vs_constant": _interval(skill_boot, skill),
        "spearman": {
            "rho": rho,
            "p": rho_p,
            "defined": bool(np.isfinite(rho)),
        },
    }


def thinking_effect_stats(
    data: metrics.ScoringData, model: str, dim: Dimension, *, b: int = B,
) -> dict | None:
    """Paired ``loss(thinking) - loss(no_thinking)`` over the tasks retained in
    both conditions, with a 95% task-bootstrap interval."""
    nt = collect(data, model, dim, "no_thinking")
    th = collect(data, model, dim, "thinking")
    if nt is None or th is None:
        return None
    shared = [t for t in nt.tasks if t in set(th.tasks)]
    if len(shared) < 5:
        return None
    a = np.array([nt.loss[nt.tasks.index(t)] for t in shared])
    c = np.array([th.loss[th.tasks.index(t)] for t in shared])
    diff = c - a

    rng = np.random.default_rng(SEED)
    rows = rng.integers(0, diff.size, size=(b, diff.size))
    boot = diff[rows].mean(axis=1)
    out = _interval(boot, float(diff.mean()))
    out.update({
        "n_tasks": len(shared),
        "loss_no_thinking": float(a.mean()),
        "loss_thinking": float(c.mean()),
        "n_tasks_improving": int((c < a).sum()),
    })
    return out


# --------------------------------------------------------------------------
# Fleet-level pooling -- summary only
# --------------------------------------------------------------------------
def _pooled(values, tasks, families, *, b: int = B) -> dict | None:
    """Mean of a cell-level quantity with the wider of two one-way cluster
    bootstraps: clustering by task and clustering by model family.

    Neither clustering alone is right -- cells share a task and cells share a
    family -- so we report whichever is more conservative rather than pretending
    a two-way estimator is available at 5 families.
    """
    v = np.asarray(values, dtype=float)
    if v.size < 5:
        return None
    boot_t = wild_cluster_bootstrap_mean(v, tasks, b, SEED)
    boot_f = wild_cluster_bootstrap_mean(v, families, b, SEED)
    ci_t, ci_f = percentile_ci(boot_t, ALPHA), percentile_ci(boot_f, ALPHA)
    wider, boot = ((ci_t, boot_t) if (ci_t[1] - ci_t[0]) >= (ci_f[1] - ci_f[0])
                   else (ci_f, boot_f))
    return {
        "estimate": float(v.mean()),
        "n_cells": int(v.size),
        "ci95_task_clustered": list(ci_t),
        "ci95_family_clustered": list(ci_f),
        "ci95_reported": list(wider),
        "p": bootstrap_p_value(boot, null=0.0),
        "excludes_zero": bool(wider[0] > 0.0 or wider[1] < 0.0),
        "note": "summary only; the per-model rows are the primary result",
    }


# --------------------------------------------------------------------------
# The headline loss and the capability-scaling claim
# --------------------------------------------------------------------------
def _aligned(data: metrics.ScoringData, models: list[str], condition: str):
    """``{model: {dim: array over data.tasks, NaN where the cell is not scored}}``.

    Aligning every dimension to the same task axis is what lets one resampled
    task vector drive all ten dimensions at once, which is the correct pairing:
    a bootstrap that drew different tasks per dimension would understate the
    spread of an average taken across dimensions.
    """
    index = {t: i for i, t in enumerate(data.tasks)}
    out: dict[str, dict[str, np.ndarray]] = {}
    for m in models:
        per_dim: dict[str, np.ndarray] = {}
        for dim in data.dimensions:
            col = np.full(len(data.tasks), np.nan)
            table = collect(data, m, dim, condition)
            if table is not None:
                for t, loss in zip(table.tasks, table.loss):
                    col[index[t]] = loss
            per_dim[dim.name] = col
        out[m] = per_dim
    return out


def _solve_rate_by_task(data: metrics.ScoringData, model: str) -> np.ndarray:
    """Per-task mean p_solve, NaN where no rollout is valid -- the task-level
    decomposition of ``metrics.empirical_solve_rate``."""
    col = np.full(len(data.tasks), np.nan)
    for i, task in enumerate(data.tasks):
        vals = [o.values["p_solve"] for o in data.outcomes[model][task] if o.valid_for("p_solve")]
        if vals:
            col[i] = float(np.mean(vals))
    return col


def _nanmean_rows(col: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Row-wise nanmean of ``col[rows]``, NaN for an all-NaN row."""
    picked = col[rows]
    ok = ~np.isnan(picked)
    counts = ok.sum(axis=1)
    total = np.where(ok, picked, 0.0).sum(axis=1)
    return np.where(counts > 0, total / np.maximum(counts, 1), np.nan)


def scaling_report(
    data: metrics.ScoringData, models: list[str], *, condition: str = "no_thinking", b: int = B,
) -> dict:
    """Per-model headline loss and solve rate with task-bootstrap CIs, plus a
    permutation test of the capability-scaling claim.

    The headline loss divides each dimension by its largest value across the 11
    models before averaging (``metrics.dimension_loss_scales``). Those divisors
    are held at their observed values inside the bootstrap: they are a display
    normalisation chosen so ``avg_tokens_step`` does not swamp the average, not
    a quantity we are trying to estimate.

    The scaling claim is tested by **permutation over models**, not by a
    bootstrap. The null is "self-forecast loss is unrelated to capability", and
    reshuffling which solve rate is attached to which model tests exactly that
    while conditioning on the 11 models we actually ran -- so it never needs
    them to be an exchangeable sample of models, which they are not. Because
    same-family models are correlated we also run the permutation over whole
    families, which is the conservative version.
    """
    dims = data.dimensions
    aligned = _aligned(data, models, condition)
    observed = {
        m: {d.name: float(np.nanmean(aligned[m][d.name])) for d in dims}
        for m in models
    }
    scales = metrics.dimension_loss_scales(observed, [d.name for d in dims])

    rng = np.random.default_rng(SEED)
    rows = rng.integers(0, len(data.tasks), size=(b, len(data.tasks)))

    per_model: dict[str, dict] = {}
    for m in models:
        boot = np.zeros(b, dtype=float)
        used = 0
        for d in dims:
            col = _nanmean_rows(aligned[m][d.name], rows) / scales[d.name]
            if np.all(np.isnan(col)):
                continue
            boot += np.nan_to_num(col, nan=float(np.nanmean(col)))
            used += 1
        boot /= max(used, 1)
        loss = float(np.mean([observed[m][d.name] / scales[d.name] for d in dims]))

        srate = _solve_rate_by_task(data, m)
        sboot = _nanmean_rows(srate, rows)
        per_model[m] = {
            "headline_loss": {
                "estimate": loss,
                "ci95": list(percentile_ci(boot, ALPHA)),
            },
            "solve_rate": {
                "estimate": float(np.nanmean(srate)),
                "ci95": list(percentile_ci(sboot, ALPHA)),
            },
        }

    x = np.array([per_model[m]["solve_rate"]["estimate"] for m in models])
    y = np.array([per_model[m]["headline_loss"]["estimate"] for m in models])
    rho = float(spearmanr(x, y).statistic)

    perm = np.array([spearmanr(rng.permutation(x), y).statistic for _ in range(b)])
    p_model = (int(np.sum(np.abs(perm) >= abs(rho))) + 1) / (b + 1)

    # The conservative version collapses each family to one point before
    # correlating, so the five Qwen variants cannot be counted as five
    # independent pieces of evidence. Five points admit only 5! = 120
    # arrangements, which are enumerated exactly; the smallest p it can return
    # is 2/120, and that floor is reported rather than hidden.
    fams = [config.model_by_id(m).family for m in models]
    fam_names = sorted(set(fams))
    fx = np.array([x[[i for i, g in enumerate(fams) if g == f]].mean() for f in fam_names])
    fy = np.array([y[[i for i, g in enumerate(fams) if g == f]].mean() for f in fam_names])
    fam_rho = float(spearmanr(fx, fy).statistic)
    arrangements = list(permutations(range(len(fam_names))))
    fam_null = np.array([spearmanr(fx[list(p)], fy).statistic for p in arrangements])
    p_family = float(np.mean(np.abs(fam_null) >= abs(fam_rho)))

    return {
        "condition": condition,
        "per_model": per_model,
        "loss_scales": scales,
        "spearman_rho": rho,
        "permutation_p_models": p_model,
        "family_level": {
            "n_families": len(fam_names),
            "families": fam_names,
            "spearman_rho": fam_rho,
            "exact_permutation_p": p_family,
            "smallest_attainable_p": 2.0 / len(arrangements),
        },
        "note": "permutation over models conditions on the 11 models run, so it "
                "does not require them to be an exchangeable sample; the "
                "family-level version collapses each family to one point first "
                "and is the conservative reading",
    }


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------
def _apply_bh(per_model: dict, path: tuple[str, ...], sign_key: str = "estimate") -> dict:
    """BH-adjust a p-value nested at ``path`` across the models that have one,
    writing ``p_bh`` / ``significant`` back in place.

    Returns the significant count split by the sign of the estimate, because
    "8 of 11 models differ from zero" hides the fact that they can differ in
    opposite directions -- which is exactly what happens on
    \\texttt{documentation}.
    """
    def dig(entry):
        node = entry
        for key in path:
            if node is None or key not in node:
                return None
            node = node[key]
        return node

    tested = [m for m, e in per_model.items()
              if e is not None and dig(e) is not None and np.isfinite(dig(e)["p"])]
    counts = {"n_significant": 0, "n_significant_positive": 0, "n_significant_negative": 0}
    if not tested:
        return counts
    adj = benjamini_hochberg([dig(per_model[m])["p"] for m in tested])
    for m, p_adj in zip(tested, adj):
        node = dig(per_model[m])
        node["p_bh"] = float(p_adj)
        node["significant"] = bool(p_adj <= ALPHA)
        if not node["significant"]:
            continue
        counts["n_significant"] += 1
        key = "n_significant_positive" if node[sign_key] > 0 else "n_significant_negative"
        counts[key] += 1
    return counts


def dimension_report(
    data: metrics.ScoringData, dim: Dimension, condition: str, models: list[str], *, b: int = B,
) -> dict:
    per_model = {m: model_dimension_stats(data, m, dim, condition, b=b) for m in models}
    live = {m: e for m, e in per_model.items() if e is not None}

    bias = _apply_bh(live, ("bias",))
    skill = _apply_bh(live, ("skill_vs_constant",))
    # _apply_bh mutates the nested dicts in place, and ``live`` shares them with
    # ``per_model``, so the wrapper below still lands the adjusted rho p-values
    # on the reported entries.
    rho_counts = _apply_bh(
        {m: {"spearman": e["spearman"]} for m, e in live.items() if e["spearman"]["defined"]},
        ("spearman",), sign_key="rho",
    )

    fam_of = {m: config.model_by_id(m).family for m in models}
    bias_vals, skill_vals, tasks, fams = [], [], [], []
    for m, e in live.items():
        table = collect(data, m, dim, condition)
        if table is None:
            continue
        bias_vals.extend((table.forecast - table.out_mean).tolist())
        base = constant_baseline_loss(table, dim)
        skill_vals.extend((table.loss - base).tolist())
        tasks.extend(table.tasks)
        fams.extend([fam_of[m]] * table.n)

    rhos = [e["spearman"]["rho"] for e in live.values() if e["spearman"]["defined"]]
    return {
        "kind": dim.kind,
        "n_models": len(live),
        "per_model": per_model,
        # bias: positive = over-predicts
        "n_models_overpredicting": sum(1 for e in live.values() if e["bias"]["estimate"] > 0),
        "n_models_bias_significant": bias["n_significant"],
        "n_models_significantly_over": bias["n_significant_positive"],
        "n_models_significantly_under": bias["n_significant_negative"],
        # skill: positive = loses to the model's own constant baseline
        "n_models_beating_constant": sum(
            1 for e in live.values() if e["skill_vs_constant"]["estimate"] < 0
        ),
        "n_models_significantly_worse_than_constant": skill["n_significant_positive"],
        "n_models_significantly_better_than_constant": skill["n_significant_negative"],
        "n_models_rho_significant": rho_counts["n_significant"],
        "n_models_rho_significant_positive": rho_counts["n_significant_positive"],
        "n_models_rho_defined": len(rhos),
        "n_models_rho_positive": sum(1 for r in rhos if r > 0),
        "median_spearman_rho": float(np.median(rhos)) if rhos else float("nan"),
        "pooled_bias": _pooled(bias_vals, tasks, fams, b=b),
        "pooled_skill_vs_constant": _pooled(skill_vals, tasks, fams, b=b),
    }


def thinking_report(
    data: metrics.ScoringData, dim: Dimension, models: list[str], *, b: int = B,
) -> dict:
    per_model = {m: thinking_effect_stats(data, m, dim, b=b) for m in models}
    live = {m: e for m, e in per_model.items() if e is not None}
    counts = _apply_bh({m: {"_": e} for m, e in live.items()}, ("_",))
    return {
        "per_model": per_model,
        "n_models": len(live),
        "n_models_improving": sum(1 for e in live.values() if e["estimate"] < 0),
        "n_models_significant": counts["n_significant"],
        "n_models_significantly_improving": counts["n_significant_negative"],
        "n_models_significantly_worsening": counts["n_significant_positive"],
        "mean_delta": float(np.mean([e["estimate"] for e in live.values()])) if live else float("nan"),
        "loss_no_thinking": float(np.mean([e["loss_no_thinking"] for e in live.values()]))
        if live else float("nan"),
        "loss_thinking": float(np.mean([e["loss_thinking"] for e in live.values()]))
        if live else float("nan"),
    }


def build_report(data: metrics.ScoringData, *, b: int = B) -> dict:
    models = [m for m in scored_models() if metrics._has_self_forecasts(data, m)]
    report: dict = {
        "method": {
            "resampling_unit": f"task ({len(data.tasks)} SWE-bench Verified "
                               f"instances, seed {config.SEED}); models held "
                               "fixed, never resampled",
            "B": b,
            "alpha": ALPHA,
            "interval": "percentile bootstrap",
            "multiplicity": "Benjamini-Hochberg within each family of tests "
                            "(11 models per dimension/statistic; 10 dimensions "
                            "for the fleet-level thinking contrast)",
            "pooled": "wider of two one-way cluster bootstraps (task, model family)",
            "identity": "for binary dimensions skill_vs_constant == REL - RES "
                        "of the Murphy decomposition",
        },
        "models": models,
        "families": {m: config.model_by_id(m).family for m in models},
        "conditions": {},
        "thinking_effect": {},
    }

    for condition in config.CONDITIONS:
        report["conditions"][condition] = {
            dim.name: dimension_report(data, dim, condition, models, b=b)
            for dim in data.dimensions
        }

    report["scaling"] = scaling_report(data, models, b=b)

    for dim in data.dimensions:
        report["thinking_effect"][dim.name] = thinking_report(data, dim, models, b=b)

    # The thinking contrast is run on every dimension at once, so its
    # fleet-level p-values carry a second BH pass across dimensions.
    fleet = {}
    for dim in data.dimensions:
        block = report["thinking_effect"][dim.name]
        live = [e for e in block["per_model"].values() if e is not None]
        if len(live) < 5:
            continue
        deltas = np.array([e["estimate"] for e in live], dtype=float)
        fams = [config.model_by_id(m).family
                for m, e in block["per_model"].items() if e is not None]
        boot = wild_cluster_bootstrap_mean(deltas, fams, b, SEED)
        lo, hi = percentile_ci(boot, ALPHA)
        fleet[dim.name] = {
            "estimate": float(deltas.mean()),
            "ci95_family_clustered": [lo, hi],
            "p": bootstrap_p_value(boot, null=0.0),
        }
    if fleet:
        names = list(fleet)
        for name, p_adj in zip(names, benjamini_hochberg([fleet[n]["p"] for n in names])):
            fleet[name]["p_bh"] = float(p_adj)
            fleet[name]["significant"] = bool(p_adj <= ALPHA)
    report["thinking_effect_fleet"] = fleet
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _ci(entry: dict | None, key: str, width: int) -> str:
    if entry is None:
        return " " * width
    lo, hi = entry[key]
    return f"{entry['estimate']:+.3g} [{lo:+.3g}, {hi:+.3g}]"


def _summarise(report: dict) -> None:
    n = len(report["models"])
    names = list(report["conditions"]["no_thinking"])
    print("\nsig counts are models whose 95% task-bootstrap CI clears 0 after BH "
          f"across the {n} models; over/under and worse/better are signed splits.\n")
    print(f"{'dimension':26s} {'pooled bias':>30s}  over under"
          f" {'pooled skill vs constant':>30s}  worse better")
    for name in names:
        blk = report["conditions"]["no_thinking"][name]
        print(f"{name:26s} {_ci(blk['pooled_bias'], 'ci95_reported', 30):>30s}"
              f"  {blk['n_models_significantly_over']:4d}"
              f" {blk['n_models_significantly_under']:5d}"
              f" {_ci(blk['pooled_skill_vs_constant'], 'ci95_reported', 30):>30s}"
              f"  {blk['n_models_significantly_worse_than_constant']:5d}"
              f" {blk['n_models_significantly_better_than_constant']:6d}")

    print(f"\n{'dimension':26s} {'fleet thinking delta':>30s}  BH  improving  sig. improving")
    for name in names:
        blk = report["thinking_effect"][name]
        fl = report["thinking_effect_fleet"].get(name)
        star = "*" if fl and fl.get("significant") else " "
        print(f"{name:26s} {_ci(fl, 'ci95_family_clustered', 30):>30s}  {star}"
              f"  {blk['n_models_improving']:6d}/{blk['n_models']}"
              f" {blk['n_models_significantly_improving']:10d}/{blk['n_models']}")

    print(f"\n{'dimension':26s} {'median rho':>11s}  positive  BH-significant")
    for name in names:
        blk = report["conditions"]["no_thinking"][name]
        print(f"{name:26s} {blk['median_spearman_rho']:11.3f}"
              f"  {blk['n_models_rho_positive']:5d}/{blk['n_models_rho_defined']}"
              f"  {blk['n_models_rho_significant_positive']:11d}/{blk['n_models_rho_defined']}")

    sc = report["scaling"]
    fam = sc["family_level"]
    print(f"\nscaling (loss vs solve rate): rho {sc['spearman_rho']:+.3f},"
          f" permutation p over {len(report['models'])} models {sc['permutation_p_models']:.4f};"
          f" collapsed to {fam['n_families']} families rho {fam['spearman_rho']:+.3f},"
          f" exact p {fam['exact_permutation_p']:.4f}"
          f" (floor {fam['smallest_attainable_p']:.4f})")
    print(f"\n{'model':32s} {'solve rate':>22s} {'headline self loss':>22s}")
    for m, e in sc["per_model"].items():
        print(f"{m:32s} {_ci(e['solve_rate'], 'ci95', 22):>22s}"
              f" {_ci(e['headline_loss'], 'ci95', 22):>22s}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Task-bootstrap CIs and BH-corrected tests for the SWE-bench claims.")
    parser.add_argument("--data-dir", default=str(_EXP_DIR / "data_real"),
                        help="Run cache directory (tasks/outcomes/forecasts).")
    parser.add_argument("--results-dir", default=str(_EXP_DIR / "results"),
                        help="Where significance.json is written.")
    parser.add_argument("-B", "--resamples", type=int, default=B,
                        help=f"Bootstrap resamples (default {B}).")
    args = parser.parse_args(argv)

    data = load_scoring_data(Path(args.data_dir))
    report = build_report(data, b=args.resamples)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "significance.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _summarise(report)
    print(f"\n[significance] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
