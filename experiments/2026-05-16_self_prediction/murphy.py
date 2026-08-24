"""Murphy decomposition of the binary-dimension Brier scores (downstream analysis).

For each binary dimension the pipeline's Brier score is decomposed into the
three classic Murphy (1973) components:

    BS  =  REL  -  RES  +  UNC

  - **UNC**  (uncertainty)   -- ``base_rate * (1 - base_rate)``: the variance of
    the outcomes themselves. The floor no forecaster can beat. Captures the
    rollout-to-rollout stochasticity of the agent.
  - **REL**  (reliability / miscalibration, lower better) -- how far the stated
    probabilities sit from the observed frequencies at that probability level
    ("says 0.7, happens 0.3").
  - **RES**  (resolution / discrimination, higher better) -- how much the
    forecasts actually separate high-rate from low-rate cases.

So "what drives errors most" becomes: is a forecaster losing because it is
*miscalibrated* (REL) or because it *cannot discriminate* which cells go which
way (low RES against the UNC floor)?

Estimators
----------
**Primary: CORP** (Dimitriadis, Gneiting & Jordan 2021, PNAS): the recalibrated
forecast is the isotonic (PAV) regression of outcomes on forecasts. Bin-free --
right for our quasi-continuous forecasts (means of <=5 samples) -- with an
*exact* identity by construction and provably non-negative REL / RES:

    REL = BS(f) - BS(pav(f)),   RES = UNC - BS(pav(f))

(tiny negative values in the report are floating-point error only).

**Secondary: fixed-bin Murphy** (``binned`` sub-dict) -- the textbook binned
estimator, kept because its per-bin table *is* the reliability diagram (plot
C2). With continuous forecasts the identity holds only up to a within-bin
residual, reported honestly as ``residual``.

Weighting -- one non-obvious choice
-----------------------------------
The pipeline's per-dim transparency (``mean_loss_self_per_dim``) is an
*unweighted mean over cells*, while cells contribute 5-8 valid rollouts each.
Events therefore carry weight ``1 / n_valid(cell)`` so every cell contributes
equally -- making the decomposed ``brier`` reproduce the pipeline's per-dim
mean loss exactly (a built-in validation invariant, asserted in tests).

Units of analysis
-----------------
Per (model, dim, condition), on **exactly the cells self-forecast scoring used**
(same retention + forecast-usability gates as ``metrics._dim_self_losses``),
then a mean across models. A pooled view (events concatenated across models)
is also reported, clearly caveated: base rates differ wildly across models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make src/ + this experiment importable when called directly.
_EXP_DIR = Path(__file__).parent.resolve()
_ROOT = _EXP_DIR.parents[1]
for _p in (str(_ROOT), str(_EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: E402
import metrics  # noqa: E402
from src.types import Forecast, Outcome  # noqa: E402

#: Default bin count for the secondary binned estimator / reliability diagrams.
N_BINS = 10

#: The binary dimensions (Brier-scored) -- the only ones Murphy applies to.
BINARY_DIMENSIONS = tuple(d for d in config.DIMENSIONS if d.is_binary)


# --------------------------------------------------------------------------
# Weighted PAV (pool-adjacent-violators) -- the CORP recalibration
# --------------------------------------------------------------------------
def pav_calibrate(forecasts: np.ndarray, outcomes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted isotonic (non-decreasing) least-squares fit of outcomes on forecasts.

    Events with *equal* forecast values are pooled first (a proper isotonic fit
    on x must be a function of x), then adjacent violating blocks are merged by
    weighted mean. Returns the fitted value per event, in the input order.
    """
    order = np.argsort(forecasts, kind="stable")
    f_sorted = forecasts[order]
    o_sorted = outcomes[order]
    w_sorted = weights[order]

    # Pool ties in f: one block per unique forecast value.
    uniq, inverse = np.unique(f_sorted, return_inverse=True)
    block_w = np.bincount(inverse, weights=w_sorted)
    block_wo = np.bincount(inverse, weights=w_sorted * o_sorted)

    # PAV over the tie-pooled blocks: merge while the running means decrease.
    means: list[float] = []   # block weighted means (kept non-decreasing)
    ws: list[float] = []      # block weights
    sizes: list[int] = []     # how many unique-f groups each block spans
    for i in range(len(uniq)):
        means.append(block_wo[i] / block_w[i])
        ws.append(block_w[i])
        sizes.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            m2, w2, s2 = means.pop(), ws.pop(), sizes.pop()
            m1, w1, s1 = means.pop(), ws.pop(), sizes.pop()
            means.append((m1 * w1 + m2 * w2) / (w1 + w2))
            ws.append(w1 + w2)
            sizes.append(s1 + s2)

    # Expand block means back to unique-f groups, then to events.
    per_group = np.empty(len(uniq))
    pos = 0
    for m, s in zip(means, sizes):
        per_group[pos: pos + s] = m
        pos += s
    fitted_sorted = per_group[inverse]
    fitted = np.empty_like(fitted_sorted)
    fitted[order] = fitted_sorted
    return fitted


# --------------------------------------------------------------------------
# The two decompositions
# --------------------------------------------------------------------------
def _wmean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def _empty_components() -> dict:
    nan = float("nan")
    return {
        "brier": nan, "reliability": nan, "resolution": nan, "uncertainty": nan,
        "bs_recalibrated": nan, "base_rate": nan, "n_events": 0,
        "binned": {"reliability": nan, "resolution": nan, "uncertainty": nan,
                   "residual": nan, "bins": []},
    }


def corp_decomposition(
    forecasts, outcomes, weights=None, *, n_bins: int = N_BINS,
) -> dict:
    """CORP Murphy decomposition of the (weighted) Brier score.

    Returns the CORP components at the top level (exact identity
    ``brier == reliability - resolution + uncertainty`` by construction) plus a
    ``binned`` sub-dict with the fixed-bin estimator and its per-bin table
    (the reliability-diagram data). Empty input -> NaN components.
    """
    f = np.asarray(forecasts, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    if f.size == 0:
        return _empty_components()
    w = np.ones_like(f) if weights is None else np.asarray(weights, dtype=float)
    if not (f.shape == o.shape == w.shape):
        raise ValueError("forecasts, outcomes, weights must have identical shapes")
    if np.any((o != 0.0) & (o != 1.0)):
        bad = o[(o != 0.0) & (o != 1.0)][:3]
        raise ValueError(f"Murphy decomposition needs binary outcomes; got {bad}")
    if np.any(w <= 0.0):
        raise ValueError("weights must be positive")

    brier = _wmean((f - o) ** 2, w)
    base_rate = _wmean(o, w)
    unc = _wmean((base_rate - o) ** 2, w)  # == p(1-p) for binary o, any weights

    recal = pav_calibrate(f, o, w)
    bs_recal = _wmean((recal - o) ** 2, w)

    return {
        "brier": brier,
        "reliability": brier - bs_recal,   # >= 0 up to float error (f is isotonic in f)
        "resolution": unc - bs_recal,      # >= 0 up to float error (constant is isotonic)
        "uncertainty": unc,
        "bs_recalibrated": bs_recal,
        "base_rate": base_rate,
        "n_events": int(f.size),
        "binned": binned_decomposition(f, o, w, n_bins=n_bins),
    }


def binned_decomposition(
    forecasts, outcomes, weights=None, *, n_bins: int = N_BINS, strategy: str = "uniform",
) -> dict:
    """Textbook fixed-bin Murphy decomposition + the per-bin reliability table.

    ``strategy``: "uniform" -> equal-width bins on [0, 1]; "quantile" ->
    (unweighted) forecast-quantile edges, deduplicated, so heavy ties collapse
    bins rather than emptying them.

    With continuous forecasts the identity holds only up to the within-bin
    ``residual`` (= brier - (REL - RES + UNC)), reported rather than hidden.
    """
    f = np.asarray(forecasts, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    if f.size == 0:
        return _empty_components()["binned"]
    w = np.ones_like(f) if weights is None else np.asarray(weights, dtype=float)

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(f, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 2:  # all forecasts identical
            edges = np.array([edges[0], edges[0] + 1e-12])
    else:
        raise ValueError(f"unknown binning strategy {strategy!r}")

    idx = np.clip(np.searchsorted(edges, f, side="right") - 1, 0, len(edges) - 2)

    total_w = float(np.sum(w))
    base_rate = _wmean(o, w)
    unc = _wmean((base_rate - o) ** 2, w)
    brier = _wmean((f - o) ** 2, w)

    rel = 0.0
    res = 0.0
    bins: list[dict] = []
    for k in range(len(edges) - 1):
        mask = idx == k
        if not np.any(mask):
            continue
        wk = float(np.sum(w[mask]))
        mean_f = _wmean(f[mask], w[mask])
        rate = _wmean(o[mask], w[mask])
        rel += (wk / total_w) * (mean_f - rate) ** 2
        res += (wk / total_w) * (rate - base_rate) ** 2
        bins.append({
            "lo": float(edges[k]), "hi": float(edges[k + 1]),
            "n_events": int(np.sum(mask)), "weight": wk,
            "mean_forecast": mean_f, "observed_rate": rate,
        })

    return {
        "reliability": rel,
        "resolution": res,
        "uncertainty": unc,
        "residual": brier - (rel - res + unc),
        "bins": bins,
    }


# --------------------------------------------------------------------------
# Event assembly from ScoringData -- mirrors metrics._dim_losses gating
# --------------------------------------------------------------------------
def binary_events(
    data: metrics.ScoringData, model: str, dim, condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """(forecasts, outcomes, weights, n_cells) event arrays for one scored slice.

    Applies exactly the gates of ``metrics._dim_self_losses`` -- retention (>=5
    valid rollouts) and a usable self forecast. Each cell's valid rollouts
    become events sharing the cell's aggregated forecast, weighted ``1 / n_valid``
    so cells contribute equally (see module docstring).
    """
    fs: list[float] = []
    os_: list[float] = []
    ws: list[float] = []
    n_cells = 0
    for task in data.tasks:
        if not data._retained[model][task][dim.name]:
            continue
        self_samples = data.self_forecasts.get(model, {}).get(task, {}).get(condition)
        self_pred = metrics._aggregate_prediction(self_samples, dim.name)
        if self_pred is None:
            continue
        cell = data.outcomes[model][task]
        valid = [o.values[dim.name] for o in cell if o.valid_for(dim.name)]
        if not valid:
            continue
        n_cells += 1
        for v in valid:
            fs.append(float(self_pred))
            os_.append(float(v))
            ws.append(1.0 / len(valid))
    return (np.asarray(fs), np.asarray(os_), np.asarray(ws), n_cells)


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------
_POOLED_CAVEAT = (
    "Pooled across models: base rates differ per model. "
    "Read the per-model decomposition as primary."
)

#: The scalar components averaged across models (bin tables are not averaged).
_SCALAR_KEYS = ("brier", "reliability", "resolution", "uncertainty",
                "bs_recalibrated", "base_rate")


def _mean_components(per_model: dict) -> dict:
    """Unweighted mean of each scalar component across models with events."""
    used = [m for m, entry in per_model.items() if entry["n_events"] > 0]
    out: dict = {"n_models": len(used)}
    for key in _SCALAR_KEYS:
        vals = [per_model[m][key] for m in used]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def murphy_report(
    data: metrics.ScoringData,
    *,
    n_bins: int = N_BINS,
    conditions=config.CONDITIONS,
    dimensions=None,
) -> dict:
    """The full Murphy readout: per (condition, binary dim), per model + mean + pooled.

    ``dimensions`` defaults to the binary dims of ``data.dimensions`` (Murphy
    applies to Brier-scored dims only).
    """
    if dimensions is None:
        dimensions = tuple(d for d in data.dimensions if d.is_binary)
    scored_models = [m for m in data.models if metrics._has_self_forecasts(data, m)]
    report: dict = {
        "method": {
            "primary": "CORP (PAV isotonic recalibration; Dimitriadis et al. 2021)",
            "secondary": f"fixed-bin Murphy, {n_bins} uniform bins (reliability diagrams)",
            "weighting": "events weighted 1/n_valid(cell); cells contribute equally, "
                         "matching the pipeline's unweighted per-cell mean loss",
            "identity": "brier == reliability - resolution + uncertainty (CORP: exact)",
        },
        "n_bins": n_bins,
        "models": scored_models,
        "conditions": {},
    }
    for cond in conditions:
        report["conditions"][cond] = {}
        for dim in dimensions:
            per_model: dict = {}
            pooled = {"f": [], "o": [], "w": []}
            for m in scored_models:
                f, o, w, n_cells = binary_events(data, m, dim, cond)
                entry = corp_decomposition(f, o, w, n_bins=n_bins)
                entry["n_cells"] = n_cells
                per_model[m] = entry
                pooled["f"].append(f)
                pooled["o"].append(o)
                pooled["w"].append(w)
            f = np.concatenate(pooled["f"]) if pooled["f"] else np.array([])
            o = np.concatenate(pooled["o"]) if pooled["o"] else np.array([])
            w = np.concatenate(pooled["w"]) if pooled["w"] else np.array([])
            pooled_view = corp_decomposition(f, o, w, n_bins=n_bins)
            report["conditions"][cond][dim.name] = {
                "per_model": per_model,
                "mean_across_models": _mean_components(per_model),
                "pooled": {**pooled_view, "caveat": _POOLED_CAVEAT},
            }
    return report


# --------------------------------------------------------------------------
# Read-only cache loader + CLI
# --------------------------------------------------------------------------
def load_scoring_data(data_dir: Path) -> metrics.ScoringData:
    """Rebuild the ScoringData from an existing run's cache, strictly read-only.

    Missing *outcome* files are a loud failure (they mean the cache is not a
    completed run); missing forecast *sample* files are tolerated exactly like
    ``run.py``'s sample-level retention.
    """
    data_dir = Path(data_dir)

    def _read(path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    task_ids = [t["instance_id"] for t in _read(data_dir / "tasks.json")]

    outcomes: dict = {}
    for model in config.MODEL_IDS:
        slug = config.model_by_id(model).slug
        outcomes[model] = {}
        for tid in task_ids:
            cell = []
            for idx in range(config.N_ROLLOUTS):
                path = data_dir / "outcomes" / slug / tid / f"r{idx}.json"
                if not path.exists():
                    raise FileNotFoundError(f"missing outcome cache file: {path}")
                cell.append(Outcome.from_dict(_read(path)))
            outcomes[model][tid] = cell

    def _read_samples(model: str, kind: str) -> dict:
        slug = config.model_by_id(model).slug
        out: dict = {}
        for tid in task_ids:
            per_condition: dict = {c: [] for c in config.CONDITIONS}
            for sample_idx in range(config.N_FORECAST_SAMPLES):
                path = data_dir / "forecasts" / kind / slug / tid / f"s{sample_idx}.json"
                if not path.exists():
                    continue
                for c, d in _read(path).items():
                    per_condition[c].append(Forecast.from_dict(d))
            out[tid] = per_condition
        return out

    self_forecasts = {m: _read_samples(m, "self") for m in config.MODEL_IDS}

    return metrics.ScoringData(
        dimensions=config.DIMENSIONS,
        models=list(config.MODEL_IDS),
        families=config.families(),
        tasks=task_ids,
        outcomes=outcomes,
        self_forecasts=self_forecasts,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Murphy decomposition of the binary-dim Brier scores.")
    parser.add_argument("--data-dir", default=str(_EXP_DIR / "data"),
                        help="Run cache directory (tasks/outcomes/forecasts).")
    parser.add_argument("--results-dir", default=str(_EXP_DIR / "results"),
                        help="Where murphy.json is written.")
    parser.add_argument("--n-bins", type=int, default=N_BINS,
                        help=f"Bin count for the reliability diagrams (default {N_BINS}).")
    args = parser.parse_args(argv)

    data = load_scoring_data(Path(args.data_dir))
    report = murphy_report(data, n_bins=args.n_bins)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "murphy.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    for dim_name, block in report["conditions"]["no_thinking"].items():
        mean = block["mean_across_models"]
        print(f"[murphy] {dim_name:26s} REL {mean['reliability']:.4f}"
              f" | RES {mean['resolution']:.4f} | UNC {mean['uncertainty']:.4f}")
    print(f"[murphy] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
