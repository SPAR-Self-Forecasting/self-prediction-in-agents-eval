"""Descriptive readout of the five continuous dimensions (paper section support).

The binary dimensions get a Murphy decomposition of the Brier score. MAE admits no
exact analogue, so this script reports the three quantities that answer the same
questions:

- **Bias** -- mean signed error, per model and pooled: which way does the forecast
  miss, and by how much.
- **Debiased MAE** -- MAE after subtracting each model's own constant offset for
  that dimension. The gap to raw MAE is the share of error a per-model intercept
  correction would remove (the reliability analogue).
- **Constant-baseline MAE** -- MAE of the constant forecast "this model's own mean
  outcome on every task". A forecast that cannot beat it carries no task-level
  signal (the resolution analogue).

Also reports per-model Spearman rho between forecast and outcome across tasks.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

import config  # noqa: E402
import plots  # noqa: E402
from murphy import load_scoring_data  # noqa: E402

_EXP_DIR = Path(__file__).resolve().parent

CONTINUOUS_DIMS = tuple(d for d in config.DIMENSIONS if d.kind == "continuous")

#: GLM-5 has outcomes but no self-forecasts in this run; the paper reports 11 models.
EXCLUDE_MODELS = ("z-ai/glm-5",)


def _cells(data, model: str, dim_name: str, condition: str):
    """Yield (forecast, rollout outcomes) for every retained cell of one model."""
    for task in data.tasks:
        if not data._retained[model][task][dim_name]:
            continue
        samples = data.self_forecasts.get(model, {}).get(task, {}).get(condition) or []
        usable = [s for s in samples if s is not None and not s.broken and dim_name in s.predictions]
        if not usable:
            continue
        outs = [o.values[dim_name] for o in data.outcomes[model][task] if o.valid_for(dim_name)]
        if not outs:
            continue
        yield float(np.mean([s.predictions[dim_name] for s in usable])), np.asarray(outs, float)


def model_dim_stats(data, model: str, dim, condition: str) -> dict | None:
    cells = list(_cells(data, model, dim.name, condition))
    if not cells:
        return None
    forecasts = np.array([f for f, _ in cells])
    actual_means = np.array([o.mean() for _, o in cells])

    mae = float(np.mean([np.mean(np.abs(f - o)) for f, o in cells]))
    bias = float(np.mean(forecasts - actual_means))
    offset = float(np.median(forecasts - actual_means))
    debiased_mae = float(np.mean([np.mean(np.abs((f - offset) - o)) for f, o in cells]))
    constant = float(np.mean(actual_means))
    constant_mae = float(np.mean([np.mean(np.abs(constant - o)) for _, o in cells]))

    rho = float("nan")
    if np.unique(forecasts).size > 1 and np.unique(actual_means).size > 1:
        rho = float(spearmanr(forecasts, actual_means)[0])

    return {
        "n_cells": len(cells),
        "mean_forecast": float(forecasts.mean()),
        "mean_actual": constant,
        "bias": bias,
        "mae": mae,
        "debiased_mae": debiased_mae,
        "constant_baseline_mae": constant_mae,
        "spearman_rho": rho,
    }


def build_report(data) -> dict:
    models = [m for m in config.MODEL_IDS if m not in EXCLUDE_MODELS]
    report: dict = {"models": models, "conditions": {}}

    for condition in config.CONDITIONS:
        per_condition: dict = {}
        for dim in CONTINUOUS_DIMS:
            per_model = {}
            for m in models:
                stats = model_dim_stats(data, m, dim, condition)
                if stats is not None:
                    per_model[m] = stats

            def col(key):
                return np.array([s[key] for s in per_model.values()], float)

            rhos = col("spearman_rho")
            rhos = rhos[~np.isnan(rhos)]
            per_condition[dim.name] = {
                "per_model": per_model,
                "mean_forecast": float(col("mean_forecast").mean()),
                "mean_actual": float(col("mean_actual").mean()),
                "mean_bias": float(col("bias").mean()),
                "median_bias": float(np.median(col("bias"))),
                "n_models_overpredicting": int((col("bias") > 0).sum()),
                "mean_mae": float(col("mae").mean()),
                "mean_debiased_mae": float(col("debiased_mae").mean()),
                "mean_constant_baseline_mae": float(col("constant_baseline_mae").mean()),
                "n_models_beating_baseline": int(
                    (col("mae") < col("constant_baseline_mae")).sum()
                ),
                "n_models_debiased_beating_baseline": int(
                    (col("debiased_mae") < col("constant_baseline_mae")).sum()
                ),
                "median_spearman_rho": float(np.median(rhos)) if rhos.size else float("nan"),
                "n_models_positive_rho": int((rhos > 0).sum()),
                "n_models_with_rho": int(rhos.size),
            }
        report["conditions"][condition] = per_condition

    report["thinking_effect"] = _thinking_effect(report, models)
    return report


def build_points(data, models: list[str]) -> dict:
    """Per-cell ``(forecast, actual)`` pairs behind the figure.

    Same cells as :func:`model_dim_stats`, so the scatter and the summary
    statistics printed on it can never drift apart.
    """
    points: dict = {}
    for condition in config.CONDITIONS:
        per_condition: dict = {}
        for dim in CONTINUOUS_DIMS:
            forecast: list[float] = []
            actual: list[float] = []
            model_of: list[str] = []
            for m in models:
                for f, outs in _cells(data, m, dim.name, condition):
                    forecast.append(f)
                    actual.append(float(outs.mean()))
                    model_of.append(m)
            per_condition[dim.name] = {
                "forecast": forecast, "actual": actual, "model": model_of,
            }
        points[condition] = per_condition
    return points


def _thinking_effect(report: dict, models: list[str]) -> dict:
    out: dict = {}
    for dim in CONTINUOUS_DIMS:
        no_think = report["conditions"]["no_thinking"][dim.name]["per_model"]
        think = report["conditions"]["thinking"][dim.name]["per_model"]
        paired = [
            (no_think[m]["mae"], think[m]["mae"]) for m in models if m in no_think and m in think
        ]
        if not paired:
            continue
        a = np.array([p[0] for p in paired])
        b = np.array([p[1] for p in paired])
        try:
            p_value = float(wilcoxon(a, b).pvalue)
        except ValueError:
            p_value = float("nan")
        out[dim.name] = {
            "n_models": len(paired),
            "mae_no_thinking": float(a.mean()),
            "mae_thinking": float(b.mean()),
            "delta": float((b - a).mean()),
            "n_models_improving": int((b < a).sum()),
            "wilcoxon_p": p_value,
        }
    return out


def main() -> int:
    data = load_scoring_data(_EXP_DIR / "data_real")
    report = build_report(data)

    out_path = _EXP_DIR / "results" / "continuous_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    points = build_points(data, report["models"])
    figures_dir = _EXP_DIR / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for layout, stem in (("2x3", "c3_numeric_calibration"),
                         ("1x5", "c3_numeric_calibration_1x5")):
        fig = plots.figure_c3_numeric_calibration(
            points=points, report=report, dimensions=CONTINUOUS_DIMS, layout=layout)
        fig.savefig(figures_dir / f"{stem}.png", dpi=200, pad_inches=0.02)
        plt.close(fig)
        print(f"[continuous] wrote {figures_dir / f'{stem}.png'}")

    for dim in CONTINUOUS_DIMS:
        b = report["conditions"]["no_thinking"][dim.name]
        print(f"\n=== {dim.name} ===")
        print(f"  forecast {b['mean_forecast']:.3f} vs actual {b['mean_actual']:.3f}"
              f"  (bias {b['mean_bias']:+.3f}, {b['n_models_overpredicting']}/11 over)")
        print(f"  MAE {b['mean_mae']:.3f} | debiased {b['mean_debiased_mae']:.3f}"
              f" | constant baseline {b['mean_constant_baseline_mae']:.3f}")
        print(f"  beats constant baseline: raw {b['n_models_beating_baseline']}/11,"
              f" debiased {b['n_models_debiased_beating_baseline']}/11")
        print(f"  median rho {b['median_spearman_rho']:.3f}"
              f" ({b['n_models_positive_rho']}/{b['n_models_with_rho']} positive)")
        t = report["thinking_effect"][dim.name]
        print(f"  thinking: {t['mae_no_thinking']:.3f} -> {t['mae_thinking']:.3f}"
              f" ({t['n_models_improving']}/{t['n_models']} improve, p={t['wilcoxon_p']:.3f})")

    print(f"\n[continuous] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
