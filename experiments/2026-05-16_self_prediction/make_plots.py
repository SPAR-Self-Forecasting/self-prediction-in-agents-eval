"""Headline figures for Experiment 1 -- the flat ``<results-dir>/plots/`` set.

Renders the three RQ readouts from ``analysis.json`` and, when ``murphy.json``
exists (stage 8), the two Murphy-decomposition figures alongside them:

    rq1_loss_by_model.png       rq2_loss_vs_solverate.png   rq3_thinking_delta.png
    dimension_accuracy.png      murphy_decomposition.png    reliability_diagrams.png

Called automatically by ``run.py`` stage 9; also runs standalone:

    python make_plots.py [--results-dir results]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_EXP_DIR = Path(__file__).parent.resolve()
_ROOT = _EXP_DIR.parents[1]
for _p in (str(_ROOT), str(_EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: E402
import plots as figmod  # noqa: E402  (canonical figure functions for C1/C2)


def _short(name: str) -> str:
    return name.split("/", 1)[-1]


# ---------------------------------------------------------------- colours
_SELF_LOSS = "#2a7fb8"
_THINKING_HELPED = "#2a7fb8"
_THINKING_HURT = "#c44"


#: Figure-coordinate box for the RQ1 caveat panel: (x0, y0, width, height).
_CAVEAT_BOX = (0.008, 0.012, 0.984, 0.170)


def _draw_point_estimate_caveat(fig: plt.Figure) -> None:
    """Spell out "point estimates only" on the figure instead of hinting at it."""
    x0, y0, w, h = _CAVEAT_BOX
    fig.patches.append(plt.Rectangle(
        (x0, y0), w, h, transform=fig.transFigure, zorder=0,
        facecolor="#fbfbfb", edgecolor="#dcdcdc", linewidth=1.0))

    pad = 0.012
    fig.text(x0 + pad, y0 + h - 0.016,
             "Point estimates only — why there are no error bars",
             fontsize=9.5, fontweight="bold", color="#222", va="top")

    left = (
        "Every bar is a single computed number, not an estimate with uncertainty around it.\n"
        "The study's inference tier — hierarchical bootstrap CIs, significance tests and the\n"
        "robustness battery — was deliberately cut for the MVP, on the logic that uncertainty\n"
        "isn't worth pricing until the primary signal is known to exist. It was scoped out,\n"
        "not attempted and abandoned."
    )
    right = (
        "So don't read the small gaps between neighbouring bars as significant. Per-model\n"
        "loss carries standard error from finite cell counts, and cells cluster within task\n"
        "and within rollout — nothing here should be quoted with significance language.\n"
        "Confound W2: loss = bias\u00b2 + forecast variance, so erratic forecasters carry\n"
        "inflated self-loss and weaker models are noisier — the weak end of this chart is a\n"
        "lower bound on self-knowledge rather than an estimate of it."
    )
    for x, body in ((x0 + pad, left), (x0 + 0.475, right)):
        fig.text(x, y0 + h - 0.042, body, fontsize=7.6, color="#333",
                 va="top", linespacing=1.45)


# ---------------------------------------------------------------- RQ1
def figure_rq1(analysis: dict) -> plt.Figure:
    rq1 = analysis["rq1_accuracy"]
    loss = rq1["per_model_loss"]
    condition = rq1.get("condition", "no_thinking")

    items = sorted(loss.items(), key=lambda kv: kv[1])
    labels = [_short(k) for k, _ in items]
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(13.2, 8.6))
    ax.barh(labels, vals, color=_SELF_LOSS, height=0.68, zorder=3)

    loss_x = 0.700
    top = len(vals) - 0.5
    x_lo = 0.0
    x_hi = max(vals) * 1.18 if vals else 1.0
    ax.set_xlim(x_lo, x_hi)

    for i, (_, v) in enumerate(items):
        ax.text(loss_x, i, f"{v:.4g}", va="center", ha="right",
                fontsize=9, fontweight="bold", color="#222", zorder=4)
    ax.text(loss_x, top + 0.55, "mean loss", va="center", ha="right",
            fontsize=8.5, color="#333", style="italic")

    ax.set_ylim(-0.7, top + 1.35)
    ax.set_xlabel("mean self-forecast loss across (task, dimension) cells\n(lower = more accurate)",
                  fontsize=10)
    ax.grid(axis="x", alpha=0.25, linestyle=":", zorder=0)
    ax.set_axisbelow(True)

    mean_loss = rq1.get("point_estimate", float(np.mean(vals)))
    fig.suptitle(
        "RQ1 — how accurately do models forecast their own behaviour?",
        fontsize=13, y=0.985)
    ax.set_title(
        f"condition: {condition}  ·  mean loss = {mean_loss:.4g}  ·  "
        f"{len(vals)} models (sorted best → worst)",
        fontsize=10, pad=24)

    fig.text(
        0.008, 0.218,
        "How to read it — every model attempts 30 SWE-bench Verified tasks and forecasts its own "
        "behaviour on 10 dimensions. Each (task, dimension) pair is one cell, scored by how far "
        "the self-forecast was from what actually happened. A bar is the mean loss across that "
        "model's cells; lower is better.",
        fontsize=8.5, color="#444", va="top", linespacing=1.5)

    _draw_point_estimate_caveat(fig)

    fig.tight_layout(rect=(0, 0.237, 1, 0.945))
    return fig


# ---------------------------------------------------------------- RQ2
def figure_rq2(analysis: dict) -> plt.Figure:
    rq2 = analysis["rq2_scaling"]
    solve = rq2["solve_rates"]
    loss = rq2["per_model_loss"]

    xs = np.array([solve[k] for k in loss])
    ys = np.array([loss[k] for k in loss])

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.scatter(xs, ys, s=60, color=_SELF_LOSS, zorder=3)
    for k in loss:
        ax.annotate(_short(k), (solve[k], loss[k]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    slope, intercept = np.polyfit(xs, ys, 1)
    xline = np.linspace(xs.min(), xs.max(), 2)
    ax.plot(xline, slope * xline + intercept, color="#c22", lw=1.6, zorder=2,
            label=f"OLS trend (slope {slope:+.2g}, descriptive)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax.set_xlabel("empirical solve rate (capability)")
    ax.set_ylabel("equal-dimension mean loss (lower = better)")
    ax.set_title(
        f"RQ2 — Self-forecast accuracy vs capability\n"
        f"Spearman rho = {rq2['spearman_rho']:.3f}  (descriptive; not a significance test)"
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- RQ3
def figure_rq3(analysis: dict) -> plt.Figure:
    rq3 = analysis["rq3_elicitation"]
    diff = rq3["per_model_difference"]
    ditems = sorted(diff.items(), key=lambda kv: kv[1])
    dlabels = [_short(k) for k, _ in ditems]
    dvals = [v for _, v in ditems]
    dcolors = [_THINKING_HELPED if v < 0 else _THINKING_HURT for v in dvals]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(dlabels, dvals, color=dcolors)
    ax.axvline(0.0, color="k", lw=1)
    ax.set_xlabel("Δ loss = loss(thinking) − loss(no_thinking)\n(negative = thinking helped)")
    ax.set_title(
        f"RQ3 — Does 'thinking' elicit better self-prediction?\n"
        f"mean Δ = {rq3['point_estimate']:+.4g}  (null 0)"
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- per-dimension
def _mean_across_models(per_model: dict, key: str, dim: str) -> float:
    vals = [rec[key][dim] for rec in per_model.values()
            if dim in rec.get(key, {}) and rec[key][dim] is not None]
    return float(np.mean(vals)) if vals else float("nan")


def _dim_label(name: str) -> str:
    """One dimension tick: name plus native unit when we have one."""
    unit = figmod.DIM_UNITS.get(name, "")
    return f"{name}  ({unit})" if unit else name


def figure_dimensions(scores: dict, *, condition: str = "no_thinking") -> plt.Figure:
    """Mean self-forecast loss per dimension, averaged across models.

    Two side-by-side panels over the same 10 dimensions:

    * **binary** dimensions share a Brier axis
    * **continuous** dimensions each have their own native units, on a log axis
    """
    per_model = scores["scores"][condition]
    dims = [d.name for d in config.DIMENSIONS]
    binary = [d.name for d in config.DIMENSIONS if d.kind == "binary"]
    n_binary = len(binary)

    self_loss = [_mean_across_models(per_model, "mean_loss_per_dim", d) for d in dims]

    x = np.arange(len(dims), dtype=float)
    n_cont = len(dims) - n_binary
    bin_vals = [v for v in self_loss[:n_binary] if np.isfinite(v)]
    cont_vals = [v for v in self_loss[n_binary:] if np.isfinite(v) and v > 0]
    fig = plt.figure(figsize=(13.8, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[n_binary, n_cont])
    ax_bin = fig.add_subplot(gs[0, 0])
    ax_cont = fig.add_subplot(gs[0, 1])

    w = 0.55
    for ax, lo, hi in ((ax_bin, 0, n_binary), (ax_cont, n_binary, len(dims))):
        s, xs = self_loss[lo:hi], x[lo:hi]
        ax.bar(xs, s, w, color=figmod.SELF_COLORS[condition],
               label="the model forecasting itself", zorder=3)
        pad = 1.06 if ax is ax_cont else 1.02
        for xi, sv in zip(xs, s):
            if not np.isfinite(sv):
                continue
            ax.text(xi, sv * pad, f"{sv:.3g}", ha="center", va="bottom",
                    fontsize=7.5, color="#333", zorder=4)
        ax.set_xlim(lo - 0.62, hi - 0.38)
        ax.set_xticks(xs, [_dim_label(d) for d in dims[lo:hi]],
                      fontsize=8, rotation=28, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", pad=2)
        ax.grid(axis="y", alpha=0.25, linestyle=":", zorder=0)
        ax.set_axisbelow(True)
    if cont_vals:
        ax_cont.set_yscale("log")
        ax_cont.set_ylim(min(cont_vals) * 0.5, max(cont_vals) * 4.0)
    if bin_vals:
        ax_bin.set_ylim(0, max(bin_vals) * 1.30)
    ax_bin.set_ylabel("mean Brier loss\n(lower = more accurate)", fontsize=9)
    ax_cont.set_ylabel("mean absolute error, log scale\n(each dimension its own unit)",
                       fontsize=9)
    ax_cont.yaxis.set_label_position("right")
    ax_cont.yaxis.tick_right()
    ax_bin.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    ax_bin.set_title("binary dimensions — one shared Brier scale",
                     fontsize=9.5, color="#444", style="italic")
    ax_cont.set_title("continuous dimensions — four different units, never compare heights across bars",
                      fontsize=9.5, color="#444", style="italic")

    fig.suptitle(
        "Self-forecast loss by dimension — what do models actually know about themselves?\n"
        f"mean over {len(per_model)} models, condition: {condition}",
        fontsize=13, y=0.985)

    fig.text(
        0.008, 0.012,
        "How to read it — each model forecasts its own behaviour on 10 dimensions. "
        "Heights compare only within a dimension (same unit). "
        "Point estimates only, averaged across models — no confidence intervals; "
        "noisier forecasters carry inflated self-loss (confound W2).",
        fontsize=8, color="#444", va="bottom", linespacing=1.5)

    fig.subplots_adjust(left=0.075, right=0.915, top=0.82, bottom=0.28, wspace=0.10)
    return fig


# ---------------------------------------------------------------- entry points
def render_headline_plots(results_dir: Path) -> list[Path]:
    """Render the flat plots/ set from ``<results_dir>/{analysis,murphy}.json``.

    The three RQ figures always; the two Murphy figures when ``murphy.json``
    exists. Returns the written paths.
    """
    results_dir = Path(results_dir)
    analysis = json.loads((results_dir / "analysis.json").read_text(encoding="utf-8"))
    outdir = results_dir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    figures = {
        "rq1_loss_by_model": figure_rq1(analysis),
        "rq2_loss_vs_solverate": figure_rq2(analysis),
        "rq3_thinking_delta": figure_rq3(analysis),
    }
    scores_path = results_dir / "scores.json"
    if scores_path.exists():
        scores = json.loads(scores_path.read_text(encoding="utf-8"))
        if scores.get("scores"):
            figures["dimension_accuracy"] = figure_dimensions(scores)

    murphy_path = results_dir / "murphy.json"
    if murphy_path.exists():
        murphy = json.loads(murphy_path.read_text(encoding="utf-8"))
        figures["murphy_decomposition"] = figmod.figure_c1_murphy_components(
            murphy=murphy, dimensions=config.DIMENSIONS)
        figures["reliability_diagrams"] = figmod.figure_c2_reliability_diagrams(
            murphy=murphy, dimensions=config.DIMENSIONS, layout="1x5")
        figures["reliability_diagrams_2x2_plus_1"] = figmod.figure_c2_reliability_diagrams(
            murphy=murphy, dimensions=config.DIMENSIONS, layout="2x2+1")

    written: list[Path] = []
    for name, fig in figures.items():
        out = outdir / f"{name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the flat headline plots/ set.")
    parser.add_argument("--results-dir", default=str(_EXP_DIR / "results"),
                        help="Directory containing analysis.json (+ optional murphy.json).")
    args = parser.parse_args(argv)
    written = render_headline_plots(Path(args.results_dir))
    print("wrote:")
    for p in written:
        print("  ", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
