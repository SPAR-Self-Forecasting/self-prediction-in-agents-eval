"""Headline figures for Experiment 1 -- the flat ``<results-dir>/plots/`` set.

Renders the three RQ readouts from ``analysis.json`` and, when ``murphy.json``
exists (stage 8), the two Murphy-decomposition figures alongside them:

    rq1_wins_by_model.png       rq2_wins_vs_solverate.png   rq3_thinking_delta.png
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


# ---------------------------------------------------------------- RQ1
_SELF_BETTER = "#2a7fb8"
_PEER_BETTER = "#c44"
_TIE = "#9a9a9a"


def _scored_cell_counts(analysis: dict, n_tasks: int | None) -> dict[str, int]:
    """Cells behind each model's wins fraction, from the missingness report.

    A cell clears two gates: ground-truth retention (per dimension) and forecast
    usability (all 10 dims at once). ``dropped_cell_rate`` covers the first. The
    second only bites when a model's forecasts are *entirely* broken for a task,
    so the count is exact while ``broken_forecast_rate`` is 0 and an upper bound
    otherwise -- the caller marks that case rather than quietly overstating it.

    ``dropped_cell_rate`` is a *fraction* of tasks, so the absolute count needs
    ``n_tasks`` from the run summary. Without it we return nothing rather than
    assume a task count and print a wrong denominator.
    """
    if not n_tasks:
        return {}
    dropped = analysis.get("missingness", {}).get("dropped_cell_rate", {})
    return {model: sum(round(n_tasks * (1.0 - r)) for r in per_dim.values())
            for model, per_dim in dropped.items()}


#: Figure-coordinate box for the RQ1 caveat panel: (x0, y0, width, height).
_CAVEAT_BOX = (0.008, 0.012, 0.984, 0.190)


def _draw_point_estimate_caveat(fig: plt.Figure, *, peer: str) -> None:
    """Spell out "point estimates only" on the figure instead of hinting at it.

    The one-line version assumed the reader knew what an inference tier is and
    which direction W2 pushes; both decide whether a gap between two bars means
    anything, so they belong on the chart rather than in the report.
    """
    x0, y0, w, h = _CAVEAT_BOX
    fig.patches.append(plt.Rectangle(
        (x0, y0), w, h, transform=fig.transFigure, zorder=0,
        facecolor="#fbfbfb", edgecolor="#dcdcdc", linewidth=1.0))

    pad = 0.012
    fig.text(x0 + pad, y0 + h - 0.016,
             "Point estimates only — why there are no error bars, and which way to lean",
             fontsize=9.5, fontweight="bold", color="#222", va="top")

    left = (
        "Every bar is a single computed number, not an estimate with uncertainty around it.\n"
        "The study's inference tier — hierarchical bootstrap CIs, significance tests and the\n"
        "robustness battery — was deliberately cut for the MVP, on the logic that uncertainty\n"
        "isn't worth pricing until the primary signal is known to exist. It was scoped out,\n"
        "not attempted and abandoned.\n"
        f"Only one peer was run, so \u201cbeats the peer\u201d means beats {peer}, not peers in general;\n"
        "the alternate-peer battery was pruned along with the rest of the tier."
    )
    right = (
        "So don't read the small gaps. A per-model wins fraction carries roughly \u00b10.03\u20130.05\n"
        "standard error from cell counts alone — about the width of the gap between neighbouring\n"
        "bars — and even that is optimistic, because cells cluster within task and within rollout.\n"
        "Nothing here should be quoted with significance language.\n"
        "Confound W2: loss = bias\u00b2 + forecast variance, and a model's self-forecast averages its\n"
        "own noisy samples while the peer is the same model for every target. Erratic forecasters\n"
        "carry inflated self-loss, and weaker models are noisier — so the weak end of this chart\n"
        "is a lower bound on self-knowledge rather than an estimate of it."
    )
    for x, body in ((x0 + pad, left), (x0 + 0.475, right)):
        fig.text(x, y0 + h - 0.042, body, fontsize=7.6, color="#333",
                 va="top", linespacing=1.45)


def figure_rq1(analysis: dict, *, cross_model: str | None = None,
               n_tasks: int | None = None) -> plt.Figure:
    rq1 = analysis["rq1_existence"]
    wins = rq1["per_model_wins"]
    condition = rq1.get("condition", "no_thinking")
    null = rq1.get("null", 0.5)

    items = sorted(wins.items(), key=lambda kv: kv[1])
    labels = [_short(k) for k, _ in items]
    vals = [v for _, v in items]
    colors = [_SELF_BETTER if v > null else _TIE if v == null else _PEER_BETTER
              for v in vals]

    counts = _scored_cell_counts(analysis, n_tasks)
    broken = analysis.get("missingness", {}).get("broken_forecast_rate", {})

    n_above = sum(v > null for v in vals)
    n_tied = sum(v == null for v in vals)
    n_below = sum(v < null for v in vals)

    fig, ax = plt.subplots(figsize=(13.2, 9.1))
    ax.barh(labels, vals, color=colors, height=0.68, zorder=3)

    x_lo, x_hi = 0.30, 0.80
    sep_x, wins_x, cells_x = 0.645, 0.700, 0.716   # numeric columns, clear of the bars
    ax.set_xlim(x_lo, x_hi)
    top = len(vals) - 0.5

    # Per-bar numbers live in fixed columns rather than at the bar tip: near the
    # null the tips bunch together and the labels would sit on the reference lines.
    for i, (model, v) in enumerate(items):
        ax.text(wins_x, i, f"{v:.3f}", va="center", ha="right",
                fontsize=9, fontweight="bold", color="#222", zorder=4)
        n = counts.get(model)
        if n:
            rate = broken.get(model, {}).get(condition, 0.0)
            prefix = "\u2264 " if rate else ""
            ax.text(cells_x, i, f"{prefix}{v * n:g} / {n}", va="center", ha="left",
                    fontsize=8.5, color="#555", family="monospace", zorder=4)
    ax.text(wins_x, top + 0.55, "wins", va="center", ha="right",
            fontsize=8.5, color="#333", style="italic")
    if counts:
        ax.text(cells_x, top + 0.55, "cells won / scored", va="center", ha="left",
                fontsize=8.5, color="#333", style="italic")
    ax.axvline(sep_x, color="#ddd", lw=1, zorder=1)

    # The null is the only reference worth drawing: a mean over 11 models with
    # unequal denominators is not a quantity anyone should read off this chart.
    ax.axvline(null, color="k", ls="--", lw=1.2, zorder=2)
    ax.text(null, top + 0.55, f" chance = {null:.2f}", va="center", ha="left",
            fontsize=9)

    # Which side of the line means what.
    ax.annotate("", xy=(null - 0.115, top + 1.35), xytext=(null - 0.004, top + 1.35),
                arrowprops=dict(arrowstyle="->", color=_PEER_BETTER, lw=1.4))
    peer_lbl = figmod.peer_phrase(cross_model)
    peer_abbr = figmod.peer_short(cross_model)
    ax.text(null - 0.12, top + 1.35,
            f"{peer_abbr} predicts this model\nbetter than it predicts itself",
            va="center", ha="right", fontsize=8.5, color=_PEER_BETTER)
    ax.annotate("", xy=(null + 0.115, top + 1.35), xytext=(null + 0.004, top + 1.35),
                arrowprops=dict(arrowstyle="->", color=_SELF_BETTER, lw=1.4))
    ax.text(null + 0.12, top + 1.35,
            f"the model predicts itself\nbetter than {peer_abbr} does",
            va="center", ha="left", fontsize=8.5, color=_SELF_BETTER)

    ax.set_ylim(-0.7, top + 1.95)
    ax.set_xticks([round(x_lo + 0.05 * i, 2) for i in range(7)])
    ax.set_xlabel("wins fraction — share of (task, dimension) cells the self-forecast won",
                  fontsize=10)
    ax.grid(axis="x", alpha=0.25, linestyle=":", zorder=0)
    ax.set_axisbelow(True)

    fig.suptitle(
        f"RQ1 — can a model forecast its own behaviour better than {peer_lbl} can?",
        fontsize=13, y=0.985)
    ax.set_title(
        f"condition: {condition}  ·  null {null:.2f}  ·  "
        f"{n_above} of {len(vals)} above chance, {n_tied} tied, {n_below} below",
        fontsize=10, pad=30)

    fig.text(
        0.008, 0.238,
        "How to read it — every model attempts 30 SWE-bench Verified tasks and forecasts its own behaviour on 10 dimensions; "
        f"{peer_lbl} forecasts the same task without being told whose behaviour it is predicting.\n"
        "Each (task, dimension) pair is one cell, scored by whichever forecast was closer to what actually happened (ties count 0.5). "
        "A bar is the share of that model's cells its own forecast won, so 0.50 is chance.",
        fontsize=8.5, color="#444", va="top", linespacing=1.5)

    _draw_point_estimate_caveat(fig, peer=peer_abbr)

    fig.tight_layout(rect=(0, 0.257, 1, 0.945))
    return fig


# ---------------------------------------------------------------- RQ2
def figure_rq2(analysis: dict) -> plt.Figure:
    rq2 = analysis["rq2_scaling"]
    solve = rq2["solve_rates"]
    w = rq2["per_model_wins"]

    xs = np.array([solve[k] for k in w])
    ys = np.array([w[k] for k in w])

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.scatter(xs, ys, s=60, color="#2a7fb8", zorder=3)
    for k in w:
        ax.annotate(_short(k), (solve[k], w[k]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(0.5, color="k", ls="--", lw=1, alpha=0.6)
    # Descriptive least-squares trend line (visual aid only -- the reported
    # statistic stays the rank-based Spearman rho in the title).
    slope, intercept = np.polyfit(xs, ys, 1)
    xline = np.linspace(xs.min(), xs.max(), 2)
    ax.plot(xline, slope * xline + intercept, color="#c22", lw=1.6, zorder=2,
            label=f"OLS trend (slope {slope:+.2f}, descriptive)")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.set_xlabel("empirical solve rate (capability)")
    ax.set_ylabel("wins fraction (self-prediction)")
    ax.set_title(
        f"RQ2 — Self-prediction vs capability\n"
        f"Spearman rho = {rq2['spearman_rho']:.3f}  (descriptive; not a significance test)"
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- RQ3
def figure_rq3(analysis: dict) -> plt.Figure:
    diff = analysis["rq3_elicitation"]["per_model_difference"]
    ditems = sorted(diff.items(), key=lambda kv: kv[1])
    dlabels = [_short(k) for k, _ in ditems]
    dvals = [v for _, v in ditems]
    dcolors = ["#2a7fb8" if v >= 0 else "#c44" for v in dvals]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(dlabels, dvals, color=dcolors)
    ax.axvline(0.0, color="k", lw=1)
    ax.set_xlabel("Δ wins = wins(thinking) − wins(no_thinking)")
    ax.set_title(
        f"RQ3 — Does 'thinking' elicit better self-prediction?\n"
        f"mean Δ = {analysis['rq3_elicitation']['point_estimate']:+.4f}  (null 0)"
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


def figure_dimensions(scores: dict, *, condition: str = "no_thinking",
                      cross_model: str | None = None) -> plt.Figure:
    """Forecast accuracy per dimension, averaged across the self-forecasting models.

    Two rows of bars over the same 10 dimensions:

    * **absolute** accuracy -- mean loss for self and peer in each dimension's
      own units. The binary dims share a Brier axis; the continuous ones are in
      tokens, steps, ratios and counts, which share nothing, so they get a
      separate log axis. Putting all ten on one linear scale would bury every
      binary bar under ``avg_tokens_step``; putting them on one *log* scale
      hides the self-vs-peer gap that is the entire point.
    * **comparative** accuracy -- the wins fraction, the only quantity here that
      *is* comparable across all ten, because binarising per cell discards units.
    """
    per_model = scores["scores"][condition]
    dims = [d.name for d in config.DIMENSIONS]
    binary = [d.name for d in config.DIMENSIONS if d.kind == "binary"]
    n_binary = len(binary)

    self_loss = [_mean_across_models(per_model, "mean_loss_self_per_dim", d) for d in dims]
    cross_loss = [_mean_across_models(per_model, "mean_loss_cross_per_dim", d) for d in dims]
    wins = [_mean_across_models(per_model, "wins_per_dim", d) for d in dims]

    x = np.arange(len(dims), dtype=float)
    n_cont = len(dims) - n_binary
    fig = plt.figure(figsize=(13.8, 8.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0],
                          width_ratios=[n_binary, n_cont])
    ax_bin = fig.add_subplot(gs[0, 0])
    ax_cont = fig.add_subplot(gs[0, 1])
    ax_win = fig.add_subplot(gs[1, :])

    # -- top: absolute accuracy, one axis per unit family ------------------
    w = 0.38
    for ax, lo, hi in ((ax_bin, 0, n_binary), (ax_cont, n_binary, len(dims))):
        s, c, xs = self_loss[lo:hi], cross_loss[lo:hi], x[lo:hi]
        ax.bar(xs - w / 2, s, w, color=figmod.SELF_COLORS[condition],
               label="the model forecasting itself", zorder=3)
        ax.bar(xs + w / 2, c, w, color=figmod.CROSS_COLORS[condition],
               label=f"{figmod.peer_phrase(cross_model)}", zorder=3)
        pad = 1.06 if ax is ax_cont else 1.02
        for xi, sv, cv in zip(xs, s, c):
            for pos, val in ((xi - w / 2, sv), (xi + w / 2, cv)):
                ax.text(pos, val * pad, f"{val:.3g}", ha="center", va="bottom",
                        fontsize=7.5, color="#333", zorder=4)
        ax.set_xlim(lo - 0.62, hi - 0.38)
        ax.set_xticks(xs, [_dim_label(d) for d in dims[lo:hi]],
                      fontsize=8, rotation=28, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", pad=2)
        ax.grid(axis="y", alpha=0.25, linestyle=":", zorder=0)
        ax.set_axisbelow(True)
    ax_cont.set_yscale("log")
    ax_bin.set_ylim(0, max(self_loss[:n_binary] + cross_loss[:n_binary]) * 1.30)
    cont_vals = self_loss[n_binary:] + cross_loss[n_binary:]
    ax_cont.set_ylim(min(cont_vals) * 0.5, max(cont_vals) * 4.0)
    ax_bin.set_ylabel("mean Brier loss\n(lower = more accurate)", fontsize=9)
    ax_cont.set_ylabel("mean absolute error, log scale\n(each dimension its own unit)",
                       fontsize=9)
    ax_cont.yaxis.set_label_position("right")
    ax_cont.yaxis.tick_right()
    # Upper right: p_solve is the tallest bar and sits on the left.
    ax_bin.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    ax_bin.set_title("binary dimensions — one shared Brier scale",
                     fontsize=9.5, color="#444", style="italic")
    ax_cont.set_title("continuous dimensions — four different units, never compare heights across bars",
                      fontsize=9.5, color="#444", style="italic")

    # -- bottom: comparative accuracy, unitless ---------------------------
    win_colors = [_SELF_BETTER if v > 0.5 else _TIE if v == 0.5 else _PEER_BETTER
                  for v in wins]
    ax_win.bar(x, wins, 0.6, color=win_colors, zorder=3)
    ax_win.axhline(0.5, color="k", ls="--", lw=1.2, zorder=4)
    ax_win.text(-0.55, 0.508, " chance = 0.50", va="bottom", ha="left", fontsize=8.5,
                zorder=5)
    for xi, v in zip(x, wins):
        ax_win.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#222", zorder=4)
    ax_win.set_ylim(0, max(wins) * 1.20)
    ax_win.set_xlim(-0.62, len(dims) - 0.38)
    ax_win.axvline(n_binary - 0.5, color="#999", lw=1.2, ls=":", zorder=1)
    ax_win.axvspan(n_binary - 0.5, len(dims) - 0.38, color="#f5f5f5", zorder=0)
    ax_win.grid(axis="y", alpha=0.25, linestyle=":", zorder=0)
    ax_win.set_axisbelow(True)
    ax_win.set_ylabel(f"wins fraction vs {figmod.peer_short(cross_model)}"
                      "\n(comparable across all ten)", fontsize=9)
    # Dimension names are long enough that horizontal ticks collide at 10 across.
    ax_win.set_xticks(x, [_dim_label(d) for d in dims],
                      fontsize=9, rotation=25, ha="right", rotation_mode="anchor")

    peer_lbl = figmod.peer_phrase(cross_model)
    peer_abbr = figmod.peer_short(cross_model)
    fig.suptitle(
        "Forecast accuracy by dimension — what do models actually know about themselves?\n"
        f"mean over {len(per_model)} models, condition: {condition}",
        fontsize=13, y=0.985)

    fig.text(
        0.008, 0.012,
        f"How to read it — each model forecasts its own behaviour on 10 dimensions; {peer_lbl} forecasts the same behaviour without being told whose it is. "
        "The top row is raw accuracy in each dimension's own units, so heights compare only within a dimension.\n"
        f"The bottom row is the wins fraction, which is comparable across dimensions because each cell is reduced to who was closer. Blue = the model beat {peer_abbr}, red = {peer_abbr} beat the model.\n"
        "Point estimates only, averaged across models — no confidence intervals, and noisier forecasters carry inflated self-loss (confound W2), so the self bars are pessimistic by an unknown amount.",
        fontsize=8, color="#444", va="bottom", linespacing=1.5)

    # Explicit margins, not tight_layout: the two-then-one gridspec is one of the
    # layouts tight_layout warns it cannot place, and it silently leaves the
    # rotated tick labels sitting on top of the footer.
    fig.subplots_adjust(left=0.075, right=0.915, top=0.855, bottom=0.22,
                        hspace=0.38, wspace=0.10)
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

    # RQ1's annotations need two facts analysis.json does not carry: who the peer
    # was (scores.json) and how many tasks the dropped-cell *rates* are over
    # (run_summary.json). Both are optional; the figure degrades rather than guesses.
    def _read(name: str, key: str):
        path = results_dir / name
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8")).get(key)

    figures = {
        "rq1_wins_by_model": figure_rq1(
            analysis,
            cross_model=_read("scores.json", "cross_model"),
            n_tasks=_read("run_summary.json", "n_tasks"),
        ),
        "rq2_wins_vs_solverate": figure_rq2(analysis),
        "rq3_thinking_delta": figure_rq3(analysis),
    }
    # The per-dimension figure needs per-model per-dim losses, which only
    # scores.json carries; skip it rather than fail when that file is absent.
    scores_path = results_dir / "scores.json"
    if scores_path.exists():
        scores = json.loads(scores_path.read_text(encoding="utf-8"))
        if scores.get("scores"):
            figures["dimension_accuracy"] = figure_dimensions(
                scores, cross_model=scores.get("cross_model"))

    murphy_path = results_dir / "murphy.json"
    if murphy_path.exists():
        murphy = json.loads(murphy_path.read_text(encoding="utf-8"))
        figures["murphy_decomposition"] = figmod.figure_c1_murphy_components(
            murphy=murphy, dimensions=config.DIMENSIONS)
        figures["reliability_diagrams"] = figmod.figure_c2_reliability_diagrams(
            murphy=murphy, dimensions=config.DIMENSIONS)

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
