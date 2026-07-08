"""Headline figures for Experiment 1 -- the flat ``<results-dir>/plots/`` set.

Renders the three RQ readouts from ``analysis.json`` and, when ``murphy.json``
exists (stage 8), the two Murphy-decomposition figures alongside them:

    rq1_wins_by_model.png       rq2_wins_vs_solverate.png   rq3_thinking_delta.png
    murphy_decomposition.png    reliability_diagrams.png

Called automatically by ``run.py`` stage 9; also runs standalone:

    python make_plots.py [--results-dir results_real]
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
def figure_rq1(analysis: dict) -> plt.Figure:
    wins = analysis["rq1_existence"]["per_model_wins"]
    items = sorted(wins.items(), key=lambda kv: kv[1])
    labels = [_short(k) for k, _ in items]
    vals = [v for _, v in items]
    colors = ["#2a7fb8" if v >= 0.5 else "#c44" for v in vals]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(labels, vals, color=colors)
    ax.axvline(0.5, color="k", ls="--", lw=1)
    ax.text(0.5, len(vals) - 0.4, "  chance = 0.50", va="top", fontsize=9)
    ax.set_xlim(0.30, 0.70)
    ax.set_xlabel("wins fraction (self-forecast beats cross-predictor)")
    ax.set_title(
        f"RQ1 — Introspective self-knowledge by model (no_thinking)\n"
        f"mean = {analysis['rq1_existence']['point_estimate']:.3f}  (null 0.5)"
    )
    fig.tight_layout()
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
        "rq1_wins_by_model": figure_rq1(analysis),
        "rq2_wins_vs_solverate": figure_rq2(analysis),
        "rq3_thinking_delta": figure_rq3(analysis),
    }
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
    parser.add_argument("--results-dir", default=str(_EXP_DIR / "results_real"),
                        help="Directory containing analysis.json (+ optional murphy.json).")
    args = parser.parse_args(argv)
    written = render_headline_plots(Path(args.results_dir))
    print("wrote:")
    for p in written:
        print("  ", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
