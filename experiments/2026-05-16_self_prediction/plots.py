"""Experiment 1 figures (EVAL_SPEC_V1.md sec 15.6 + 15.7).

Five figures cover every pre-registered result and the §15.7 native-unit
transparency layer:

  A1 -- ``wins(M, condition)`` grouped bar chart      (RQ1 existence)
  A2 -- ``wins`` vs empirical solve rate scatter      (RQ2 scaling)
  A3 -- per-model Δwins forest plot                   (RQ3 elicitation)
  B1 -- per-dim self vs cross mean loss bars          (sec 15.7 transparency)
  B2 -- per-(model, dim) wins-fraction heatmap        (sec 15.7 transparency)

Every figure shows **both forecast conditions together** (`no_thinking` and
`thinking`); the user reads the thinking effect off the same plot as the
existence claim. A3 is unchanged (Δwins already encodes both).

MVP scope: the inferential statistics tier is pruned, so the figures render
**point estimates only** -- no confidence-interval bands, p-value annotations,
or fitted scaling curves. A single cross-predictor is rendered under
``results/figures/primary__<cross_slug>/``.

Read-only: every figure is built from ``results/scores.json`` and
``results/analysis.json`` -- no scoring logic, no ``ScoringData`` reconstruction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

# Headless-safe backend before any pyplot import; the test suite and CI both
# run without a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

# Make src/ + this experiment importable when called directly.
_EXP_DIR = Path(__file__).parent.resolve()
_ROOT = _EXP_DIR.parents[1]
for _p in (str(_ROOT), str(_EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: E402  (used for the default (model -> family) map)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="paper")
_FAMILY_PALETTE = "colorblind"
_NEUTRAL_GREY = "#6b6b6b"

#: Condition colors -- distinct enough for grouped bars / scatter markers /
#: the per-condition heatmap titles.
_CONDITION_COLORS = {
    "no_thinking": "#3b6fb6",   # blue
    "thinking":    "#d4663b",   # orange
}
#: Self-vs-cross colors within a single condition (B1's 4-bar groups).
_SELF_COLORS = {
    "no_thinking": "#3b6fb6",   # solid blue
    "thinking":    "#7aa3d4",   # lighter blue
}
_CROSS_COLORS = {
    "no_thinking": "#d4663b",   # solid orange
    "thinking":    "#e9a380",   # lighter orange
}

#: Pretty axis labels per dimension -- native units the §15.7 transparency
#: layer reads literally as ("self off by X tokens", "by Y steps", etc.).
_DIM_UNITS: dict[str, str] = {
    "p_solve": "Brier",
    "p_first_attempt_correct": "Brier",
    "documentation": "Brier",
    "defensive_code": "Brier",
    "express_uncertainty": "Brier",
    "avg_tokens_step": "tokens",
    "total_expected_steps": "steps",
    "explore_edit_ratio": "ratio",
}

_CONDITIONS = ("no_thinking", "thinking")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _family_color_map(families: list[str]) -> dict[str, tuple]:
    palette = sns.color_palette(_FAMILY_PALETTE, n_colors=max(1, len(families)))
    return dict(zip(families, palette))


def _default_families_by_model() -> dict[str, str]:
    """The canonical (model_id -> family) mapping from ``config.MODELS``."""
    return {m.openrouter_id: m.family for m in config.MODELS}


def _short_label(model_id: str) -> str:
    """A compact axis label -- strip the OpenRouter ``vendor/`` prefix."""
    return model_id.split("/", 1)[-1]


def _slug(model_id: str) -> str:
    """Filesystem-safe slug (matches ``config.ModelSpec.slug``)."""
    return model_id.replace("/", "__")


def _sort_by_solve_rate(
    models: list[str], solve_rates: dict[str, float], descending: bool = True
) -> list[str]:
    return sorted(models, key=lambda m: solve_rates.get(m, 0.0), reverse=descending)


def _value(per_cond_dict: dict, cond: str, key: str, default=float("nan")) -> float:
    """Safe nested lookup for ``{cond: {key: ...}}`` style maps."""
    return per_cond_dict.get(cond, {}).get(key, default)


# --------------------------------------------------------------------------
# A1 -- grouped wins-fraction bar chart (both conditions)
# --------------------------------------------------------------------------
def figure_a1_wins_bar(
    *, scores: dict, analysis: dict, families_by_model: dict[str, str],
) -> plt.Figure:
    """RQ1 visualisation: per-model ``wins(M)`` for both conditions side-by-side.

    Two horizontal bars per model -- one per condition -- with the pooled
    ``no_thinking`` mean shown as a vertical line (the canonical RQ1 reading).
    The reference at 0.5 is the RQ1 null.
    """
    rq1 = analysis["rq1_existence"]
    per_model_nt = rq1["per_model_wins"]
    # The model set is the IAI pool (cross excluded), identical across conditions.
    models = sorted(per_model_nt.keys(), key=lambda m: per_model_nt[m])
    wins_nt = scores["wins"]["no_thinking"]
    wins_th = scores["wins"]["thinking"]

    fig, ax = plt.subplots(figsize=(8.0, 0.55 * len(models) + 1.8))
    y = np.arange(len(models))
    h = 0.38
    ax.barh(y - h / 2, [wins_nt.get(m, np.nan) for m in models], height=h,
            color=_CONDITION_COLORS["no_thinking"], edgecolor="white",
            label="no_thinking")
    ax.barh(y + h / 2, [wins_th.get(m, np.nan) for m in models], height=h,
            color=_CONDITION_COLORS["thinking"], edgecolor="white",
            label="thinking")
    ax.set_yticks(y, [_short_label(m) for m in models])
    ax.axvline(0.5, color=_NEUTRAL_GREY, linestyle="--", linewidth=1.2,
               label="RQ1 null (0.5)")

    # Pooled (no_thinking) mean line -- the canonical RQ1 point estimate.
    ax.axvline(rq1["point_estimate"],
               color=_CONDITION_COLORS["no_thinking"], linewidth=1.4,
               label=f"no_thinking mean = {rq1['point_estimate']:.3f}")

    ax.set_xlabel("wins(M, condition)")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("RQ1 — fraction of (task, dim) cells where self beats cross")
    ax.legend(loc="lower right", framealpha=0.95, fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# A2 -- scaling scatter (wins ~ solve rate) for both conditions
# --------------------------------------------------------------------------
def figure_a2_scaling(
    *, scores: dict, analysis: dict, families_by_model: dict[str, str],
) -> plt.Figure:
    """RQ2 visualisation: ``wins(M, cond)`` against empirical solve rate, both
    conditions overlaid.

    One scatter point per (model, condition) -- filled marker for
    ``no_thinking``, hollow for ``thinking`` -- with the family-coloured dot.
    Descriptive scatter only: the fitted scaling curve belongs to the pruned
    statistics tier and is not drawn.
    """
    solve_rates = analysis["solve_rates"]
    wins_nt = scores["wins"]["no_thinking"]
    wins_th = scores["wins"]["thinking"]
    models = list(wins_nt.keys())
    families_present = sorted({families_by_model[m] for m in models if m in families_by_model})
    color_map = _family_color_map(families_present)

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    for m in models:
        fam_color = color_map.get(families_by_model.get(m, ""), _NEUTRAL_GREY)
        x = solve_rates.get(m, np.nan)
        # filled = no_thinking, hollow ring = thinking
        ax.scatter(x, wins_nt.get(m, np.nan),
                   facecolor=fam_color, edgecolor="white", s=90,
                   linewidth=1.2, zorder=3)
        ax.scatter(x, wins_th.get(m, np.nan),
                   facecolor="white", edgecolor=fam_color, s=90,
                   linewidth=2.0, zorder=3)
        ax.annotate(
            _short_label(m), (x, wins_nt.get(m, np.nan)),
            xytext=(7, 4), textcoords="offset points", fontsize=8, color="#222",
        )

    ax.axhline(0.5, color=_NEUTRAL_GREY, linestyle="--", linewidth=1.0,
               label="RQ1 null (0.5)")

    ax.set_xlabel("empirical solve rate (mean p_solve)")
    ax.set_ylabel("wins(M, condition)")
    ax.set_title("RQ2 — introspection vs model capability (descriptive)")
    ax.set_ylim(0.0, 1.0)

    family_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_map[f], label=f) for f in families_present
    ]
    cond_handles = [
        plt.Line2D([], [], color="#444", marker="o", linestyle="", markersize=8,
                   label="no_thinking (filled)"),
        plt.Line2D([], [], color="#444", marker="o", linestyle="",
                   markerfacecolor="white", markeredgewidth=2, markersize=8,
                   label="thinking (hollow)"),
    ]
    ax.legend(handles=family_handles + cond_handles + ax.get_legend_handles_labels()[0],
              loc="lower right", framealpha=0.95, fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# A3 -- Δwins forest plot (unchanged: Δ already encodes both conditions)
# --------------------------------------------------------------------------
def figure_a3_delta_wins_forest(
    *, scores: dict, analysis: dict, families_by_model: dict[str, str],
) -> plt.Figure:
    """RQ3 visualisation: per-model Δwins(M) = wins(thinking) − wins(no_thinking).

    Per-model dots, family-coloured; pooled mean shown as a diamond;
    reference line at Δwins = 0.
    """
    rq3 = analysis["rq3_elicitation"]
    per_model = rq3["per_model_difference"]
    models = sorted(per_model.keys(), key=lambda m: per_model[m])
    families_present = sorted({families_by_model[m] for m in models if m in families_by_model})
    color_map = _family_color_map(families_present)

    fig, ax = plt.subplots(figsize=(8.0, 0.50 * len(models) + 1.8))
    y = np.arange(len(models))
    for i, m in enumerate(models):
        ax.scatter(per_model[m], y[i],
                   color=color_map.get(families_by_model.get(m, ""), _NEUTRAL_GREY),
                   s=80, edgecolor="white", linewidth=1.0, zorder=3)
    pooled_y = -1
    ax.scatter(rq3["point_estimate"], pooled_y, color="black", marker="D", s=110,
               zorder=4, label=f"pooled mean = {rq3['point_estimate']:.3f}")
    ax.set_yticks(list(y) + [pooled_y],
                  [_short_label(m) for m in models] + ["pooled mean"])
    ax.axvline(0.0, color=_NEUTRAL_GREY, linestyle="--", linewidth=1.2,
               label="RQ3 null (Δ = 0)")

    if rq3.get("excluded_models"):
        excl = ", ".join(_short_label(m) for m in rq3["excluded_models"])
        ax.text(0.02, 0.02, f"excluded: {excl}", transform=ax.transAxes,
                fontsize=8, color="#666", va="bottom")

    ax.set_xlabel("Δwins(M)  =  wins(thinking) − wins(no_thinking)")
    ax.set_title("RQ3 — does thinking elicit better self-prediction?")
    family_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_map[f], label=f) for f in families_present
    ]
    ax.legend(handles=family_handles + ax.get_legend_handles_labels()[0],
              loc="lower right", framealpha=0.95, fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# B1 -- grouped per-dim self vs cross mean loss (both conditions, native units)
# --------------------------------------------------------------------------
def figure_b1_per_dim_loss(
    *, scores: dict, analysis: dict, dimensions,
) -> plt.Figure:
    """§15.7 transparency: per-dim mean self loss vs mean cross loss, both
    conditions side-by-side.

    Each dimension gets its own subplot (its own y-axis, native units). Inside
    each subplot: 4 bars in two condition-pair groups
    ``[self_nt, cross_nt] | [self_t, cross_t]``. BH-FDR significance from the
    canonical (no_thinking) exploratory test still anchors the asterisk in
    the title.
    """
    per_dim_p_bh = analysis.get("exploratory", {}).get("per_dimension_pvalue_bh", {})

    # Grid sized to the dimension count (A6: 10 dims -> 3x4). Generalises so the
    # battery can grow without a hardcoded subplot count.
    n_dims = len(dimensions)
    ncols = 4
    nrows = math.ceil(n_dims / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.7 * nrows))
    flat = np.atleast_1d(axes).flatten()
    for ax in flat:
        ax.set_visible(False)

    for k, dim in enumerate(dimensions):
        name = dim.name if hasattr(dim, "name") else dim[0]
        ax = flat[k]
        ax.set_visible(True)

        bars: list[tuple[float, float, str, str]] = []
        # (x, value, color, label)
        x_positions = [0.0, 0.7, 1.8, 2.5]
        advantages: dict[str, float] = {}
        for j, cond in enumerate(_CONDITIONS):
            cond_models = list(scores["scores"][cond].keys())
            self_vals = np.array(
                [scores["scores"][cond][m]["mean_loss_self_per_dim"].get(name, np.nan)
                 for m in cond_models], dtype=float,
            )
            cross_vals = np.array(
                [scores["scores"][cond][m]["mean_loss_cross_per_dim"].get(name, np.nan)
                 for m in cond_models], dtype=float,
            )
            mean_self = (float(np.nanmean(self_vals))
                         if np.any(~np.isnan(self_vals)) else float("nan"))
            mean_cross = (float(np.nanmean(cross_vals))
                          if np.any(~np.isnan(cross_vals)) else float("nan"))
            advantages[cond] = mean_cross - mean_self
            bars.append((x_positions[2 * j],     mean_self,
                         _SELF_COLORS[cond], f"self ({cond[:2]})"))
            bars.append((x_positions[2 * j + 1], mean_cross,
                         _CROSS_COLORS[cond], f"cross ({cond[:2]})"))

        xs = [b[0] for b in bars]
        ys = [b[1] for b in bars]
        colors = [b[2] for b in bars]
        ax.bar(xs, ys, width=0.6, color=colors, edgecolor="white")
        for x, val in zip(xs, ys):
            if not np.isnan(val):
                ax.text(x, val, f"{val:.3f}" if val < 10 else f"{val:.1f}",
                        ha="center", va="bottom", fontsize=7)
        ax.set_xticks([0.35, 2.15], ["no_thinking", "thinking"], fontsize=8)
        # under-axis self/cross sub-tick guide
        ax.set_xticks(xs, ["s", "c", "s", "c"], minor=True)
        ax.tick_params(axis="x", which="minor", labelsize=7, pad=12, length=0)

        units = _DIM_UNITS.get(name, "")
        sig = ""
        p_bh = per_dim_p_bh.get(name)
        if p_bh is not None and not np.isnan(p_bh) and p_bh < 0.05:
            sig = " *"
        ax.set_title(
            f"{name}{sig}\nadv: nt = {advantages['no_thinking']:+.3f}, "
            f"t = {advantages['thinking']:+.3f} ({units})",
            fontsize=8,
        )
        ax.set_ylabel(f"mean loss ({units})", fontsize=8)
        ax.margins(y=0.20)

    # Legend on the figure once.
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=_SELF_COLORS["no_thinking"], label="self (no_thinking)"),
        plt.Rectangle((0, 0), 1, 1, color=_CROSS_COLORS["no_thinking"], label="cross (no_thinking)"),
        plt.Rectangle((0, 0), 1, 1, color=_SELF_COLORS["thinking"], label="self (thinking)"),
        plt.Rectangle((0, 0), 1, 1, color=_CROSS_COLORS["thinking"], label="cross (thinking)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=9, framealpha=0.95)
    fig.suptitle(
        "§15.7 transparency — per-dimension mean loss in native units\n"
        "darker = no_thinking, lighter = thinking; * = BH-FDR-significant (p < 0.05)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    return fig


# --------------------------------------------------------------------------
# B2 -- side-by-side per-(model, dim) wins heatmaps (both conditions)
# --------------------------------------------------------------------------
def figure_b2_wins_heatmap(
    *, scores: dict, analysis: dict, dimensions,
) -> plt.Figure:
    """§15.7 transparency: where (model, dim) introspection wins vs loses,
    one heatmap per condition.

    Rows are the same IAI-eligible models in both panels (sorted by empirical
    solve rate, strongest at top); columns are the 8 dimensions in spec order.
    Diverging RdBu colormap is centred at 0.5 -- red = self wins the cell,
    blue = self loses.
    """
    solve_rates = analysis["solve_rates"]
    # Use the no_thinking row set as the canonical model list (the IAI pool is
    # identical across conditions by construction).
    models = _sort_by_solve_rate(
        list(scores["scores"]["no_thinking"].keys()), solve_rates, descending=True,
    )
    dim_names = [d.name if hasattr(d, "name") else d[0] for d in dimensions]

    fig, axes = plt.subplots(
        1, 2, figsize=(2.0 * len(dim_names) + 2.5, 0.55 * len(models) + 2.6),
        sharey=True,
    )
    last_im = None
    for ax, cond in zip(axes, _CONDITIONS):
        per_model = scores["scores"][cond]
        matrix = np.full((len(models), len(dim_names)), np.nan)
        for i, m in enumerate(models):
            wpd = per_model.get(m, {}).get("wins_per_dim", {})
            for j, name in enumerate(dim_names):
                if name in wpd:
                    matrix[i, j] = wpd[name]
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=0.0, vmax=1.0, aspect="auto")
        last_im = im
        ax.set_xticks(range(len(dim_names)), dim_names, rotation=40, ha="right")
        ax.set_title(cond, fontsize=10)
        for i in range(len(models)):
            for j in range(len(dim_names)):
                v = matrix[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7,
                            color="black" if 0.35 <= v <= 0.65 else "white")
    axes[0].set_yticks(range(len(models)), [_short_label(m) for m in models])
    cbar = fig.colorbar(last_im, ax=axes, shrink=0.85,
                        label="wins fraction (self vs cross)")
    cbar.ax.axhline(0.5, color="black", linewidth=1.0)
    fig.suptitle(
        "§15.7 transparency — where introspection wins  "
        "(rows sorted by empirical solve rate, top = strongest)",
        fontsize=11, y=0.99,
    )
    # Reserve headroom for the suptitle (rect bottom=0.04 keeps room for x-tick labels).
    fig.subplots_adjust(top=0.86, bottom=0.18, wspace=0.10)
    return fig


# --------------------------------------------------------------------------
# Per-cross iteration & rendering
# --------------------------------------------------------------------------
def _iter_cross_views(scores_obj: dict, analysis_obj: dict):
    """Yield ``(role, slug, scores_view, analysis_view)`` for the cross-predictor.

    ``scores_view`` is ``scores_obj`` itself (``wins`` and ``scores`` at the top
    level) and ``analysis_view`` is the top-level ``analysis_obj``. The MVP
    renders a single cross-predictor (the alt-cross sensitivity battery is
    pruned), so exactly one view is yielded.
    """
    primary_cross = scores_obj.get("cross_model", "primary")
    yield "primary", _slug(primary_cross), scores_obj, analysis_obj


def _render_one_cross(
    role: str, cross_slug: str, scores_view: dict, analysis_view: dict,
    figures_root: Path,
    *, families_by_model: dict[str, str], dimensions, formats: tuple[str, ...],
) -> list[Path]:
    out_dir = figures_root / f"{role}__{cross_slug}"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "a1_wins_bar": figure_a1_wins_bar(
            scores=scores_view, analysis=analysis_view,
            families_by_model=families_by_model,
        ),
        "a2_scaling": figure_a2_scaling(
            scores=scores_view, analysis=analysis_view,
            families_by_model=families_by_model,
        ),
        "a3_delta_wins_forest": figure_a3_delta_wins_forest(
            scores=scores_view, analysis=analysis_view,
            families_by_model=families_by_model,
        ),
        "b1_per_dim_loss": figure_b1_per_dim_loss(
            scores=scores_view, analysis=analysis_view, dimensions=dimensions,
        ),
        "b2_wins_heatmap": figure_b2_wins_heatmap(
            scores=scores_view, analysis=analysis_view, dimensions=dimensions,
        ),
    }
    written: list[Path] = []
    for name, fig in figures.items():
        for fmt in formats:
            out = out_dir / f"{name}.{fmt}"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            written.append(out)
        plt.close(fig)
    return written


def render_all(
    scores: dict, analysis: dict, figures_dir: Path,
    *, families_by_model: dict[str, str] | None = None,
    dimensions=None,
    formats=("png",),
) -> list[Path]:
    """Render the five figures for the cross-predictor under ``figures_dir``.

    The cross-predictor lands in a ``primary__<slug>/`` subfolder. Returns the
    list of written file paths in render order.
    """
    families_by_model = families_by_model or _default_families_by_model()
    dimensions = dimensions if dimensions is not None else config.DIMENSIONS
    figures_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for role, cross_slug, sv, av in _iter_cross_views(scores, analysis):
        written.extend(_render_one_cross(
            role, cross_slug, sv, av, figures_dir,
            families_by_model=families_by_model, dimensions=dimensions,
            formats=formats,
        ))
    return written


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _load_results(results_dir: Path) -> tuple[dict, dict]:
    scores_path = results_dir / "scores.json"
    analysis_path = results_dir / "analysis.json"
    if not scores_path.exists() or not analysis_path.exists():
        raise SystemExit(
            f"plots: missing scores.json / analysis.json under {results_dir} -- "
            f"run.py must run first."
        )
    return (json.loads(scores_path.read_text(encoding="utf-8")),
            json.loads(analysis_path.read_text(encoding="utf-8")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render Experiment 1 figures from results/")
    parser.add_argument("--results-dir", default=str(_EXP_DIR / "results"),
                        help="Directory containing scores.json + analysis.json.")
    parser.add_argument("--figures-dir", default=None,
                        help="Output directory (default: <results-dir>/figures).")
    parser.add_argument("--format", action="append", choices=["png", "pdf"], default=None,
                        help="Output format(s); repeatable. Default: png.")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir) if args.figures_dir else results_dir / "figures"
    formats = tuple(args.format) if args.format else ("png",)

    scores, analysis = _load_results(results_dir)
    written = render_all(scores, analysis, figures_dir, formats=formats)
    print(f"[plots] wrote {len(written)} figure file(s) under {figures_dir}")
    cross_folders = sorted({p.parent.name for p in written})
    for folder in cross_folders:
        print(f"  - {folder}/  ({sum(1 for p in written if p.parent.name == folder)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
