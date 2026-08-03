"""Experiment 1 figures (EVAL_SPEC_V1.md sec 15.6 + 15.7).

Five figures cover every pre-registered result and the §15.7 native-unit
transparency layer:

  A1 -- ``wins(M, condition)`` grouped bar chart      (RQ1 existence)
  A2 -- ``wins`` vs empirical solve rate scatter      (RQ2 scaling)
  A3 -- per-model Δwins forest plot                   (RQ3 elicitation)
  B1 -- per-dim self vs cross mean loss bars          (sec 15.7 transparency)
  B2 -- per-(model, dim) wins-fraction heatmap        (sec 15.7 transparency)

Two further downstream figures render when ``results/murphy.json`` exists
(produced by ``murphy.py`` -- the Murphy decomposition of the binary dims):

  C1 -- Murphy components (REL / RES vs the UNC floor), mean across models
  C2 -- pooled reliability diagrams (calibration curves) per binary dim

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
SELF_COLORS = {
    "no_thinking": "#3b6fb6",   # solid blue
    "thinking":    "#7aa3d4",   # lighter blue
}
CROSS_COLORS = {
    "no_thinking": "#d4663b",   # solid orange
    "thinking":    "#e9a380",   # lighter orange
}

#: Pretty axis labels per dimension -- native units the §15.7 transparency
#: layer reads literally as ("self off by X tokens", "by Y steps", etc.).
DIM_UNITS: dict[str, str] = {
    "p_solve": "Brier",
    "p_first_attempt_correct": "Brier",
    "documentation": "Brier",
    "defensive_code": "Brier",
    "express_uncertainty": "Brier",
    "avg_tokens_step": "tokens",
    "total_expected_steps": "steps",
    "explore_edit_ratio": "ratio",
    "mean_action_block_size": "cmds/step",
    "n_constraint_violations": "count",
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


def peer_phrase(cross_model: str | None) -> str:
    """How the peer forecast is named in figure text, e.g. "glm5 forecasting anonymous".

    Figures used to label this series "cross", which overstated what was run in two
    ways: only *one* model ever produced a peer forecast (the stage-4 designated
    cross-predictor), and it was never told whose behaviour it was predicting. The
    label now says both, so a reader cannot mistake it for a panel of peers or for
    a peer that knew its subject.

    Derived from the data, never hardcoded -- the predictor is whichever model tops
    the solve-rate ranking at stage 4, so a re-run on different models relabels
    itself.
    """
    if not cross_model:
        return "peer forecasting anonymous"
    spec = config.MODEL_BY_ID.get(cross_model)
    if spec is not None:
        # "GLM-5" -> "glm5", "Kimi K2.5" -> "kimik2.5" -- compact enough for legends.
        name = spec.name.lower().replace(" ", "").replace("-", "")
    else:
        name = _short_label(cross_model).replace("-", "")
    return f"{name} forecasting anonymous"


def peer_short(cross_model: str | None) -> str:
    """Just the predictor's name, for annotations with no room for the phrase."""
    return _short_label(cross_model) if cross_model else "peer"


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
    ax.set_title("RQ1 — fraction of (task, dim) cells where self beats "
                 f"{peer_phrase(scores.get('cross_model'))}")
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

    # Descriptive Spearman rho (no_thinking) -- monotonic-association summary,
    # explicitly an upper bound (W2 variance confound), not a significance test.
    rho = analysis.get("rq2_scaling", {}).get("spearman_rho")
    if rho is not None and not (isinstance(rho, float) and np.isnan(rho)):
        ax.text(0.02, 0.97,
                f"Spearman ρ = {rho:.3f}\n(upper bound; not a test)",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#cccccc"))

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
    ``[self_nt, cross_nt] | [self_t, cross_t]``. Descriptive means only -- the
    per-dimension significance test (and its asterisk) was part of the pruned
    statistics tier.
    """
    # Grid sized to the dimension count (A6: 10 dims -> 3x4). Generalises so the
    # battery can grow without a hardcoded subplot count.
    peer_lbl = peer_phrase(scores.get("cross_model"))
    peer_abbr = peer_short(scores.get("cross_model"))
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
                         SELF_COLORS[cond], f"self ({cond[:2]})"))
            bars.append((x_positions[2 * j + 1], mean_cross,
                         CROSS_COLORS[cond], f"{peer_abbr} ({cond[:2]})"))

        xs = [b[0] for b in bars]
        ys = [b[1] for b in bars]
        colors = [b[2] for b in bars]
        ax.bar(xs, ys, width=0.6, color=colors, edgecolor="white")
        for x, val in zip(xs, ys):
            if not np.isnan(val):
                ax.text(x, val, f"{val:.3f}" if val < 10 else f"{val:.1f}",
                        ha="center", va="bottom", fontsize=7)
        ax.set_xticks([0.35, 2.15], ["no_thinking", "thinking"], fontsize=8)
        # under-axis sub-tick guide: s = self, p = peer (spelled out in the legend)
        ax.set_xticks(xs, ["s", "p", "s", "p"], minor=True)
        ax.tick_params(axis="x", which="minor", labelsize=7, pad=12, length=0)

        units = DIM_UNITS.get(name, "")
        ax.set_title(
            f"{name}\nadv: nt = {advantages['no_thinking']:+.3f}, "
            f"t = {advantages['thinking']:+.3f} ({units})",
            fontsize=8,
        )
        ax.set_ylabel(f"mean loss ({units})", fontsize=8)
        ax.margins(y=0.20)

    # Legend on the figure once.
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=SELF_COLORS["no_thinking"], label="self (no_thinking)"),
        plt.Rectangle((0, 0), 1, 1, color=CROSS_COLORS["no_thinking"],
                      label=f"{peer_lbl} (no_thinking)"),
        plt.Rectangle((0, 0), 1, 1, color=SELF_COLORS["thinking"], label="self (thinking)"),
        plt.Rectangle((0, 0), 1, 1, color=CROSS_COLORS["thinking"],
                      label=f"{peer_lbl} (thinking)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=9, framealpha=0.95)
    fig.suptitle(
        "§15.7 transparency — per-dimension mean loss in native units\n"
        "darker = no_thinking, lighter = thinking",
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
    solve rate, strongest at top); columns are the 10 dimensions in spec order.
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
                        label=f"wins fraction (self vs {peer_phrase(scores.get('cross_model'))})")
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
# C1 -- Murphy decomposition components (binary dims; downstream analysis)
# --------------------------------------------------------------------------
def _binary_dim_names(dimensions) -> list[str]:
    """Names of the binary (Brier-scored) dimensions, in battery order."""
    out = []
    for d in dimensions:
        name = d.name if hasattr(d, "name") else d[0]
        kind = d.kind if hasattr(d, "kind") else d[1]
        if kind == "binary":
            out.append(name)
    return out


#: Axes label marking the C1 explainer panel (it is not a data subplot).
MURPHY_KEY_LABEL = "murphy-key"

#: Legend text for the C1 grid. Kept beside the figure it explains so the two
#: cannot drift apart; the terms are the Murphy components, not our coinages.
_MURPHY_KEY_TERMS = (
    ("REL — lower is better",
     "reliability (miscalibration): it states 0.7 where 0.3\n"
     "happens. 0 = perfectly calibrated."),
    ("RES — higher is better",
     "resolution (discrimination), subtracted because it helps:\n"
     "does the forecast separate cells where the outcome happens\n"
     "from those where it doesn't? Ceiling is UNC."),
    ("UNC — fixed floor",
     "uncertainty (base-rate variance p(1\u2212p)): the floor neither\n"
     "side can beat, here the agent's own rollout-to-rollout\n"
     "randomness. Same for both forecasters, drawn as the dashed line."),
)

_MURPHY_KEY_CODA = (
    "A perfect forecaster: REL 0, RES = UNC \u2192 Brier 0.\n"
    "One that only states the base rate: REL 0, RES 0 \u2192 Brier = UNC.\n"
    "Tall left-hand bars mean miscalibrated; flat right-hand bars mean\n"
    "uninformative \u2014 and only the second cannot be fixed after the fact."
)


def _draw_murphy_key(ax, *, body_size: float = 7.2) -> None:
    """Render the REL/RES/UNC explainer into a spare subplot slot.

    Blocks are walked down the panel by line count rather than placed at hand
    tuned offsets -- the term bodies are prose and any future edit to their
    length would silently overlap the block beneath.
    """
    ax.set_visible(True)
    ax.axis("off")
    ax.set_label(MURPHY_KEY_LABEL)   # so callers/tests can tell it from a data panel
    ax.add_patch(plt.Rectangle(
        (0.01, 0.02), 0.98, 0.96, transform=ax.transAxes, clip_on=False,
        facecolor="#fbfbfb", edgecolor="#dcdcdc", linewidth=1.0, zorder=0))

    ax.text(0.5, 0.952, "How to read these panels", transform=ax.transAxes,
            ha="center", va="top", fontsize=9.5, fontweight="bold")
    ax.text(0.5, 0.872, "Brier  =  REL \u2212 RES + UNC", transform=ax.transAxes,
            ha="center", va="top", fontsize=10.5, family="monospace")
    ax.text(0.5, 0.803, "CORP estimator \u2014 exact identity, non-negative components",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.8,
            color="#777", style="italic")

    step, gap, y = 0.052, 0.018, 0.742
    for term, body in _MURPHY_KEY_TERMS:
        ax.text(0.045, y, term, transform=ax.transAxes, va="top",
                fontsize=8.5, fontweight="bold", color="#222")
        ax.text(0.21, y, body, transform=ax.transAxes, va="top",
                fontsize=body_size, color="#333", linespacing=1.45)
        y -= step * (body.count("\n") + 1) + gap

    ax.text(0.045, y - 0.01, _MURPHY_KEY_CODA, transform=ax.transAxes,
            va="top", fontsize=body_size, color="#333", linespacing=1.45)


def figure_c1_murphy_components(
    *, murphy: dict, dimensions,
) -> plt.Figure:
    """Murphy decomposition (CORP), mean across models: what drives Brier errors.

    One subplot per binary dimension. Two component groups on the x-axis --
    reliability (miscalibration, lower better) and resolution (discrimination,
    higher better) -- each with the B1-style 4 bars ``[self_nt, cross_nt,
    self_t, cross_t]``. The uncertainty floor (identical for self and cross by
    construction) is drawn as one dashed line per condition. The Brier totals
    (= REL - RES + UNC, exact under CORP) are annotated per subplot.
    """
    peer_lbl = peer_phrase(murphy.get("cross_model"))
    peer_abbr = peer_short(murphy.get("cross_model"))
    dim_names = _binary_dim_names(dimensions)
    n_dims = len(dim_names)
    ncols = min(3, n_dims) or 1
    nrows = math.ceil(n_dims / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.9 * nrows))
    flat = np.atleast_1d(axes).flatten()
    for ax in flat:
        ax.set_visible(False)

    # x layout: [self_nt, cross_nt, self_t, cross_t] within each component group.
    offsets = [0.0, 0.7, 1.8, 2.5]
    group_x = {"reliability": 0.0, "resolution": 4.5}

    for k, name in enumerate(dim_names):
        ax = flat[k]
        ax.set_visible(True)
        block = murphy["conditions"]
        brier_note: list[str] = []
        unc_by_cond: dict[str, float] = {}
        for j, cond in enumerate(_CONDITIONS):
            mean = block.get(cond, {}).get(name, {}).get("mean_across_models", {})
            s = mean.get("self", {})
            c = mean.get("cross", {})
            unc_by_cond[cond] = s.get("uncertainty", float("nan"))
            brier_note.append(
                f"{cond[:2]}: BS self {s.get('brier', float('nan')):.3f} / "
                f"{peer_abbr} {c.get('brier', float('nan')):.3f}"
            )
            for comp, gx in group_x.items():
                xs = [gx + offsets[2 * j], gx + offsets[2 * j + 1]]
                ys = [s.get(comp, float("nan")), c.get(comp, float("nan"))]
                ax.bar(xs, ys, width=0.6,
                       color=[SELF_COLORS[cond], CROSS_COLORS[cond]],
                       edgecolor="white")
                for x, val in zip(xs, ys):
                    if not np.isnan(val):
                        # bbox + zorder keep the label legible when a bar top
                        # sits just under a UNC floor line (which would
                        # otherwise strike through the digits).
                        ax.text(x, val, f"{val:.3f}", ha="center", va="bottom",
                                fontsize=6, zorder=5,
                                bbox=dict(boxstyle="square,pad=0.05",
                                          facecolor="white", edgecolor="none",
                                          alpha=0.8))

        # Uncertainty floor -- one dashed line per condition (self == cross).
        for cond, ls in zip(_CONDITIONS, ("--", ":")):
            v = unc_by_cond.get(cond, float("nan"))
            if not np.isnan(v):
                ax.axhline(v, color=_NEUTRAL_GREY, linestyle=ls, linewidth=1.1,
                           label=f"UNC ({cond})" if k == 0 else None)

        ax.set_xticks([group_x["reliability"] + 1.25, group_x["resolution"] + 1.25],
                      ["reliability\n(lower is better)", "resolution\n(higher is better)"],
                      fontsize=8.5)
        ax.set_title(f"{name}\n{';  '.join(brier_note)}", fontsize=8)
        ax.set_ylabel("Brier component", fontsize=8)
        ax.margins(y=0.25)

    # The dimension grid rarely fills its last row; spend the spare slot on the
    # key rather than leaving a hole. With no spare slot the figure is unchanged.
    if len(flat) > n_dims:
        _draw_murphy_key(flat[n_dims])

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=SELF_COLORS["no_thinking"], label="self (no_thinking)"),
        plt.Rectangle((0, 0), 1, 1, color=CROSS_COLORS["no_thinking"],
                      label=f"{peer_lbl} (no_thinking)"),
        plt.Rectangle((0, 0), 1, 1, color=SELF_COLORS["thinking"], label="self (thinking)"),
        plt.Rectangle((0, 0), 1, 1, color=CROSS_COLORS["thinking"],
                      label=f"{peer_lbl} (thinking)"),
        plt.Line2D([], [], color=_NEUTRAL_GREY, linestyle="--", label="UNC floor (no_thinking)"),
        plt.Line2D([], [], color=_NEUTRAL_GREY, linestyle=":", label="UNC floor (thinking)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=8, framealpha=0.95)
    fig.suptitle(
        "Murphy decomposition (CORP) — Brier = REL − RES + UNC, mean across models\n"
        "self loses by being miscalibrated (high REL) or by not discriminating (low RES vs the UNC floor)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.90])
    return fig


# --------------------------------------------------------------------------
# C2 -- reliability diagrams (pooled bins; binary dims)
# --------------------------------------------------------------------------
def figure_c2_reliability_diagrams(
    *, murphy: dict, dimensions,
) -> plt.Figure:
    """Calibration curves per binary dim: observed rate vs stated probability.

    Pooled events across models (per-model bins are too thin), self vs the peer x
    both conditions, from the fixed-bin table in ``murphy.json``. Marker area
    scales with the bin's share of total weight; the diagonal is perfect
    calibration. The grey horizontal is the base rate of the primary condition
    only (they differ slightly per condition; see the axhline comment below).
    Pooling caveat (mixed base rates; one shared anonymous forecast) is stated in
    the suptitle.

    The peer series is labelled ``"<predictor> forecasting anonymous"`` rather than
    "cross" -- see :func:`peer_phrase` for why the old label misled.
    """
    peer_lbl = peer_phrase(murphy.get("cross_model"))
    dim_names = _binary_dim_names(dimensions)
    n_dims = len(dim_names)
    ncols = min(3, n_dims) or 1
    nrows = math.ceil(n_dims / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows))
    flat = np.atleast_1d(axes).flatten()
    for ax in flat:
        ax.set_visible(False)

    colors = {"self": SELF_COLORS, "cross": CROSS_COLORS}
    for k, name in enumerate(dim_names):
        ax = flat[k]
        ax.set_visible(True)
        ax.plot([0, 1], [0, 1], color=_NEUTRAL_GREY, linestyle="--", linewidth=1.0,
                zorder=1)
        base_rates: dict[str, float] = {}
        for cond in _CONDITIONS:
            pooled = murphy["conditions"].get(cond, {}).get(name, {}).get("pooled", {})
            for kind in ("self", "cross"):
                comp = pooled.get(kind, {})
                bins = comp.get("binned", {}).get("bins", [])
                if not bins:
                    continue
                total_w = sum(b["weight"] for b in bins) or 1.0
                xs = [b["mean_forecast"] for b in bins]
                ys = [b["observed_rate"] for b in bins]
                sizes = [30 + 220 * (b["weight"] / total_w) for b in bins]
                style = "-" if cond == "no_thinking" else ":"
                ax.plot(xs, ys, style, color=colors[kind][cond], linewidth=1.4,
                        zorder=2)
                ax.scatter(xs, ys, s=sizes, color=colors[kind][cond],
                           edgecolor="white", linewidth=0.8, zorder=3)
            base = pooled.get("self", {}).get("base_rate", float("nan"))
            if not np.isnan(base):
                base_rates[cond] = base

        # One base-rate line, from the primary condition. The conditions differ
        # by ~0.01 (a cell is scored only where both forecasts parse, and the
        # surviving cell set differs), which drew two indistinguishable greys
        # for a legend entry that promises one.
        primary = next((c for c in _CONDITIONS if c in base_rates), None)
        if primary is not None:
            ax.axhline(base_rates[primary], color=_NEUTRAL_GREY, linewidth=0.8,
                       alpha=0.6, zorder=1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("stated probability", fontsize=8)
        ax.set_ylabel("observed frequency", fontsize=8)
        ax.set_title(name, fontsize=9)

    legend_handles = [
        plt.Line2D([], [], color=SELF_COLORS["no_thinking"], marker="o", label="self (no_thinking)"),
        plt.Line2D([], [], color=CROSS_COLORS["no_thinking"], marker="o",
                   label=f"{peer_lbl} (no_thinking)"),
        plt.Line2D([], [], color=SELF_COLORS["thinking"], marker="o", linestyle=":", label="self (thinking)"),
        plt.Line2D([], [], color=CROSS_COLORS["thinking"], marker="o", linestyle=":",
                   label=f"{peer_lbl} (thinking)"),
        plt.Line2D([], [], color=_NEUTRAL_GREY, linestyle="--", label="perfect calibration"),
        plt.Line2D([], [], color=_NEUTRAL_GREY, linewidth=0.8, alpha=0.6,
                   label=f"pooled base rate ({_CONDITIONS[0]})"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.04), fontsize=8, framealpha=0.95)
    fig.suptitle(
        "Reliability diagrams — pooled events across models\n"
        f"caveat: pooling mixes per-model base rates; {peer_short(murphy.get('cross_model'))}'s "
        "anonymous forecast is one set shared across all subjects",
        fontsize=11,
    )
    fig.text(
        0.5, 0.008,
        "Dot size shows how much evidence each bin carries: larger dots pooled more (model, task) events "
        "at that stated probability; smaller dots are thin bins where the calibration point is less trustworthy. "
        "The dashed diagonal is perfect calibration; the grey horizontal is the pooled base rate.",
        ha="center", va="bottom", fontsize=8.5, color="#444", linespacing=1.45,
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.88])
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
    murphy: dict | None = None,
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
    if murphy is not None:
        figures["c1_murphy_components"] = figure_c1_murphy_components(
            murphy=murphy, dimensions=dimensions,
        )
        figures["c2_reliability_diagrams"] = figure_c2_reliability_diagrams(
            murphy=murphy, dimensions=dimensions,
        )
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
    murphy: dict | None = None,
) -> list[Path]:
    """Render the figures for the cross-predictor under ``figures_dir``.

    The Core 5 always; the two Murphy figures (C1 components, C2 reliability
    diagrams) additionally when a ``murphy`` report dict is supplied. The
    cross-predictor lands in a ``primary__<slug>/`` subfolder. Returns the
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
            formats=formats, murphy=murphy,
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
    # The Murphy figures render only when murphy.py has produced its report.
    murphy_path = results_dir / "murphy.json"
    murphy = (json.loads(murphy_path.read_text(encoding="utf-8"))
              if murphy_path.exists() else None)
    written = render_all(scores, analysis, figures_dir, formats=formats, murphy=murphy)
    print(f"[plots] wrote {len(written)} figure file(s) under {figures_dir}")
    cross_folders = sorted({p.parent.name for p in written})
    for folder in cross_folders:
        print(f"  - {folder}/  ({sum(1 for p in written if p.parent.name == folder)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
