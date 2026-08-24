"""Experiment 1 figures (EVAL_SPEC_V1.md sec 15.6 + 15.7).

Seven figures cover every pre-registered result and the §15.7 native-unit
transparency layer:

  A1 -- mean self loss grouped bar chart              (RQ1 accuracy)
  A2 -- self loss vs empirical solve rate scatter     (RQ2 scaling)
  A3 -- per-model Δloss forest plot                   (RQ3 elicitation)
  B1 -- per-dim mean self loss bars                   (sec 15.7 transparency)
  B2 -- per-(model, dim) self loss heatmap            (sec 15.7 transparency)

Two further downstream figures render when ``results/murphy.json`` exists
(produced by ``murphy.py`` -- the Murphy decomposition of the binary dims):

  C1 -- Murphy components (REL / RES vs the UNC floor), mean across models
  C2 -- pooled reliability diagrams (calibration curves) per binary dim

Every figure shows **both forecast conditions together** (`no_thinking` and
`thinking`); the user reads the thinking effect off the same plot as the
existence claim. A3 is unchanged (Δloss already encodes both).

MVP scope: the inferential statistics tier is pruned, so the figures render
**point estimates only** -- no confidence-interval bands, p-value annotations,
or fitted scaling curves.

Read-only: every figure is built from ``results/scores.json`` and
``results/analysis.json`` -- no scoring logic, no ``ScoringData`` reconstruction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path

import matplotlib

# Headless-safe backend before any pyplot import; the test suite and CI both
# run without a display.
matplotlib.use("Agg")

import matplotlib.patheffects as path_effects  # noqa: E402
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
import metrics  # noqa: E402

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="paper")
_FAMILY_PALETTE = "colorblind"
_NEUTRAL_GREY = "#6b6b6b"

#: Condition colors -- distinct enough for grouped bars / scatter markers /
#: the per-condition heatmap titles.
SELF_COLORS = {
    "no_thinking": "#3b6fb6",   # solid blue
    "thinking":    "#7aa3d4",   # lighter blue
}

#: Pretty axis labels per dimension -- native units the §15.7 transparency
#: layer reads literally as ("off by X tokens", "by Y steps", etc.).
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


def _sort_by_solve_rate(
    models: list[str], solve_rates: dict[str, float], descending: bool = True
) -> list[str]:
    return sorted(models, key=lambda m: solve_rates.get(m, 0.0), reverse=descending)


def _model_mean_loss(scores: dict, condition: str, model: str) -> float:
    """Equal-dimension headline self loss for one model in one condition."""
    per_model = scores["scores"][condition].get(model, {})
    if "mean_loss_equal_dim" in per_model:
        return per_model["mean_loss_equal_dim"]
    dim_names = list(config.DIMENSION_NAMES)
    mlpds = {
        m: s["mean_loss_per_dim"]
        for m, s in scores["scores"][condition].items()
        if "mean_loss_per_dim" in s
    }
    if model not in mlpds:
        return float("nan")
    scales = metrics.dimension_loss_scales(mlpds, dim_names)
    return metrics.equal_dimension_mean_loss(mlpds[model], scales, dim_names)


# --------------------------------------------------------------------------
# Collision-aware point labels
# --------------------------------------------------------------------------
# Labels are searched over a ring of candidate offsets and scored, rather than
# nudged by hand per model: a re-run with different models or solve rates has to
# stay legible without anyone re-tuning a table of offsets.
_LABEL_ANGLES_DEG = tuple(range(0, 360, 15))
_LABEL_RADII_PT = (11.0, 16.0, 22.0, 30.0, 40.0, 52.0, 66.0, 82.0)
_LABEL_COLOR = "#333333"
#: Breathing room around each label when scoring, so neighbours never touch.
_LABEL_PAD_PX = 2.0
_LABEL_LEADER = dict(arrowstyle="-", color="#a0a0a0", lw=0.7, shrinkA=1.0, shrinkB=4.0)
#: White outline so a label stays readable where it crosses a gridline.
_LABEL_HALO = (path_effects.withStroke(linewidth=2.6, foreground="white"),)

# Scoring weights. Overlap terms are px^2, distance/crowding terms px, chosen so
# any real overlap outweighs the pull toward the marker -- a long leader line to
# clear space always beats two labels sitting on top of each other.
_W_LABEL_OVERLAP = 4.0
_W_MARKER_OVERLAP = 2.5
_W_OBSTACLE_OVERLAP = 2.5
# Flat surcharge for touching anything at all. Overlapping text is a hard
# failure rather than a graded cost, and a pure area penalty under-punishes the
# clipped-by-a-hair case that still reads as broken.
_W_ANY_OVERLAP = 400.0
_W_OUTSIDE_AXES = 4.0   # spilling past the frame is worse than a long leader
_W_DISTANCE = 4.0
_W_CROWDING = 70.0
_CROWDING_RADIUS_PX = 85.0
# A long name parked alongside its dot fences off a whole horizontal band, so
# wide labels are pushed into the vertical space above or below the point.
_W_WIDE_SIDEWAYS = 1.6
_WIDE_LABEL_PX = 85.0
# Reward for landing in open space. Without it, "no overlap" is the whole test
# and a label happily wedges into the narrow gap between two dots rather than
# spending a few more pixels of leader line to reach a wide-open band. Weighted
# above _W_DISTANCE on purpose: for a long name in a cluster, reaching clear
# space matters more than hugging the dot.
_W_CLEARANCE = 5.0
_CLEARANCE_CAP_PX = 50.0
# Tie-break toward the space above a point: with a scatter this sparse several
# offsets are equally collision-free, and above-and-slightly-right is the
# convention readers expect.
_W_UPWARD = 35.0
#: Below this marker-to-text gap the label reads as attached; no leader drawn.
_LEADER_MIN_GAP_PX = 8.0


def _box_overlap(a: tuple, b: tuple) -> float:
    """Intersection area of two ``(x0, y0, x1, y1)`` boxes, 0 if disjoint."""
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    return dx * dy if dx > 0 and dy > 0 else 0.0


def _area_outside(box: tuple, frame: tuple) -> float:
    """Area of ``box`` falling outside ``frame`` (both ``(x0, y0, x1, y1)``)."""
    area = (box[2] - box[0]) * (box[3] - box[1])
    return max(0.0, area - _box_overlap(box, frame))


def _gap_to_point(box: tuple, px: float, py: float) -> float:
    """Shortest distance from ``(px, py)`` to the edge of ``box`` (0 if inside)."""
    dx = max(box[0] - px, 0.0, px - box[2])
    dy = max(box[1] - py, 0.0, py - box[3])
    return math.hypot(dx, dy)


def _box_gap(a: tuple, b: tuple) -> float:
    """Shortest distance between two boxes (0 if they touch or overlap)."""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def _text_px_size(ax: plt.Axes, label: str, fontsize: float) -> tuple[float, float]:
    """Rendered ``(width, height)`` of ``label`` in pixels, measured off-canvas."""
    probe = ax.text(0, 0, label, fontsize=fontsize, transform=None)
    extent = probe.get_window_extent(ax.figure.canvas.get_renderer())
    probe.remove()
    return extent.width, extent.height


def _alignment_for(dx: float, dy: float) -> tuple[str, str]:
    """Alignment that grows the text away from the marker it belongs to."""
    ha = "left" if dx > 0.3 else "right" if dx < -0.3 else "center"
    va = "bottom" if dy > 0.3 else "top" if dy < -0.3 else "center"
    return ha, va


def _offset_box(
    anchor: tuple[float, float], offset: tuple[float, float],
    size: tuple[float, float], ha: str, va: str,
) -> tuple[float, float, float, float]:
    """Pixel box a label occupies at ``offset`` from ``anchor`` under ``ha/va``."""
    w, h = size
    x = anchor[0] + offset[0]
    y = anchor[1] + offset[1]
    x0 = x if ha == "left" else x - w if ha == "right" else x - w / 2
    y0 = y if va == "bottom" else y - h if va == "top" else y - h / 2
    return (x0, y0, x0 + w, y0 + h)


#: Candidate offsets in points, tried at every radius on every ring.
_LABEL_CANDIDATES = tuple(
    (r * math.cos(math.radians(a)), r * math.sin(math.radians(a)))
    for r in _LABEL_RADII_PT for a in _LABEL_ANGLES_DEG
)


def _rank_label_offsets(
    *,
    size: tuple[float, float],
    mid: tuple,
    up: tuple,
    down: tuple,
    frame: tuple,
    marker_boxes: Sequence[tuple],
    own_idx: frozenset,
    neighbours: Sequence[tuple[float, float]],
    placed: Sequence[tuple],
    obstacles: Sequence[tuple],
    px_per_pt: float,
) -> list[tuple]:
    """Score every candidate offset for one label, cheapest first.

    ``mid``, ``up`` and ``down`` are the ``(pixel, data)`` anchors a label can
    hang from -- the midpoint of the model's dots, its top dot, its bottom dot.
    Returns ``(cost, box, offset_pt, ha, va, base_px, base_data)`` tuples. The
    cost sums, in comparable px / px^2 units: distance from the dot, a penalty
    for parking a wide name alongside its dot where it fences off a horizontal
    band, a tie-break toward the space above, area spilling past the axes,
    overlap with markers / placed labels / obstacles (plus a flat surcharge for
    any overlap at all), minus a reward for landing in the open, plus a push
    away from neighbouring points.
    """
    w, _ = size
    sideways = _W_WIDE_SIDEWAYS * max(0.0, w - 2 * _LABEL_PAD_PX - _WIDE_LABEL_PX)
    scored: list[tuple] = []
    for dx_pt, dy_pt in _LABEL_CANDIDATES:
        offset = (dx_pt * px_per_pt, dy_pt * px_per_pt)
        dist = math.hypot(*offset)
        ux, uy = offset[0] / dist, offset[1] / dist
        ha, va = _alignment_for(ux, uy)
        # A label going up starts from the top dot of the pair, not from the
        # midpoint -- otherwise every vertical placement pays for crossing the
        # model's own stem and the search never looks at the space above.
        base_px, base_data = up if uy > 0.3 else down if uy < -0.3 else mid
        box = _offset_box(base_px, offset, size, ha, va)

        cost = _W_DISTANCE * dist
        if va == "center":
            cost += sideways   # only a truly alongside label fences off a band
        cost += _W_UPWARD * (1.0 - uy) / 2.0
        cost += _W_OUTSIDE_AXES * _area_outside(box, frame)
        # Clearance ignores the label's own dots: being near them is the point,
        # and counting them would just push every label outward.
        clearance = _CLEARANCE_CAP_PX
        collides = False
        for other in placed:
            hit = _box_overlap(box, other)
            cost += _W_LABEL_OVERLAP * hit
            collides = collides or hit > 0
            clearance = min(clearance, _box_gap(box, other))
        for i, mbox in enumerate(marker_boxes):
            hit = _box_overlap(box, mbox)
            cost += _W_MARKER_OVERLAP * hit
            collides = collides or hit > 0
            if i not in own_idx:
                clearance = min(clearance, _box_gap(box, mbox))
        for obox in obstacles:
            hit = _box_overlap(box, obox)
            cost += _W_OBSTACLE_OVERLAP * hit
            collides = collides or hit > 0
            clearance = min(clearance, _box_gap(box, obox))
        if collides:
            cost += _W_ANY_OVERLAP
        cost -= _W_CLEARANCE * clearance
        for mx, my in neighbours:
            vx, vy = mx - base_px[0], my - base_px[1]
            norm = math.hypot(vx, vy)
            toward = (ux * vx + uy * vy) / norm
            if toward > 0:
                cost += _W_CROWDING * toward * (1.0 - norm / _CROWDING_RADIUS_PX)

        scored.append((cost, box, (dx_pt, dy_pt), ha, va, base_px, base_data))
    scored.sort(key=lambda s: s[0])
    return scored


def _place_point_labels(
    ax: plt.Axes,
    labels: Sequence[tuple[float, float, str, Sequence[tuple[float, float]]]],
    *,
    avoid: Sequence[tuple[float, float]] = (),
    obstacles: Sequence[tuple] = (),
    fontsize: float = 8.0,
    marker_radius_px: float = 9.0,
) -> None:
    """Annotate scatter points with labels that do not collide with anything.

    ``labels`` are ``(x, y, text, own)`` in data coordinates, where ``own`` lists
    the marker centres the label describes -- its own dots are the one thing it
    is allowed to sit close to. ``avoid`` lists every marker centre in the
    scatter so a label never lands on a point it does not describe;
    ``obstacles`` are extra pixel boxes to dodge, e.g. an in-axes note.

    Placement is greedy: the hardest label picks first, taking the cheapest
    offset per ``_rank_label_offsets``, and each choice becomes an obstacle for
    the rest. A thin leader line is drawn only when the label ends up far enough
    from its dot to be ambiguous.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72.0

    frame = tuple(ax.get_window_extent(renderer).extents)
    marker_px = [tuple(ax.transData.transform((x, y))) for x, y in avoid]
    marker_boxes = [
        (mx - marker_radius_px, my - marker_radius_px,
         mx + marker_radius_px, my + marker_radius_px)
        for mx, my in marker_px
    ]

    entries = []
    for x, y, text, own in labels:
        anchor = tuple(ax.transData.transform((x, y)))
        own_set = {tuple(p) for p in own}
        own_idx = frozenset(i for i, p in enumerate(avoid) if tuple(p) in own_set)
        crowding = sum(
            1 for i, (mx, my) in enumerate(marker_px)
            if i not in own_idx
            and math.hypot(mx - anchor[0], my - anchor[1]) < _CROWDING_RADIUS_PX
        )
        width = _text_px_size(ax, text, fontsize)[0]
        # Hardest first: whoever picks last gets the leftovers, and difficulty is
        # both how many neighbours crowd the point and how much room the name
        # needs -- a long name in a cluster has the fewest places left to go.
        difficulty = crowding + width / _WIDE_LABEL_PX
        entries.append((difficulty, (x, y), anchor, text, own_idx))
    entries.sort(key=lambda e: -e[0])

    placed: list[tuple] = []
    for _, data_xy, anchor, text, own_idx in entries:
        w, h = _text_px_size(ax, text, fontsize)
        neighbours = [
            (mx, my) for i, (mx, my) in enumerate(marker_px)
            if i not in own_idx
            and math.hypot(mx - anchor[0], my - anchor[1]) < _CROWDING_RADIUS_PX
        ]
        mid = (anchor, data_xy)
        ranked = sorted(own_idx, key=lambda i: marker_px[i][1])
        up = (marker_px[ranked[-1]], avoid[ranked[-1]]) if ranked else mid
        down = (marker_px[ranked[0]], avoid[ranked[0]]) if ranked else mid

        best = _rank_label_offsets(
            size=(w + 2 * _LABEL_PAD_PX, h + 2 * _LABEL_PAD_PX),
            mid=mid, up=up, down=down, frame=frame,
            marker_boxes=marker_boxes, own_idx=own_idx, neighbours=neighbours,
            placed=placed, obstacles=obstacles, px_per_pt=px_per_pt,
        )[0]
        _, box, offset_pt, ha, va, base_px, base_data = best

        needs_leader = _gap_to_point(box, *base_px) > _LEADER_MIN_GAP_PX
        ax.annotate(
            text, base_data, xytext=offset_pt, textcoords="offset points",
            ha=ha, va=va, fontsize=fontsize, color=_LABEL_COLOR,
            path_effects=list(_LABEL_HALO), zorder=6,
            arrowprops=dict(_LABEL_LEADER) if needs_leader else None,
        )
        placed.append(box)


def _padded_limits(values: Sequence[float], fraction: float) -> tuple[float, float]:
    """``(lo, hi)`` covering ``values`` with breathing room for labels."""
    lo, hi = min(values), max(values)
    span = hi - lo
    pad = fraction * span if span > 0 else max(0.05, abs(hi) * 0.1)
    return lo - pad, hi + pad


# --------------------------------------------------------------------------
# A1 -- grouped mean self loss bar chart (both conditions)
# --------------------------------------------------------------------------
def figure_a1_loss_bar(
    *, scores: dict, analysis: dict, families_by_model: dict[str, str],
) -> plt.Figure:
    """RQ1 visualisation: per-model mean self loss for both conditions side-by-side.

    Two horizontal bars per model -- one per condition -- with the pooled
    ``no_thinking`` mean shown as a vertical line (the canonical RQ1 reading).
    Lower loss is better.
    """
    rq1 = analysis["rq1_accuracy"]
    per_model_nt = rq1["per_model_loss"]
    models = sorted(per_model_nt.keys(), key=lambda m: per_model_nt[m])
    loss_nt = per_model_nt
    loss_th = {m: _model_mean_loss(scores, "thinking", m) for m in models}

    fig, ax = plt.subplots(figsize=(8.0, 0.55 * len(models) + 1.8))
    y = np.arange(len(models))
    h = 0.38
    ax.barh(y - h / 2, [loss_nt.get(m, np.nan) for m in models], height=h,
            color=SELF_COLORS["no_thinking"], edgecolor="white",
            label="no_thinking")
    ax.barh(y + h / 2, [loss_th.get(m, np.nan) for m in models], height=h,
            color=SELF_COLORS["thinking"], edgecolor="white",
            label="thinking")
    ax.set_yticks(y, [_short_label(m) for m in models])

    # Pooled (no_thinking) mean line -- the canonical RQ1 point estimate.
    ax.axvline(rq1["point_estimate"],
               color=SELF_COLORS["no_thinking"], linewidth=1.4,
               label=f"no_thinking mean = {rq1['point_estimate']:.3f}")

    ax.set_xlabel("mean self loss(M, condition)")
    ax.set_title("RQ1 — self-forecast accuracy (lower is better)")
    ax.legend(loc="lower right", framealpha=0.95, fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# A2 -- scaling scatter (loss ~ solve rate) for both conditions
# --------------------------------------------------------------------------
def figure_a2_scaling(
    *, scores: dict, analysis: dict, families_by_model: dict[str, str],
) -> plt.Figure:
    """RQ2 visualisation: equal-dimension mean self loss vs empirical solve rate.

    Each dimension's native-unit loss is divided by the cross-model peak for
    that dimension (within the condition) before averaging, so token/step MAE
    does not dominate the pooled y-axis. One scatter point per (model,
    condition) -- filled = ``no_thinking``, hollow = ``thinking`` -- joined by a
    faint stem so the pair reads as one model. Model names are placed by
    ``_place_point_labels``.
    """
    solve_rates = analysis.get("solve_rates") or analysis["rq2_scaling"].get("solve_rates")
    models = sorted(
        set(scores["scores"]["no_thinking"]) & set(scores["scores"]["thinking"]),
        key=lambda m: solve_rates.get(m, 0.0),
        reverse=True,
    )
    loss_nt = {m: _model_mean_loss(scores, "no_thinking", m) for m in models}
    loss_th = {m: _model_mean_loss(scores, "thinking", m) for m in models}
    families_present = sorted({families_by_model[m] for m in models if m in families_by_model})
    color_map = _family_color_map(families_present)

    fig, ax = plt.subplots(figsize=(8.8, 5.6))

    marker_points: list[tuple[float, float]] = []
    label_points: list[tuple[float, float, str, list[tuple[float, float]]]] = []

    for m in models:
        fam_color = color_map.get(families_by_model.get(m, ""), _NEUTRAL_GREY)
        x = solve_rates.get(m, np.nan)
        y_nt = loss_nt.get(m, np.nan)
        y_th = loss_th.get(m, np.nan)
        if np.isnan(x):
            continue
        # The stem ties a model's two conditions together; without it the two
        # near-identical rings read as two unrelated models.
        if not (np.isnan(y_nt) or np.isnan(y_th)):
            ax.plot([x, x], [y_nt, y_th], color=fam_color, linewidth=1.3,
                    alpha=0.4, solid_capstyle="round", zorder=2)
        # filled = no_thinking, hollow ring = thinking
        ax.scatter(x, y_nt,
                   facecolor=fam_color, edgecolor="white", s=90,
                   linewidth=1.2, zorder=3)
        ax.scatter(x, y_th,
                   facecolor="white", edgecolor=fam_color, s=90,
                   linewidth=2.0, zorder=3)
        own = [(x, y) for y in (y_nt, y_th) if not np.isnan(y)]
        marker_points += own
        y_mid = float(np.nanmean([y_nt, y_th]))
        if not np.isnan(y_mid):
            label_points.append((x, y_mid, _short_label(m), own))

    x_vals = [x for x, _ in marker_points]
    y_vals = [y for _, y in marker_points]
    if x_vals:
        ax.set_xlim(*_padded_limits(x_vals, 0.12))
    if y_vals:
        ax.set_ylim(*_padded_limits(y_vals, 0.11))

    ax.set_xlabel("empirical solve rate (mean p_solve)")
    ax.set_ylabel("equal-dimension mean loss (lower = better)")
    ax.set_title("RQ2 — self-forecast accuracy vs model capability (descriptive)",
                 fontsize=11.5, pad=12)
    for spine in ax.spines.values():
        spine.set_edgecolor("#d0d0d0")

    # Two legends: what the colour means, then what the fill means. Merging them
    # into one block puts families and conditions in the same column and reads
    # as though "thinking" were a vendor.
    family_handles = [
        plt.Line2D([], [], color=color_map[f], marker="o", linestyle="",
                   markersize=8, markeredgecolor="white", label=f)
        for f in families_present
    ]
    cond_handles = [
        plt.Line2D([], [], color=_NEUTRAL_GREY, marker="o", linestyle="",
                   markersize=8, markeredgecolor="white", label="no_thinking"),
        plt.Line2D([], [], color=_NEUTRAL_GREY, marker="o", linestyle="",
                   markerfacecolor="white", markeredgewidth=2, markersize=8,
                   label="thinking"),
    ]
    # Legends below the axes so high-solve-rate points (lower-right) stay visible.
    fam_legend = ax.legend(
        handles=family_handles, loc="upper left", bbox_to_anchor=(0.0, -0.12),
        ncol=len(family_handles) or 1, title="model family",
        framealpha=0.0, fontsize=8, borderpad=0.4, handletextpad=0.4,
        columnspacing=1.4,
    )
    fam_legend.get_title().set_fontsize(8)
    fam_legend.get_title().set_color("#555")
    fam_legend.set_alignment("left")
    ax.add_artist(fam_legend)
    cond_legend = ax.legend(
        handles=cond_handles, loc="upper right", bbox_to_anchor=(1.0, -0.12),
        ncol=2, title="forecast condition",
        framealpha=0.0, fontsize=8, borderpad=0.4, handletextpad=0.4,
        columnspacing=1.4,
    )
    cond_legend.get_title().set_fontsize(8)
    cond_legend.get_title().set_color("#555")
    cond_legend.set_alignment("left")

    # Labels are placed last: the search works in pixels, so the axes must have
    # its final size before any offset is scored.
    fig.tight_layout(rect=[0, 0.10, 1, 1])

    _place_point_labels(ax, label_points, avoid=marker_points,
                        obstacles=[], fontsize=8.0)
    return fig


# --------------------------------------------------------------------------
# A3 -- Δloss forest plot (Δ already encodes both conditions)
# --------------------------------------------------------------------------
def figure_a3_delta_loss_forest(
    *, scores: dict, analysis: dict, families_by_model: dict[str, str],
) -> plt.Figure:
    """RQ3 visualisation: per-model Δloss(M) = loss(thinking) − loss(no_thinking).

    Per-model dots, family-coloured; pooled mean shown as a diamond;
    reference line at Δloss = 0. Negative Δloss means thinking helped.
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

    ax.set_xlabel("Δloss(M)  =  loss(thinking) − loss(no_thinking)")
    ax.set_title("RQ3 — does thinking elicit better self-prediction?")
    family_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_map[f], label=f) for f in families_present
    ]
    ax.legend(handles=family_handles + ax.get_legend_handles_labels()[0],
              loc="lower right", framealpha=0.95, fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# B1 -- grouped per-dim mean self loss (both conditions, native units)
# --------------------------------------------------------------------------
def figure_b1_per_dim_loss(
    *, scores: dict, analysis: dict, dimensions,
) -> plt.Figure:
    """§15.7 transparency: per-dim mean self loss, both conditions side-by-side.

    Each dimension gets its own subplot (its own y-axis, native units). Inside
    each subplot: 2 bars ``[no_thinking, thinking]``. Descriptive means only.
    """
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
        x_positions = [0.0, 1.0]
        deltas: dict[str, float] = {}
        for j, cond in enumerate(_CONDITIONS):
            cond_models = list(scores["scores"][cond].keys())
            vals = np.array(
                [scores["scores"][cond][m]["mean_loss_per_dim"].get(name, np.nan)
                 for m in cond_models], dtype=float,
            )
            mean_val = (float(np.nanmean(vals))
                        if np.any(~np.isnan(vals)) else float("nan"))
            deltas[cond] = mean_val
            bars.append((x_positions[j], mean_val, SELF_COLORS[cond], cond))

        xs = [b[0] for b in bars]
        ys = [b[1] for b in bars]
        colors = [b[2] for b in bars]
        ax.bar(xs, ys, width=0.6, color=colors, edgecolor="white")
        for x, val in zip(xs, ys):
            if not np.isnan(val):
                ax.text(x, val, f"{val:.3f}" if val < 10 else f"{val:.1f}",
                        ha="center", va="bottom", fontsize=7)
        ax.set_xticks(xs, ["no_thinking", "thinking"], fontsize=8)

        units = DIM_UNITS.get(name, "")
        delta = deltas["thinking"] - deltas["no_thinking"]
        ax.set_title(
            f"{name}\nnt = {deltas['no_thinking']:.3f}, "
            f"t = {deltas['thinking']:.3f}, "
            f"Δ = {delta:+.3f} ({units})",
            fontsize=8,
        )
        ax.set_ylabel(f"mean loss ({units})", fontsize=8)
        ax.margins(y=0.20)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=SELF_COLORS["no_thinking"],
                      label="no_thinking"),
        plt.Rectangle((0, 0), 1, 1, color=SELF_COLORS["thinking"],
                      label="thinking"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), fontsize=9, framealpha=0.95)
    fig.suptitle(
        "§15.7 transparency — per-dimension mean self loss in native units\n"
        "darker = no_thinking, lighter = thinking; lower is better",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    return fig


# --------------------------------------------------------------------------
# B2 -- side-by-side per-(model, dim) self loss heatmaps (both conditions)
# --------------------------------------------------------------------------
# Token/step MAE lives on a 10–1000+ scale; keeping it on the same color scale
# as Brier/MAE-in-[0,1] dims washes out every other column.
_B2_HEATMAP_EXCLUDE = frozenset({"avg_tokens_step", "total_expected_steps"})


def _b2_heatmap_dimensions(dimensions) -> list:
    """Dimensions shown in B2 -- omits high-scale count metrics."""
    out = []
    for d in dimensions:
        name = d.name if hasattr(d, "name") else d[0]
        if name not in _B2_HEATMAP_EXCLUDE:
            out.append(d)
    return out


def figure_b2_loss_heatmap(
    *, scores: dict, analysis: dict, dimensions,
) -> plt.Figure:
    """§15.7 transparency: per-(model, dim) self loss, one heatmap per condition.

    Rows are models sorted by empirical solve rate (strongest at top); columns
    are dimensions in spec order (excluding ``avg_tokens_step`` and
    ``total_expected_steps``, which share the color scale poorly). YlOrRd
    colormap -- lighter = lower loss (better); no diverging centre.
    """
    solve_rates = analysis["solve_rates"]
    models = _sort_by_solve_rate(
        list(scores["scores"]["no_thinking"].keys()), solve_rates, descending=True,
    )
    heatmap_dims = _b2_heatmap_dimensions(dimensions)
    dim_names = [d.name if hasattr(d, "name") else d[0] for d in heatmap_dims]

    fig, axes = plt.subplots(
        1, 2, figsize=(2.0 * len(dim_names) + 2.5, 0.55 * len(models) + 2.6),
        sharey=True,
    )
    last_im = None
    all_vals: list[float] = []
    for cond in _CONDITIONS:
        per_model = scores["scores"][cond]
        for m in models:
            mlpd = per_model.get(m, {}).get("mean_loss_per_dim", {})
            for name in dim_names:
                v = mlpd.get(name, float("nan"))
                if not np.isnan(v):
                    all_vals.append(v)
    vmin = min(all_vals) if all_vals else 0.0
    vmax = max(all_vals) if all_vals else 1.0

    for ax, cond in zip(axes, _CONDITIONS):
        per_model = scores["scores"][cond]
        matrix = np.full((len(models), len(dim_names)), np.nan)
        for i, m in enumerate(models):
            mlpd = per_model.get(m, {}).get("mean_loss_per_dim", {})
            for j, name in enumerate(dim_names):
                if name in mlpd:
                    matrix[i, j] = mlpd[name]
        im = ax.imshow(matrix, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="auto")
        last_im = im
        ax.set_xticks(range(len(dim_names)), dim_names, rotation=40, ha="right")
        ax.set_title(cond, fontsize=10)
        for i in range(len(models)):
            for j in range(len(dim_names)):
                v = matrix[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if v > (vmin + vmax) / 2 else "black")
    axes[0].set_yticks(range(len(models)), [_short_label(m) for m in models])
    fig.colorbar(last_im, ax=axes, shrink=0.85,
                 label="mean self loss (lower is better)")
    fig.suptitle(
        "§15.7 transparency — per-(model, dim) self loss  "
        "(rows sorted by empirical solve rate, top = strongest)",
        fontsize=11, y=0.99,
    )
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
     "forecaster can beat, here the agent's own rollout-to-rollout\n"
     "randomness. Same for both conditions, drawn as the dashed line."),
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
    higher better) -- each with 2 bars ``[no_thinking, thinking]``. The
    uncertainty floor is drawn as one dashed line per condition. The Brier totals
    (= REL - RES + UNC, exact under CORP) are annotated per subplot.
    """
    dim_names = _binary_dim_names(dimensions)
    n_dims = len(dim_names)
    ncols = min(3, n_dims) or 1
    nrows = math.ceil(n_dims / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.9 * nrows))
    flat = np.atleast_1d(axes).flatten()
    for ax in flat:
        ax.set_visible(False)

    # x layout: [no_thinking, thinking] within each component group.
    offsets = [0.0, 0.7]
    group_x = {"reliability": 0.0, "resolution": 2.5}

    for k, name in enumerate(dim_names):
        ax = flat[k]
        ax.set_visible(True)
        block = murphy["conditions"]
        brier_note: list[str] = []
        unc_by_cond: dict[str, float] = {}
        for j, cond in enumerate(_CONDITIONS):
            mean = block.get(cond, {}).get(name, {}).get("mean_across_models", {})
            unc_by_cond[cond] = mean.get("uncertainty", float("nan"))
            brier_note.append(
                f"{cond[:2]}: BS {mean.get('brier', float('nan')):.3f}"
            )
            for comp, gx in group_x.items():
                x = gx + offsets[j]
                val = mean.get(comp, float("nan"))
                ax.bar(x, val, width=0.6, color=SELF_COLORS[cond], edgecolor="white")
                if not np.isnan(val):
                    ax.text(x, val, f"{val:.3f}", ha="center", va="bottom",
                            fontsize=6, zorder=5,
                            bbox=dict(boxstyle="square,pad=0.05",
                                      facecolor="white", edgecolor="none",
                                      alpha=0.8))

        # Uncertainty floor -- one dashed line per condition.
        for cond, ls in zip(_CONDITIONS, ("--", ":")):
            v = unc_by_cond.get(cond, float("nan"))
            if not np.isnan(v):
                ax.axhline(v, color=_NEUTRAL_GREY, linestyle=ls, linewidth=1.1,
                           label=f"UNC ({cond})" if k == 0 else None)

        ax.set_xticks([group_x["reliability"] + 0.35, group_x["resolution"] + 0.35],
                      ["reliability\n(lower is better)", "resolution\n(higher is better)"],
                      fontsize=8.5)
        ax.set_title(f"{name}\n{';  '.join(brier_note)}", fontsize=8)
        ax.set_ylabel("Brier component", fontsize=8)
        ax.margins(y=0.25)

    if len(flat) > n_dims:
        _draw_murphy_key(flat[n_dims])

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=SELF_COLORS["no_thinking"],
                      label="no_thinking"),
        plt.Rectangle((0, 0), 1, 1, color=SELF_COLORS["thinking"],
                      label="thinking"),
        plt.Line2D([], [], color=_NEUTRAL_GREY, linestyle="--",
                   label="UNC floor (no_thinking)"),
        plt.Line2D([], [], color=_NEUTRAL_GREY, linestyle=":",
                   label="UNC floor (thinking)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), fontsize=8, framealpha=0.95)
    fig.suptitle(
        "Murphy decomposition (CORP) — Brier = REL − RES + UNC, mean across models\n"
        "high REL = miscalibrated; low RES vs the UNC floor = uninformative",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.90])
    return fig


# --------------------------------------------------------------------------
# C2 -- reliability diagrams (pooled bins; binary dims)
# --------------------------------------------------------------------------
_RELIABILITY_DIM_CAPTIONS: dict[str, str] = {
    "p_solve": "P(final patch passes all tests)",
    "p_first_attempt_correct": "P(solve without backtracking after first test run)",
    "documentation": "P(patch adds type hints, comments, or docstrings)",
    "defensive_code": "P(patch adds guards, validation, or error handling beyond the minimal fix)",
    "express_uncertainty": "P(reasoning voices doubt, hedges, or seeks clarification)",
}

_RELIABILITY_LAYOUTS = ("grid_3col", "2x2+1", "1x5")

# Vertical gap between 1x5 panels, as a fraction of panel height. Must stay
# large enough to clear each panel's xlabel plus the next panel's two-line title.
_RELIABILITY_1X5_HSPACE = 0.25


def _reliability_panel_title(name: str) -> str:
    caption = _RELIABILITY_DIM_CAPTIONS.get(name)
    if caption:
        return f"{name}\n{caption}"
    return name


def _draw_reliability_panel(
    ax: plt.Axes, name: str, murphy: dict, *, compact: bool = False,
    show_xlabel: bool = True,
) -> None:
    """Draw one reliability-diagram panel on ``ax``."""
    title_fs = 7.5 if compact else 8.5
    label_fs = 7 if compact else 8
    dot_scale = (24, 170) if compact else (30, 220)
    ax.plot([0, 1], [0, 1], color=_NEUTRAL_GREY, linestyle="--", linewidth=1.0,
            zorder=1)
    base_rates: dict[str, float] = {}
    for cond in _CONDITIONS:
        pooled = murphy["conditions"].get(cond, {}).get(name, {}).get("pooled", {})
        bins = pooled.get("binned", {}).get("bins", [])
        if not bins:
            continue
        total_w = sum(b["weight"] for b in bins) or 1.0
        xs = [b["mean_forecast"] for b in bins]
        ys = [b["observed_rate"] for b in bins]
        sizes = [dot_scale[0] + dot_scale[1] * (b["weight"] / total_w) for b in bins]
        style = "-" if cond == "no_thinking" else ":"
        ax.plot(xs, ys, style, color=SELF_COLORS[cond], linewidth=1.4,
                zorder=2)
        ax.scatter(xs, ys, s=sizes, color=SELF_COLORS[cond],
                   edgecolor="white", linewidth=0.8, zorder=3)
        base = pooled.get("base_rate", float("nan"))
        if not np.isnan(base):
            base_rates[cond] = base

    primary = next((c for c in _CONDITIONS if c in base_rates), None)
    if primary is not None:
        ax.axhline(base_rates[primary], color=_NEUTRAL_GREY, linewidth=0.8,
                   alpha=0.6, zorder=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    if show_xlabel:
        ax.set_xlabel("stated probability", fontsize=label_fs)
    ax.set_ylabel("observed frequency", fontsize=label_fs)
    ax.set_title(_reliability_panel_title(name), fontsize=title_fs, linespacing=1.15)
    ax.tick_params(labelsize=label_fs - 0.5, labelbottom=show_xlabel)


def _reliability_figure_axes(
    n_dims: int, layout: str,
) -> tuple[plt.Figure, np.ndarray]:
    """Return ``(fig, axes)`` for a reliability-diagram layout."""
    if layout not in _RELIABILITY_LAYOUTS:
        raise ValueError(f"layout must be one of {_RELIABILITY_LAYOUTS}, got {layout!r}")

    if layout == "1x5":
        # Aspect matches a US-letter text block (6.5in x 8.6in) so the figure
        # fills a whole page at \textwidth without the class clamping height.
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(6.5, 8.6))
        gs = GridSpec(n_dims, 1, figure=fig, hspace=_RELIABILITY_1X5_HSPACE)
        axes = np.array([fig.add_subplot(gs[i, 0]) for i in range(n_dims)])
        return fig, axes

    if layout == "2x2+1":
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(8.8, 10.5))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1.05],
                      hspace=0.38, wspace=0.30)
        axes = np.array([
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[2, :]),
        ])
        return fig, axes

    ncols = min(3, n_dims) or 1
    nrows = math.ceil(n_dims / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows))
    return fig, np.atleast_1d(axes).flatten()


_FOOTNOTE_RELIABILITY = (
    "Forecasts are grouped into ten equal-width bins of stated probability "
    "(0.0-0.1, 0.1-0.2, ... 0.9-1.0). Each dot is one non-empty bin, drawn at the mean "
    "stated probability inside that bin rather than at the bin centre, so a gap in a "
    "curve means no forecast landed in that range. Dot size shows how much evidence "
    "each bin carries: larger dots pooled more (model, task) events at that stated "
    "probability; smaller dots are thin bins where the calibration point is less "
    "trustworthy. The dashed diagonal is perfect calibration; the grey horizontal is "
    "the pooled base rate under no_thinking only, not under thinking."
)


def _plot_column_bounds(fig: plt.Figure) -> tuple[float, float, float]:
    """Return ``(left, width, center)`` in figure coordinates for the data panels."""
    data_axes = [ax for ax in fig.axes if ax.get_visible() and ax.has_data()]
    if not data_axes:
        return 0.1, 0.8, 0.5
    fig.canvas.draw()
    boxes = [ax.get_position() for ax in data_axes]
    left = min(b.x0 for b in boxes)
    right = max(b.x1 for b in boxes)
    width = right - left
    return left, width, left + width / 2


def _add_wrapped_panel_caption(
    fig: plt.Figure, left: float, width: float, bottom: float, height: float,
    text: str, *, fontsize: float = 8.5,
) -> None:
    """Add wrapped caption text constrained to the plot-column width."""
    ax = fig.add_axes([left, bottom, width, height])
    ax.set_axis_off()
    ax.text(
        0.5, 0.5, text, transform=ax.transAxes,
        ha="center", va="center", fontsize=fontsize, color="#444",
        wrap=True, linespacing=1.35,
    )


def _decorate_reliability_figure(fig: plt.Figure, *, layout: str) -> None:
    legend_handles = [
        plt.Line2D([], [], color=SELF_COLORS["no_thinking"], marker="o",
                   label="no_thinking"),
        plt.Line2D([], [], color=SELF_COLORS["thinking"], marker="o", linestyle=":",
                   label="thinking"),
        plt.Line2D([], [], color=_NEUTRAL_GREY, linestyle="--",
                   label="perfect calibration"),
        plt.Line2D([], [], color=_NEUTRAL_GREY, linewidth=0.8, alpha=0.6,
                   label=f"pooled base rate ({_CONDITIONS[0]})"),
    ]
    if layout == "1x5":
        rect_bottom, rect_top = 0.168, 0.905
        legend_ncol, legend_h, footnote_h = 4, 0.026, 0.090
        suptitle_fs = 10
        footnote_fs = 7.0
    elif layout == "2x2+1":
        rect_bottom, rect_top = 0.21, 0.90
        legend_ncol, legend_h, footnote_h = 2, 0.038, 0.105
        suptitle_fs = 11
        footnote_fs = 8.5
    else:
        rect_bottom, rect_top = 0.21, 0.88
        legend_ncol, legend_h, footnote_h = 2, 0.038, 0.100
        suptitle_fs = 11
        footnote_fs = 8.5

    if layout == "2x2+1":
        fig.subplots_adjust(left=0.08, right=0.98, top=rect_top, bottom=rect_bottom)
    elif layout == "1x5":
        fig.subplots_adjust(left=0.095, right=0.985, top=rect_top, bottom=rect_bottom)
    else:
        fig.tight_layout(rect=[0, rect_bottom, 1, rect_top])

    left, width, center = _plot_column_bounds(fig)
    fig.suptitle(
        "Reliability diagrams — pooled self-forecast events across models\n"
        "caveat: pooling mixes per-model base rates",
        fontsize=suptitle_fs,
        x=center,
        y=0.995 if layout == "1x5" else None,
        va="top" if layout == "1x5" else "top",
    )

    foot_bottom = 0.006 if layout == "1x5" else 0.018
    legend_bottom = foot_bottom + footnote_h + 0.004
    _add_wrapped_panel_caption(
        fig, left, width, foot_bottom, footnote_h,
        _FOOTNOTE_RELIABILITY, fontsize=footnote_fs,
    )
    legend_ax = fig.add_axes([left, legend_bottom, width, legend_h])
    legend_ax.set_axis_off()
    legend_ax.legend(
        handles=legend_handles, loc="center", ncol=legend_ncol,
        fontsize=7.5 if layout == "1x5" else 8, framealpha=0.95,
    )


def figure_c2_reliability_diagrams(
    *, murphy: dict, dimensions, layout: str = "grid_3col",
) -> plt.Figure:
    """Calibration curves per binary dim: observed rate vs stated probability.

    Pooled events across models (per-model bins are too thin), both conditions,
    from the fixed-bin table in ``murphy.json``. Marker area scales with the
    bin's share of total weight; the diagonal is perfect calibration. The grey
    horizontal is the base rate of the primary condition only.

    ``layout`` options:
      - ``grid_3col`` -- legacy 3-column grid (3+2 for five dims)
      - ``2x2+1``     -- four panels on top, fifth spans the bottom row
      - ``1x5``       -- full-page single-column stack
    """
    dim_names = _binary_dim_names(dimensions)
    compact = layout == "1x5"
    fig, axes = _reliability_figure_axes(len(dim_names), layout)
    for ax in axes:
        ax.set_visible(False)

    for k, name in enumerate(dim_names):
        axes[k].set_visible(True)
        # In the single-column stack every panel shares the same x axis, so only
        # the bottom one needs the label; the repeats just eat vertical space.
        _draw_reliability_panel(
            axes[k], name, murphy, compact=compact,
            show_xlabel=not compact or k == len(dim_names) - 1)

    _decorate_reliability_figure(fig, layout=layout)
    return fig


# --------------------------------------------------------------------------
# C3 -- forecast vs outcome for the numeric (MAE-scored) dims
# --------------------------------------------------------------------------
# The binary dims get a Murphy decomposition; MAE has no exact analogue, so
# this figure carries the same two questions geometrically. Distance from the
# diagonal is the error, and the spread of the cloud along the diagonal is the
# task-level signal -- a flat cloud means the forecast never beats the constant
# baseline.
#: What each numeric dimension asks the model to predict about its own run.
_NUMERIC_DIM_CAPTIONS: dict[str, str] = {
    "avg_tokens_step": "mean tokens emitted per step",
    "total_expected_steps": "steps taken before stopping",
    "explore_edit_ratio": "share of actions that read rather than edit",
    "mean_action_block_size": "commands issued per assistant turn",
    "n_constraint_violations": "task-prompt instructions disobeyed",
}

_NUMERIC_LAYOUTS = ("1x5", "2x3")

#: Axis clip percentile. ``avg_tokens_step`` and ``total_expected_steps`` have
#: long right tails -- a few runaway rollouts would squash every other cell
#: into the bottom-left corner. Off-scale cells are counted in the panel note.
_NUMERIC_AXIS_PCTL = 99.0


def _numeric_dim_names(dimensions) -> list[str]:
    """Names of the numeric (MAE-scored) dimensions, in battery order."""
    out = []
    for d in dimensions:
        name = d.name if hasattr(d, "name") else d[0]
        kind = d.kind if hasattr(d, "kind") else d[1]
        if kind != "binary":
            out.append(name)
    return out


def _numeric_panel_title(name: str) -> str:
    unit = DIM_UNITS.get(name, "")
    head = f"{name}  ({unit})" if unit else name
    caption = _NUMERIC_DIM_CAPTIONS.get(name)
    return f"{head}\n{caption}" if caption else head


def _numeric_axis_limits(values: np.ndarray) -> tuple[float, float]:
    """Shared x/y limits for one panel, clipped off the long right tail."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    hi = float(np.percentile(finite, _NUMERIC_AXIS_PCTL))
    lo = float(min(0.0, finite.min()))
    if hi <= lo:
        hi = lo + 1.0
    pad = 0.06 * (hi - lo)
    return lo - pad, hi + pad


def _draw_numeric_panel(
    ax, name: str, points: dict, report: dict, *, compact: bool, show_xlabel: bool,
) -> None:
    """One forecast-vs-outcome panel: cells, diagonal, constant baseline."""
    spread = [
        np.asarray(points[cond][name][key], float)
        for cond in _CONDITIONS if points.get(cond, {}).get(name)
        for key in ("forecast", "actual")
    ]
    lo, hi = _numeric_axis_limits(np.concatenate(spread)) if spread else (0.0, 1.0)

    label_fs = 7.5 if compact else 8.5
    title_fs = 8.0 if compact else 9.5

    off_scale = 0
    for cond in _CONDITIONS:
        rec = points.get(cond, {}).get(name)
        if not rec or not rec["forecast"]:
            continue
        fore = np.asarray(rec["forecast"], float)
        act = np.asarray(rec["actual"], float)
        off_scale += int(((fore > hi) | (act > hi)).sum())
        ax.scatter(fore, act, s=8, alpha=0.32, linewidths=0,
                   color=SELF_COLORS[cond], zorder=3)

    ax.plot([lo, hi], [lo, hi], linestyle="--", color=_NEUTRAL_GREY,
            linewidth=1.0, zorder=2)
    summary = report["conditions"][_CONDITIONS[0]][name]
    ax.axhline(summary["mean_actual"], color=_NEUTRAL_GREY, linewidth=0.8,
               alpha=0.6, zorder=1)

    n_models = len(report["models"])
    note = f"{summary['n_models_beating_baseline']}/{n_models} beat baseline"
    if off_scale:
        note += f"\n{off_scale} cells beyond the axis range (not shown)"
    ax.text(0.035, 0.965, note, transform=ax.transAxes, ha="left", va="top",
            fontsize=label_fs - 0.7, color="#333", linespacing=1.3,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#ddd",
                  "boxstyle": "round,pad=0.28"}, zorder=6)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_box_aspect(1)
    if show_xlabel:
        ax.set_xlabel("forecast", fontsize=label_fs)
    ax.set_ylabel("actual", fontsize=label_fs)
    ax.set_title(_numeric_panel_title(name), fontsize=title_fs, linespacing=1.15)
    ax.tick_params(labelsize=label_fs - 1)


def _numeric_figure_axes(
    n_dims: int, layout: str,
) -> tuple[plt.Figure, np.ndarray, plt.Axes | None]:
    """Return ``(fig, panel_axes, legend_axes)`` for a forecast-vs-outcome layout.

    ``legend_axes`` is the spare grid cell when the layout leaves one free, and
    ``None`` when the legend has to go in the figure margin instead.
    """
    if layout not in _NUMERIC_LAYOUTS:
        raise ValueError(f"layout must be one of {_NUMERIC_LAYOUTS}, got {layout!r}")

    from matplotlib.gridspec import GridSpec
    if layout == "1x5":
        # A single \textwidth-wide strip: square panels stay square when the
        # class scales the figure down to the 6.5in text block.
        fig = plt.figure(figsize=(13.0, 3.5))
        gs = GridSpec(1, n_dims, figure=fig, wspace=0.34)
        axes = [fig.add_subplot(gs[0, i]) for i in range(n_dims)]
        return fig, np.array(axes), None

    # 2x3: five panels, and the spare sixth cell carries the legend and the
    # how-to-read note rather than pushing them into the bottom margin.
    fig = plt.figure(figsize=(7.6, 9.6))
    gs = GridSpec(3, 2, figure=fig, hspace=0.40, wspace=0.28)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(n_dims)]
    legend_ax = fig.add_subplot(gs[(n_dims) // 2, (n_dims) % 2])
    legend_ax.set_axis_off()
    return fig, np.array(axes), legend_ax


_FOOTNOTE_NUMERIC = (
    "Each dot is one (model, task) cell: the model's forecast against that cell's mean "
    "rollout outcome. The dashed diagonal is a perfect forecast, so a cell's vertical "
    "distance from it is that cell's bias (forecast minus mean outcome), not the "
    "per-rollout MAE reported elsewhere. The grey horizontal is "
    "the constant baseline, the fixed \"this model's own mean outcome\" forecast: a cloud "
    "that lies flat along it carries no task-level signal."
)


def _decorate_numeric_figure(
    fig: plt.Figure, *, layout: str, legend_ax: plt.Axes | None,
) -> None:
    legend_handles = [
        plt.Line2D([], [], color=SELF_COLORS["no_thinking"], marker="o",
                   linestyle="", label="no_thinking cell"),
        plt.Line2D([], [], color=SELF_COLORS["thinking"], marker="o",
                   linestyle="", label="thinking cell"),
        plt.Line2D([], [], color=_NEUTRAL_GREY, linestyle="--",
                   label="perfect forecast"),
        plt.Line2D([], [], color=_NEUTRAL_GREY, linewidth=0.8, alpha=0.6,
                   label=f"constant baseline ({_CONDITIONS[0]})"),
    ]
    title = "Forecast vs outcome — the five numeric dimensions"

    if legend_ax is not None:
        fig.subplots_adjust(left=0.095, right=0.975, top=0.915, bottom=0.045)
        legend = legend_ax.legend(handles=legend_handles, loc="upper center",
                                  ncol=1, fontsize=8.5, framealpha=0.95)
        # The caption goes under whatever height the legend actually took, which
        # is only knowable after a draw.
        fig.canvas.draw()
        box = legend.get_window_extent().transformed(legend_ax.transAxes.inverted())
        # matplotlib's wrap=True measures against the figure, not the axes, so
        # a cell-width caption has to be wrapped by hand.
        legend_ax.text(
            0.5, box.y0 - 0.05, textwrap.fill(_FOOTNOTE_NUMERIC, 48),
            transform=legend_ax.transAxes, ha="center", va="top",
            fontsize=7.4, color="#444", linespacing=1.42,
        )
        _, _, center = _plot_column_bounds(fig)
        fig.suptitle(title, fontsize=11.5, x=center, y=0.995, va="top")
        return

    fig.subplots_adjust(left=0.045, right=0.995, top=0.80, bottom=0.30)
    left, width, center = _plot_column_bounds(fig)
    fig.suptitle(title, fontsize=11.5, x=center, y=0.995, va="top")
    foot_bottom, footnote_h, legend_h = 0.012, 0.155, 0.055
    _add_wrapped_panel_caption(
        fig, left, width, foot_bottom, footnote_h, _FOOTNOTE_NUMERIC, fontsize=7.6,
    )
    strip_ax = fig.add_axes([left, foot_bottom + footnote_h + 0.006, width, legend_h])
    strip_ax.set_axis_off()
    strip_ax.legend(handles=legend_handles, loc="center", ncol=4,
                    fontsize=8, framealpha=0.95)


def figure_c3_numeric_calibration(
    *, points: dict, report: dict, dimensions, layout: str = "2x3",
) -> plt.Figure:
    """Forecast vs outcome per numeric dim -- the MAE counterpart to C2.

    ``points`` holds the per-cell ``(forecast, actual)`` pairs and ``report``
    the summary stats, both from ``analyze_continuous``, so the scatter and the
    quoted baseline counts are computed over identical cells.

    ``layout`` options:
      - ``2x3``  -- full-page portrait, five panels plus a legend cell
      - ``1x5``  -- single \\textwidth-wide strip
    """
    dim_names = _numeric_dim_names(dimensions)
    compact = layout == "1x5"
    fig, axes, legend_ax = _numeric_figure_axes(len(dim_names), layout)
    for k, name in enumerate(dim_names):
        _draw_numeric_panel(
            axes[k], name, points, report,
            compact=compact,
            # Every panel in the strip sits on the figure floor; in the portrait
            # grid the spare cell means the last two panels are both exposed.
            show_xlabel=compact or k >= len(dim_names) - 2,
        )
    _decorate_numeric_figure(fig, layout=layout, legend_ax=legend_ax)
    return fig


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def render_all(
    scores: dict, analysis: dict, figures_dir: Path,
    *, families_by_model: dict[str, str] | None = None,
    dimensions=None,
    formats=("png",),
    murphy: dict | None = None,
) -> list[Path]:
    """Render all figures directly under ``figures_dir``.

    The Core 5 always; the two Murphy figures (C1 components, C2 reliability
    diagrams) additionally when a ``murphy`` report dict is supplied. Returns
    the list of written file paths in render order.
    """
    families_by_model = families_by_model or _default_families_by_model()
    dimensions = dimensions if dimensions is not None else config.DIMENSIONS
    figures_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        # "a1_loss_bar": figure_a1_loss_bar(
        #     scores=scores, analysis=analysis,
        #     families_by_model=families_by_model,
        # ),
        "a2_scaling": figure_a2_scaling(
            scores=scores, analysis=analysis,
            families_by_model=families_by_model,
        ),
        # "a3_delta_loss_forest": figure_a3_delta_loss_forest(
        #     scores=scores, analysis=analysis,
        #     families_by_model=families_by_model,
        # ),
        # "b1_per_dim_loss": figure_b1_per_dim_loss(
        #     scores=scores, analysis=analysis, dimensions=dimensions,
        # ),
        # "b2_loss_heatmap": figure_b2_loss_heatmap(
        #     scores=scores, analysis=analysis, dimensions=dimensions,
        # ),
    }
    if murphy is not None:
        # figures["c1_murphy_components"] = figure_c1_murphy_components(
        #     murphy=murphy, dimensions=dimensions,
        # )
        figures["c2_reliability_diagrams"] = figure_c2_reliability_diagrams(
            murphy=murphy, dimensions=dimensions, layout="1x5",
        )
        figures["c2_reliability_diagrams_2x2_plus_1"] = figure_c2_reliability_diagrams(
            murphy=murphy, dimensions=dimensions, layout="2x2+1",
        )

    written: list[Path] = []
    for name, fig in figures.items():
        for fmt in formats:
            out = figures_dir / f"{name}.{fmt}"
            fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.02)
            written.append(out)
        plt.close(fig)
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
    murphy_path = results_dir / "murphy.json"
    murphy = (json.loads(murphy_path.read_text(encoding="utf-8"))
              if murphy_path.exists() else None)
    written = render_all(scores, analysis, figures_dir, formats=formats, murphy=murphy)
    print(f"[plots] wrote {len(written)} figure file(s) under {figures_dir}")
    for p in sorted(written):
        print(f"  - {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
