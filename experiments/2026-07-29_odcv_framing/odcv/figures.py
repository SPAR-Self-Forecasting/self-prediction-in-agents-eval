"""Figures A1-A3, B1-B3, C1-C2 (spec §3 step 6c). One figure = one question.

Style: dataviz-skill reference palette (validated); conditions keep a FIXED
categorical assignment everywhere (self=blue, simulated_cross=orange;
ground truth = neutral ink); sequential single-hue for the
A3 count heatmap; one axis per chart; recessive grid; selective direct labels.
Every function takes plain data and a path -- no global state -- so the whole
module smoke-tests offline.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config, groundtruth

# dataviz reference palette (validated instance; light mode)
C_SELF = "#2a78d6"        # slot 1 blue
C_SIM = "#eb6834"         # slot 2 orange
C_TRUTH = "#52514e"       # neutral secondary ink -- ground truth is not a series
C_GRID = "#d9d8d4"
C_TEXT = "#0b0b0b"
CONDITION_COLORS = {"self": C_SELF, "simulated_cross": C_SIM}
CONDITION_LABELS = {"self": "self", "simulated_cross": "simulated-cross"}


def _short(model_id: str) -> str:
    return config.model_by_id(model_id).name


def _style(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, fontsize=11, color=C_TEXT, loc="left")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, axis="x" if ax.get_xlabel() else "both", color=C_GRID, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _paired_framing(per_model: dict, title: str, path, pooled: dict | None = None,
                    funnel_note: str = ""):
    """Per-model dumbbell: self vs simulated-cross mean severity on one axis."""
    rows = [
        (spec.openrouter_id, per_model[spec.openrouter_id])
        for spec in config.MODELS
        if spec.openrouter_id in per_model and "mean_self" in per_model[spec.openrouter_id]
    ]
    n_rows = len(rows) + (1 if pooled and "mean_self" in pooled else 0)
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * n_rows + 1.6))
    ys = np.arange(len(rows), 0, -1)
    cap = 4

    def _point(val, lo, hi, y, color, marker):
        ax.errorbar(
            [val], [y],
            xerr=[[val - lo], [hi - val]],
            fmt=marker, color=color, ecolor=color, markerfacecolor=color,
            capsize=cap, capthick=1.1, elinewidth=1.8, markersize=7, zorder=3,
        )

    xmax = 0.0
    for y, (m, d) in zip(ys, rows):
        self_v, sim_v = d["mean_self"], d["mean_simulated_cross"]
        self_lo, self_hi = d["ci95_self"]
        sim_lo, sim_hi = d["ci95_simulated_cross"]
        xmax = max(xmax, self_hi, sim_hi)
        ax.plot([self_v, sim_v], [y, y], color=C_TRUTH, linewidth=1.2, alpha=0.45, zorder=1)
        _point(self_v, self_lo, self_hi, y, C_SELF, "o")
        _point(sim_v, sim_lo, sim_hi, y, C_SIM, "s")
        gap = d["mean_gap"]
        star = "*" if d.get("significant") else ""
        ax.annotate(f"Δ={gap:+.2f}{star}", (max(self_hi, sim_hi), y),
                    textcoords="offset points", xytext=(8, -3), fontsize=7.5, color=C_TEXT)

    if pooled and "mean_self" in pooled:
        self_v, sim_v = pooled["mean_self"], pooled["mean_simulated_cross"]
        self_lo, self_hi = pooled["ci95_self"]
        sim_lo, sim_hi = pooled["ci95_simulated_cross"]
        xmax = max(xmax, self_hi, sim_hi)
        ax.plot([self_v, sim_v], [0, 0], color=C_TRUTH, linewidth=1.2, alpha=0.45, zorder=1)
        _point(self_v, self_lo, self_hi, 0, C_TRUTH, "o")
        _point(sim_v, sim_lo, sim_hi, 0, C_TRUTH, "s")
        ax.annotate("pooled (summary only)", (max(self_hi, sim_hi), 0),
                    textcoords="offset points", xytext=(8, -3), fontsize=7.5, color=C_TRUTH)

    ax.set_xlim(-0.05, xmax * 1.08 + 0.35)
    ax.set_yticks(list(ys) + ([0] if pooled and "mean_self" in pooled else []))
    ax.set_yticklabels([_short(m) for m, _ in rows]
                       + (["pooled"] if pooled and "mean_self" in pooled else []))
    handles = [
        plt.Line2D([0], [0], marker="o", color=C_SELF, linestyle="None", markersize=7, label="self"),
        plt.Line2D([0], [0], marker="s", color=C_SIM, linestyle="None", markersize=7,
                   label="simulated-cross"),
    ]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="lower right")
    _style(ax, title, xlabel="mean predicted misalignment severity (0–5 rubric)")
    note_parts = ["Error bars = 95% CI", "wild bootstrap, scenario-clustered"]
    if funnel_note:
        note_parts.append(funnel_note)
    note = ("Each scenario gets one score, the median of its 5 predictions; "
            "a point is those per-scenario scores averaged over the model's scenarios.\n"
            + "; ".join(note_parts))
    ax.annotate(note, (0, -0.13), xycoords="axes fraction", va="top",
                fontsize=7, color=C_TRUTH)
    _save(fig, path)


# --------------------------------------------------------------------------
def figure_a1(rq1: dict, path):
    """THE HEADLINE: per-model self vs simulated-cross severity (paired dumbbell)."""
    _paired_framing(
        rq1["per_model"],
        "A1 - Mean predicted severity by framing - "
        f"{rq1['k_significant']} of {rq1['n_tested']} models show a significant "
        "one-sided framing gap",
        path,
        pooled=rq1.get("pooled"),
        funnel_note="Δ = simulated-cross − self; * = one-sided p≤0.05 after Benjamini–Hochberg",
    )


def figure_a2(dist: dict, path):
    """Do predictions reproduce reality's bimodal severity shape?
    dist: {"self": [...], "simulated_cross": [...], "ground_truth": [...]}"""
    levels = np.arange(6)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    width = 0.26
    series = [("self", C_SELF, -1), ("simulated_cross", C_SIM, 0), ("ground_truth", C_TRUTH, 1)]
    for name, color, k in series:
        vals = np.asarray(dist[name], dtype=float)
        counts = [np.sum(np.floor(vals) == lv) for lv in levels]
        frac = np.asarray(counts) / max(1, len(vals))
        label = CONDITION_LABELS.get(name, "ground truth")
        ax.bar(levels + k * width, frac, width=width * 0.92, color=color,
               edgecolor="white", linewidth=1, label=f"{label} (n={len(vals)})")
    ax.set_xticks(levels)
    ax.legend(fontsize=8, frameon=False)
    _style(ax, "A2 - Severity distributions: predictions vs ground truth",
           xlabel="severity level (half-steps floored)", ylabel="fraction of cells")
    _save(fig, path)


def figure_a3(pairs: list[tuple[float, float]], path):
    """6x6 joint count heatmap self x simulated-cross -- a heatmap, not a
    scatter, deliberately: 36 possible positions would overplot."""
    grid = np.zeros((6, 6))
    for v_self, v_sim in pairs:
        grid[int(round(v_sim)), int(round(v_self))] += 1
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    im = ax.imshow(grid, origin="lower", cmap="Blues", aspect="equal")
    for i in range(6):
        for j in range(6):
            if grid[i, j]:
                ax.text(j, i, int(grid[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if grid[i, j] > grid.max() * 0.6 else C_TEXT)
    ax.plot([-0.5, 5.5], [-0.5, 5.5], color=C_TRUTH, linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    fig.colorbar(im, ax=ax, shrink=0.8, label="cells")
    _style(ax, "A3 - Joint counts: self (x) vs simulated-cross (y) - mass above the diagonal = self understates",
           xlabel="self median severity", ylabel="simulated-cross median severity")
    ax.grid(False)
    _save(fig, path)


def figure_b1(rq2: dict, path):
    """RQ2 binary accuracy per condition per model, vs ceiling + baseline."""
    models = [m.openrouter_id for m in config.MODELS]
    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for k, cond in enumerate(("self", "simulated_cross")):
        per = rq2["per_condition"][cond]["per_model"]
        vals = [per[m]["binary_accuracy"] if per[m]["binary_accuracy"] is not None else np.nan for m in models]
        offset = (k - 0.5) * width
        ax.bar(x + offset, vals, width=width * 0.92,
               color=CONDITION_COLORS[cond], edgecolor="white", linewidth=1,
               label=CONDITION_LABELS[cond])
    base = rq2["per_condition"]["self"]["per_model"]
    for i, m in enumerate(models):
        b = base[m]["constant_baseline"]
        if b is not None:
            ax.plot([i - 1.6 * width, i + 1.6 * width], [b, b], color=C_TEXT, linewidth=1.4)
    if rq2.get("judge_ceiling"):
        ax.axhline(rq2["judge_ceiling"], color=C_TRUTH, linewidth=1.2, linestyle="--")
        ax.annotate(f"judge-noise ceiling {rq2['judge_ceiling']:.1%}",
                    (0.99, rq2["judge_ceiling"]), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=7.5, color=C_TRUTH)
    ax.set_xticks(x); ax.set_xticklabels([_short(m) for m in models], rotation=30, ha="right")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8, frameon=False, ncols=2, loc="upper left")
    _style(ax, "B1 - Binary accuracy (all cells with ground truth) vs ceiling and per-model constant baseline (black tick)",
           ylabel="agreement on crossing >=3")
    _save(fig, path)


def figure_b2(rq2: dict, path):
    """The directional asymmetry: FN vs FP counts per condition (D25)."""
    conds = ("self", "simulated_cross")
    x = np.arange(len(conds))
    fn = [rq2["per_condition"][c]["confusion"]["fn"] for c in conds]
    fp = [rq2["per_condition"][c]["confusion"]["fp"] for c in conds]
    # Error-type hues deliberately DIFFERENT from the condition palette --
    # blue/orange already mean self/simulated-cross everywhere else.
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(x - 0.18, fn, width=0.33, color="#eda100", edgecolor="white", linewidth=1,
           label="false negatives (predicted clean, actually violated)")
    ax.bar(x + 0.18, fp, width=0.33, color="#4a3aa7", edgecolor="white", linewidth=1,
           label="false positives (predicted violate, actually clean)")
    for i, (a, b) in enumerate(zip(fn, fp)):
        ax.annotate(str(a), (i - 0.18, a), ha="center", va="bottom", fontsize=8)
        ax.annotate(str(b), (i + 0.18, b), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([CONDITION_LABELS[c] for c in conds])
    ax.legend(fontsize=8, frameon=False)
    _style(ax, "B2 - Error asymmetry per condition: the hypothesis predicts MORE FN under self-framing",
           ylabel="error count (all cells with ground truth)")
    _save(fig, path)


def figure_b3(rq2: dict, path):
    """2x2 confusion matrices for both conditions, side by side (complements
    B2's FN/FP bar view with the full tp/fp/fn/tn breakdown, row-normalized
    so shading reflects within-truth-class rates rather than raw counts)."""
    conds = ("self", "simulated_cross")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    labels = ["predicted\nclean", "predicted\nviolation"]
    truth_labels = ["actually\nclean", "actually\nviolated"]
    for ax, cond in zip(axes, conds):
        cm = rq2["per_condition"][cond]["confusion"]
        tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
        counts = np.array([[tn, fp], [fn, tp]])
        n_clean, n_viol = tn + fp, fn + tp
        row_totals = np.array([[n_clean], [n_viol]])
        rate = counts / row_totals
        im = ax.imshow(rate, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{counts[i, j]}\n({rate[i, j]:.0%})", ha="center", va="center",
                        fontsize=9.5, color="white" if rate[i, j] > 0.6 else C_TEXT)
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_yticks([0, 1]); ax.set_yticklabels(truth_labels, fontsize=8.5)
        ax.set_title(f"{CONDITION_LABELS[cond]}  (n={n_clean + n_viol})", fontsize=10,
                     color=CONDITION_COLORS[cond], loc="left")
        ax.grid(False)
        for side in ax.spines.values():
            side.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.75, label="share of row (truth class)", pad=0.02)
    fig.suptitle("B3 - Confusion matrices: cell shade = within-truth-class rate, label = count (rate)",
                 fontsize=11, color=C_TEXT, x=0.02, ha="left", y=1.03)
    _save(fig, path)


def figure_c1(exploratory: dict, path):
    """EXPLORATORY: gap by actual ground-truth severity bucket."""
    buckets = exploratory["gap_by_gt_severity"]
    names = list(buckets)
    vals = [buckets[b]["mean_gap"] if buckets[b]["mean_gap"] is not None else np.nan for b in names]
    ns = [buckets[b]["n"] for b in names]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(range(len(names)), vals, width=0.6, color=C_SELF, edgecolor="white", linewidth=1)
    for i, (v, n) in enumerate(zip(vals, ns)):
        if not np.isnan(v):
            ax.annotate(f"n={n}", (i, v), ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=8, color=C_TRUTH)
    ax.axhline(0, color=C_TRUTH, linewidth=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    _style(ax, "C1 - mean framing gap by actual ground-truth severity",
           xlabel="ground-truth severity bucket", ylabel="mean gap")
    _save(fig, path)


def figure_c2(layer0: dict, path):
    """Diagnostic: within-cell sample variance per model per condition --
    is the median trustworthy?"""
    models = [m.openrouter_id for m in config.MODELS]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    wv = layer0["within_cell_variance"]
    for k, cond in enumerate(("self", "simulated_cross")):
        vals = [wv.get(f"{m}|{cond}") if wv.get(f"{m}|{cond}") is not None else np.nan for m in models]
        ax.bar(x + (k - 0.5) * 0.36, vals, width=0.33, color=CONDITION_COLORS[cond],
               edgecolor="white", linewidth=1, label=CONDITION_LABELS[cond])
    ax.set_xticks(x); ax.set_xticklabels([_short(m) for m in models], rotation=30, ha="right")
    ax.legend(fontsize=8, frameon=False)
    _style(ax, "C2 - Within-cell sample variance (5 samples): high variance = median is noise in a costume",
           ylabel="mean per-cell variance")
    _save(fig, path)


def figure_d1(cal: dict, path):
    """EXPLORATORY (D37): reliability diagram on the implicit probability
    (fraction of a cell's 5 samples predicting violation). Diagonal = a
    perfectly calibrated response distribution."""
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.plot([0, 1], [0, 1], color=C_TRUTH, linewidth=1, linestyle="--", alpha=0.7)
    ax.annotate("perfect calibration", (0.62, 0.56), rotation=38, fontsize=7.5, color=C_TRUTH)
    for k, cond in enumerate(("self", "simulated_cross")):
        rel = cal[cond]["reliability"]
        xs = [float(b) for b in rel]
        ys = [rel[b]["observed_violation_rate"] for b in rel]
        ns = [rel[b]["n"] for b in rel]
        color = CONDITION_COLORS[cond]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=8,
                label=CONDITION_LABELS[cond])
        # stagger the n-labels per series so shared bins don't collide,
        # and tint them by series so ownership stays readable
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                        xytext=(0, 8 + 9 * k), fontsize=6.5, ha="center", color=color)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _style(ax, "D1 - Reliability of the implicit probability (share of 5 samples saying violate)",
           xlabel="implicit probability of predicting violation",
           ylabel="observed violation rate (ground truth)")
    _save(fig, path)


def figure_d2(cal: dict, path):
    """EXPLORATORY (D37): ranking skill (AUC) per model per condition, scoring
    each cell by its MEAN severity (D39 -- the median ties 94% of self-framing
    pairs and pins five models to a bare 0.5).
    0.5 = the predicted severities carry no ordering signal.

    Rows are ordered by Epoch Capabilities Index, most capable at the top, so
    the reader can see whether discrimination tracks general capability.
    Models Epoch does not score are slotted in by `eci_sort_hint` and keep a
    'no ECI' label, so their position is never read as a measured score.
    """
    def _rank(s):
        # unscored and unslotted models fall to the bottom rather than crashing
        return s.eci if s.eci is not None else (
            s.eci_sort_hint if s.eci_sort_hint is not None else float("-inf"))

    specs = sorted(config.MODELS, key=_rank, reverse=True)
    models = [s.openrouter_id for s in specs]
    labels = [f"{s.name}  ({s.eci:.0f})" if s.eci is not None else f"{s.name}  (no ECI)"
              for s in specs]
    fig, ax = plt.subplots(figsize=(7.6, 0.46 * (len(models) + 2) + 1.2))
    ys = np.arange(len(models), 0, -1)
    offsets = {"self": 0.16, "simulated_cross": -0.16}
    for cond in ("self", "simulated_cross"):
        per = cal[cond]["per_model_auc"]
        band = cal[cond].get("per_model_auc_band") or {}
        color = CONDITION_COLORS[cond]
        # The grey bar is the per-model chance band: the central 95% of AUCs
        # that model's own scores reach when its labels are permuted. A point
        # inside it is not distinguishable from no ranking at all at n=40.
        for y, m in zip(ys, models):
            b = band.get(m)
            if b:
                lo, hi = b["chance_band95"]
                ax.plot([lo, hi], [y + offsets[cond]] * 2, color=C_GRID,
                        linewidth=5, solid_capstyle="butt", zorder=1)
        vals = [per[m] for m in models]
        ax.plot(vals, ys + offsets[cond], "o", color=color, markersize=7.5,
                label=CONDITION_LABELS[cond], alpha=0.95, zorder=3)
        pooled_ci = cal[cond].get("pooled_auc_ci95")
        pooled = cal[cond]["pooled_auc"]
        if pooled_ci:
            ax.plot(pooled_ci, [offsets[cond]] * 2, color=color, linewidth=1.8,
                    solid_capstyle="butt", zorder=2)
        ax.plot(pooled, offsets[cond], "D", color=color, markersize=8, zorder=3)
    ax.axvline(0.5, color=C_TRUTH, linewidth=1, linestyle="--", alpha=0.7)
    ax.annotate("no signal", (0.5, len(models) + 0.5), fontsize=7.5, ha="center", color=C_TRUTH)
    ax.set_yticks(list(ys) + [0])
    ax.set_yticklabels(labels + ["pooled"])
    ax.set_xlim(0.25, 0.92)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _style(ax, "D2 - Discrimination: AUC of mean predicted severity as a ranking score",
           xlabel="AUC  (P(violating cell outranks clean cell); 0.5 = chance)",
           ylabel="models (ordered by Epoch Capabilities Index)")
    ax.annotate(
        "Grey bar = the AUCs this model's own scores reach when its labels are shuffled "
        "(10k draws, central 95%): the no-signal range,\n"
        "NOT a confidence interval on the dot -- a dot outside it beats 95% of shuffles "
        "(raw p < 0.05, before correcting across the 18 tests).\n"
        "A narrow band means the model emitted almost one score, not that its AUC "
        "was measured precisely.\n"
        "Pooled bar = 95% CI, scenario-clustered bootstrap.",
        (0, -0.10), xycoords="axes fraction", va="top", fontsize=7, color=C_TRUTH)
    _save(fig, path)


# --------------------------------------------------------------------------
def render_all(analysis_result: dict, cells: dict, out_dir=None) -> list:
    """Render figures from a run_full_analysis() result + cells."""
    from .analysis import _cell_value  # local import to avoid a cycle

    out_dir = out_dir or config.FIGURES_DIR
    gt = groundtruth.load_ground_truth(config.VARIANT)
    dist = {"self": [], "simulated_cross": [], "ground_truth": []}
    pairs = []
    for spec in config.MODELS:
        m = spec.openrouter_id
        for s in groundtruth.scenarios():
            v_self = _cell_value(cells, m, s, "self")
            v_sim = _cell_value(cells, m, s, "simulated_cross")
            if v_self is not None:
                dist["self"].append(v_self)
            if v_sim is not None:
                dist["simulated_cross"].append(v_sim)
            if gt[m][s] is not None:
                dist["ground_truth"].append(gt[m][s])
            if v_self is not None and v_sim is not None:
                pairs.append((v_self, v_sim))

    made = []
    for name, fn, args in (
        ("figure_a1.png", figure_a1, (analysis_result["rq1"],)),
        ("figure_a2.png", figure_a2, (dist,)),
        ("figure_a3.png", figure_a3, (pairs,)),
        ("figure_b1.png", figure_b1, (analysis_result["rq2"],)),
        ("figure_b2.png", figure_b2, (analysis_result["rq2"],)),
        ("figure_b3.png", figure_b3, (analysis_result["rq2"],)),
        ("figure_c1.png", figure_c1, (analysis_result["exploratory"],)),
        ("figure_c2.png", figure_c2, (analysis_result["layer0"],)),
    ) + (
        (
            ("figure_d1.png", figure_d1, (analysis_result["calibration_proxies"],)),
            ("figure_d2.png", figure_d2, (analysis_result["calibration_proxies"],)),
        )
        if "calibration_proxies" in analysis_result
        else ()
    ):
        path = out_dir / name
        fn(*args, path)
        made.append(path)
    return made
