"""Tests for plots.py -- the five core figures (EVAL_SPEC_V1.md sec 15.6 / 15.7).

Every figure shows BOTH conditions together (no_thinking + thinking) and, with
the statistics tier pruned, renders point estimates only. ``main()`` renders
the Core 5 for the single cross-predictor under a ``primary__<slug>/`` subfolder.

These tests build synthetic ``scores.json`` / ``analysis.json`` dicts in memory
and assert each figure's *structure* (right number of bars/markers/images, the
0.5/0.0 reference lines, etc). They do not lock visual choices.
"""

from __future__ import annotations

import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: F401  (kept for B1 isinstance filter)
import matplotlib.pyplot as plt
import pytest

import plots


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
MODELS = ["qwen/qwen3-8b", "qwen/qwen3-14b", "google/gemini-2.5-flash-lite", "z-ai/glm-4.7"]
FAMILIES = {
    "qwen/qwen3-8b": "Qwen", "qwen/qwen3-14b": "Qwen",
    "google/gemini-2.5-flash-lite": "Google", "z-ai/glm-4.7": "Zhipu",
}
DIMS = (
    ("p_solve", "binary"),
    ("p_first_attempt_correct", "binary"),
    ("documentation", "binary"),
    ("defensive_code", "binary"),
    ("express_uncertainty", "binary"),
    ("avg_tokens_step", "continuous"),
    ("total_expected_steps", "continuous"),
    ("explore_edit_ratio", "continuous"),
)
DIM_NAMES = [d[0] for d in DIMS]


def _model_score(mean_self, mean_cross):
    return {
        "wins": 0.6,
        "wins_per_dim": {d: 0.5 + 0.05 * (i - 4) for i, d in enumerate(DIM_NAMES)},
        "mean_loss_self_per_dim": {d: mean_self for d in DIM_NAMES},
        "mean_loss_cross_per_dim": {d: mean_cross for d in DIM_NAMES},
        "raw_iai_per_dim": {d: mean_cross - mean_self for d in DIM_NAMES},
        "n_tasks_used": {d: 30 for d in DIM_NAMES},
    }


def make_scores(cross_model: str = "z-ai/glm-5") -> dict:
    per_model_nt = {m: _model_score(0.18, 0.22) for m in MODELS}
    per_model_th = {m: _model_score(0.16, 0.22) for m in MODELS}
    return {
        "cross_model": cross_model,
        "wins": {
            "no_thinking": {m: 0.50 + 0.05 * i for i, m in enumerate(MODELS)},
            "thinking":    {m: 0.55 + 0.05 * i for i, m in enumerate(MODELS)},
        },
        "scores": {"no_thinking": per_model_nt, "thinking": per_model_th},
    }


def make_analysis(cross_model: str = "z-ai/glm-5") -> dict:
    per_model_wins = {m: 0.5 + 0.05 * i for i, m in enumerate(MODELS)}
    per_model_diff = {m: 0.05 for m in MODELS}
    solve_rates = {m: 0.2 + 0.1 * i for i, m in enumerate(MODELS)}
    return {
        "rq1_existence": {
            "estimand": "mean wins", "null": 0.5, "condition": "no_thinking",
            "n_models": 4, "point_estimate": 0.575,
            "per_model_wins": per_model_wins,
        },
        "rq2_scaling": {
            "estimand": "Spearman rho", "condition": "no_thinking", "n_models": 4,
            "spearman_rho": 0.8, "per_model_wins": per_model_wins,
            "solve_rates": solve_rates, "caveat": "upper bound; not a test",
        },
        "rq3_elicitation": {
            "estimand": "mean Δwins", "null": 0.0, "n_models": 4, "excluded_models": [],
            "point_estimate": 0.05, "per_model_difference": per_model_diff,
        },
        "exploratory": {
            "per_dimension_raw_iai_mean": {d: 0.02 for d in DIM_NAMES},
            "partition_capability_mean_raw_iai": 0.02,
            "partition_behavioral_mean_raw_iai": 0.02,
        },
        "missingness": {},
        "solve_rates": solve_rates,
    }


@pytest.fixture
def scores() -> dict:
    return make_scores()


@pytest.fixture
def analysis() -> dict:
    return make_analysis()


@pytest.fixture
def dims() -> tuple:
    return tuple(DIMS)


@pytest.fixture
def families_by_model() -> dict:
    return dict(FAMILIES)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --------------------------------------------------------------------------
# A1 -- grouped wins bars (2 per model: no_thinking + thinking)
# --------------------------------------------------------------------------
def test_a1_wins_bar_has_two_bars_per_iai_model(scores, analysis, families_by_model):
    fig = plots.figure_a1_wins_bar(
        scores=scores, analysis=analysis, families_by_model=families_by_model,
    )
    ax = fig.axes[0]
    bar_containers = [c for c in ax.containers if hasattr(c, "patches")]
    assert len(bar_containers) == 2, "A1 must have two BarContainers -- one per condition"
    n_bars = sum(len(c) for c in bar_containers)
    assert n_bars == 2 * len(MODELS)


def test_a1_wins_bar_has_reference_line_at_0_5(scores, analysis, families_by_model):
    fig = plots.figure_a1_wins_bar(
        scores=scores, analysis=analysis, families_by_model=families_by_model,
    )
    ax = fig.axes[0]
    found_ref = any(
        any(abs(v - 0.5) < 1e-9 for v in line.get_xdata())
        or any(abs(v - 0.5) < 1e-9 for v in line.get_ydata())
        for line in ax.get_lines()
    )
    assert found_ref, "A1 must draw a reference line at the RQ1 null (0.5)"


# --------------------------------------------------------------------------
# A2 -- scaling scatter (2 markers per model: no_thinking + thinking)
# --------------------------------------------------------------------------
def test_a2_scaling_has_two_markers_per_iai_model(scores, analysis, families_by_model):
    fig = plots.figure_a2_scaling(
        scores=scores, analysis=analysis, families_by_model=families_by_model,
    )
    ax = fig.axes[0]
    n_points = sum(len(coll.get_offsets()) for coll in ax.collections)
    assert n_points == 2 * len(MODELS)


def test_a2_scaling_annotates_spearman_rho(scores, analysis, families_by_model):
    fig = plots.figure_a2_scaling(
        scores=scores, analysis=analysis, families_by_model=families_by_model,
    )
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert any("ρ" in t or "Spearman" in t for t in texts), \
        "A2 must annotate the descriptive Spearman rho"


# --------------------------------------------------------------------------
# A3 -- Δwins forest (unchanged: 1 dot per model + pooled diamond)
# --------------------------------------------------------------------------
def test_a3_delta_wins_forest_has_one_row_per_model(scores, analysis, families_by_model):
    fig = plots.figure_a3_delta_wins_forest(
        scores=scores, analysis=analysis, families_by_model=families_by_model,
    )
    ax = fig.axes[0]
    n_points = sum(len(coll.get_offsets()) for coll in ax.collections)
    assert n_points >= len(MODELS)


def test_a3_delta_wins_forest_has_zero_reference(scores, analysis, families_by_model):
    fig = plots.figure_a3_delta_wins_forest(
        scores=scores, analysis=analysis, families_by_model=families_by_model,
    )
    ax = fig.axes[0]
    found_ref = any(
        any(abs(v) < 1e-9 for v in line.get_xdata())
        or any(abs(v) < 1e-9 for v in line.get_ydata())
        for line in ax.get_lines()
    )
    assert found_ref


# --------------------------------------------------------------------------
# B1 -- per-dim 4-bar grouped (self_nt, cross_nt, self_t, cross_t)
# --------------------------------------------------------------------------
def test_b1_per_dim_loss_renders_one_subplot_per_dimension(scores, analysis, dims):
    fig = plots.figure_b1_per_dim_loss(scores=scores, analysis=analysis, dimensions=dims)
    visible = [ax for ax in fig.axes if ax.get_visible() and ax.has_data()]
    assert len(visible) == len(dims)


def test_b1_per_dim_loss_has_four_bars_per_subplot(scores, analysis, dims):
    fig = plots.figure_b1_per_dim_loss(scores=scores, analysis=analysis, dimensions=dims)
    visible = [ax for ax in fig.axes if ax.get_visible() and ax.has_data()]
    for ax in visible:
        bar_containers = [c for c in ax.containers if hasattr(c, "patches")]
        n_bars = sum(len(c) for c in bar_containers)
        assert n_bars == 4, "each B1 subplot must show 4 bars: self_nt, cross_nt, self_t, cross_t"


# --------------------------------------------------------------------------
# B2 -- two side-by-side per-(model, dim) wins heatmaps (one per condition)
# --------------------------------------------------------------------------
def test_b2_wins_heatmap_renders_two_panels(scores, analysis, dims):
    fig = plots.figure_b2_wins_heatmap(scores=scores, analysis=analysis, dimensions=dims)
    # Two real axes for the heatmaps; matplotlib also adds the colorbar axis.
    image_axes = [ax for ax in fig.axes if ax.get_images()]
    assert len(image_axes) == 2


def test_b2_wins_heatmap_cell_shape_is_models_by_dims(scores, analysis, dims):
    fig = plots.figure_b2_wins_heatmap(scores=scores, analysis=analysis, dimensions=dims)
    image_axes = [ax for ax in fig.axes if ax.get_images()]
    for ax in image_axes:
        arr = ax.get_images()[0].get_array()
        assert arr.shape == (len(MODELS), len(dims))


# --------------------------------------------------------------------------
# CLI smoke -- writes 5 PNGs under primary__<slug>/
# --------------------------------------------------------------------------
def test_main_writes_five_pngs_under_primary_subfolder(tmp_path, scores, analysis):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "scores.json").write_text(json.dumps(scores), encoding="utf-8")
    (results_dir / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
    figures_dir = tmp_path / "figures"

    rc = plots.main(["--results-dir", str(results_dir), "--figures-dir", str(figures_dir)])
    assert rc == 0
    subfolders = sorted(p.name for p in figures_dir.iterdir() if p.is_dir())
    assert subfolders == [f"primary__{scores['cross_model'].replace('/', '__')}"]
    primary_folder = figures_dir / subfolders[0]
    pngs = sorted(p.name for p in primary_folder.glob("*.png"))
    assert pngs == sorted([
        "a1_wins_bar.png", "a2_scaling.png", "a3_delta_wins_forest.png",
        "b1_per_dim_loss.png", "b2_wins_heatmap.png",
    ])
    for p in primary_folder.glob("*.png"):
        assert p.stat().st_size > 0


def test_main_errors_loudly_on_missing_results(tmp_path):
    missing = tmp_path / "nowhere"
    with pytest.raises(SystemExit):
        plots.main(["--results-dir", str(missing), "--figures-dir", str(tmp_path / "out")])


# --------------------------------------------------------------------------
# C1/C2 -- Murphy decomposition figures (render only when murphy.json exists)
# --------------------------------------------------------------------------
BINARY_DIM_NAMES = [d[0] for d in DIMS if d[1] == "binary"]


def _components(brier, rel, res, unc):
    return {
        "brier": brier, "reliability": rel, "resolution": res, "uncertainty": unc,
        "bs_recalibrated": brier - rel, "base_rate": 0.4, "n_events": 200,
        "binned": {
            "reliability": rel, "resolution": res, "uncertainty": unc, "residual": 0.001,
            "bins": [
                {"lo": 0.0, "hi": 0.5, "n_events": 120, "weight": 15.0,
                 "mean_forecast": 0.3, "observed_rate": 0.25},
                {"lo": 0.5, "hi": 1.0, "n_events": 80, "weight": 10.0,
                 "mean_forecast": 0.7, "observed_rate": 0.6},
            ],
        },
    }


def make_murphy() -> dict:
    conditions = {}
    for cond in ("no_thinking", "thinking"):
        conditions[cond] = {}
        for name in BINARY_DIM_NAMES:
            per_model = {m: {"n_cells": 25,
                             "self": _components(0.20, 0.05, 0.09, 0.24),
                             "cross": _components(0.22, 0.07, 0.09, 0.24)}
                         for m in MODELS}
            conditions[cond][name] = {
                "per_model": per_model,
                "mean_across_models": {
                    "self": {k: v for k, v in _components(0.20, 0.05, 0.09, 0.24).items()
                             if k != "binned"},
                    "cross": {k: v for k, v in _components(0.22, 0.07, 0.09, 0.24).items()
                              if k != "binned"},
                    "n_models": len(MODELS),
                },
                "pooled": {"self": _components(0.20, 0.05, 0.09, 0.24),
                           "cross": _components(0.22, 0.07, 0.09, 0.24),
                           "caveat": "pooled caveat"},
            }
    return {"method": {}, "n_bins": 10, "cross_model": "z-ai/glm-5",
            "models": MODELS, "conditions": conditions}


@pytest.fixture
def murphy_report() -> dict:
    return make_murphy()


def _data_panels(fig):
    """Visible subplots that carry data -- excludes the C1 explainer panel."""
    return [ax for ax in fig.axes if ax.get_visible() and ax.has_data()
            and ax.get_label() != plots.MURPHY_KEY_LABEL]


def _key_panel(fig):
    return next(ax for ax in fig.axes if ax.get_label() == plots.MURPHY_KEY_LABEL)


def test_c1_murphy_renders_one_subplot_per_binary_dim(murphy_report, dims):
    fig = plots.figure_c1_murphy_components(murphy=murphy_report, dimensions=dims)
    assert len(_data_panels(fig)) == len(BINARY_DIM_NAMES)


def test_c1_murphy_has_eight_bars_and_unc_lines_per_subplot(murphy_report, dims):
    fig = plots.figure_c1_murphy_components(murphy=murphy_report, dimensions=dims)
    for ax in _data_panels(fig):
        n_bars = sum(len(c) for c in ax.containers if hasattr(c, "patches"))
        assert n_bars == 8, "REL + RES groups x [self_nt, cross_nt, self_t, cross_t]"
        # the UNC floor: one dashed horizontal line per condition
        hlines = [ln for ln in ax.get_lines()
                  if len(set(ln.get_ydata())) == 1 and len(ln.get_ydata()) > 0]
        assert len(hlines) >= 2


def test_c2_reliability_renders_curves_and_diagonal(murphy_report, dims):
    fig = plots.figure_c2_reliability_diagrams(murphy=murphy_report, dimensions=dims)
    visible = [ax for ax in fig.axes if ax.get_visible() and ax.has_data()]
    assert len(visible) == len(BINARY_DIM_NAMES)
    for ax in visible:
        # diagonal + 4 calibration curves (self/cross x 2 conditions) at minimum
        assert len(ax.get_lines()) >= 5
        # scatter points carry the bin weights (4 curves x 2 bins each)
        n_points = sum(len(coll.get_offsets()) for coll in ax.collections)
        assert n_points == 8


def test_c1_key_panel_uses_the_spare_slot_and_defines_every_component(murphy_report, dims):
    """The 5 binary dims leave a hole in the 2x3 grid; the key fills it."""
    fig = plots.figure_c1_murphy_components(murphy=murphy_report, dimensions=dims)
    try:
        blob = " ".join(t.get_text() for t in _key_panel(fig).texts)
        for term in ("REL", "RES", "UNC", "Brier"):
            assert term in blob
        for word in ("reliability", "resolution", "uncertainty"):
            assert word in blob, f"{word} is never spelled out"
        assert "lower is better" in blob and "higher is better" in blob
    finally:
        plt.close(fig)


def test_c1_axis_labels_say_which_direction_is_better(murphy_report, dims):
    """Arrows alone did not communicate 'lower/higher is better'."""
    fig = plots.figure_c1_murphy_components(murphy=murphy_report, dimensions=dims)
    try:
        data_axes = [ax for ax in fig.axes if ax.get_label() != plots.MURPHY_KEY_LABEL and ax.has_data()]
        labels = " ".join(t.get_text() for ax in data_axes for t in ax.get_xticklabels())
        assert "lower is better" in labels and "higher is better" in labels
        assert "\u2191" not in labels and "\u2193" not in labels
    finally:
        plt.close(fig)


def test_c1_key_panel_text_stays_inside_its_box(murphy_report, dims):
    """Prose laid out by line count overflows the moment someone adds a line."""
    fig = plots.figure_c1_murphy_components(murphy=murphy_report, dimensions=dims)
    try:
        key = _key_panel(fig)
        fig.canvas.draw()
        box = key.patches[0].get_window_extent()
        pad = 5.0   # px; the last line only grazed the border when this broke
        for text in key.texts:
            extent = text.get_window_extent()
            label = text.get_text()[:40]
            assert extent.y0 >= box.y0 + pad, f"text spills below the box: {label!r}"
            assert extent.y1 <= box.y1 - pad, f"text spills above the box: {label!r}"
            assert extent.x1 <= box.x1 - pad, f"text spills right of the box: {label!r}"
    finally:
        plt.close(fig)


def test_c2_footnote_explains_dot_size(murphy_report, dims):
    fig = plots.figure_c2_reliability_diagrams(murphy=murphy_report, dimensions=dims)
    try:
        blob = " ".join(t.get_text() for t in fig.texts).lower()
        assert "dot size" in blob or "larger dots" in blob
        assert "smaller dots" in blob or "thin bin" in blob
    finally:
        plt.close(fig)


def test_c2_draws_exactly_one_base_rate_line(murphy_report, dims):
    """One grey horizontal per panel, not one per condition.

    The per-condition base rates differ slightly (the scored-cell set differs
    where forecasts fail to parse), so drawing both renders as an unexplained
    double-grey against a legend that promises a single line.
    """
    fig = plots.figure_c2_reliability_diagrams(murphy=murphy_report, dimensions=dims)
    for ax in [a for a in fig.axes if a.get_visible() and a.has_data()]:
        flat = [ln for ln in ax.get_lines()
                if len(ln.get_xdata()) == 2 and len(set(ln.get_ydata())) == 1]
        assert len(flat) == 1, f"expected 1 base-rate line, got {len(flat)}"


def test_main_adds_murphy_figures_when_report_present(tmp_path, scores, analysis, murphy_report):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "scores.json").write_text(json.dumps(scores), encoding="utf-8")
    (results_dir / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
    (results_dir / "murphy.json").write_text(json.dumps(murphy_report), encoding="utf-8")
    figures_dir = tmp_path / "figures"

    rc = plots.main(["--results-dir", str(results_dir), "--figures-dir", str(figures_dir)])
    assert rc == 0
    primary_folder = figures_dir / f"primary__{scores['cross_model'].replace('/', '__')}"
    pngs = sorted(p.name for p in primary_folder.glob("*.png"))
    assert pngs == sorted([
        "a1_wins_bar.png", "a2_scaling.png", "a3_delta_wins_forest.png",
        "b1_per_dim_loss.png", "b2_wins_heatmap.png",
        "c1_murphy_components.png", "c2_reliability_diagrams.png",
    ])


# --------------------------------------------------------------------------
# Peer labelling -- "cross" overstated what was actually run
# --------------------------------------------------------------------------
def _all_text(fig) -> list[str]:
    """Every string a reader can see: titles, axis labels, ticks, legends, annotations."""
    out: list[str] = []
    if fig._suptitle is not None:
        out.append(fig._suptitle.get_text())
    for ax in fig.axes:
        out += [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
        out += [t.get_text() for t in ax.texts]
        out += [t.get_text() for t in ax.get_xticklabels() + ax.get_yticklabels()]
        legend = ax.get_legend()
        if legend is not None:
            out += [t.get_text() for t in legend.get_texts()]
    for legend in fig.legends:
        out += [t.get_text() for t in legend.get_texts()]
    return [s for s in out if s]


def test_peer_phrase_names_the_predictor_and_its_blindness():
    assert plots.peer_phrase("z-ai/glm-5") == "glm5 forecasting anonymous"
    assert plots.peer_short("z-ai/glm-5") == "glm-5"


def test_peer_phrase_is_derived_not_hardcoded():
    """The predictor is chosen by solve rate at stage 4, so a different winner
    must relabel the figures rather than keep saying glm."""
    assert plots.peer_phrase("moonshotai/kimi-k2.5") == "kimik2.5 forecasting anonymous"
    assert "glm" not in plots.peer_phrase("moonshotai/kimi-k2.5")


def test_peer_phrase_degrades_without_a_designated_predictor():
    assert plots.peer_phrase(None) == "peer forecasting anonymous"
    assert plots.peer_short(None) == "peer"


@pytest.mark.parametrize("name", ["a1", "b1", "b2", "c1", "c2"])
def test_no_figure_calls_the_peer_series_cross(name, scores, analysis, murphy_report):
    """"cross" implied a panel of peers, and a peer that knew its subject. Only
    one model ever forecast, and it was never told whose behaviour it was."""
    if name == "a1":
        fig = plots.figure_a1_wins_bar(scores=scores, analysis=analysis,
                                       families_by_model=FAMILIES)
    elif name == "b1":
        fig = plots.figure_b1_per_dim_loss(scores=scores, analysis=analysis, dimensions=DIMS)
    elif name == "b2":
        fig = plots.figure_b2_wins_heatmap(scores=scores, analysis=analysis, dimensions=DIMS)
    elif name == "c1":
        fig = plots.figure_c1_murphy_components(murphy=murphy_report, dimensions=DIMS)
    else:
        fig = plots.figure_c2_reliability_diagrams(murphy=murphy_report, dimensions=DIMS)
    text = " || ".join(_all_text(fig))
    plt.close(fig)
    # Word-bounded: "across" is innocent, "cross" and "cross-predictor" are not.
    stray = re.search(r"\bcross\b|\bcross-", text, re.IGNORECASE)
    assert stray is None, f"{name} still says 'cross': ...{text[max(0, stray.start() - 60):stray.end() + 60]}..."


@pytest.mark.parametrize("name", ["a1", "b1", "b2", "c1", "c2"])
def test_every_peer_figure_names_the_predictor(name, scores, analysis, murphy_report):
    if name == "a1":
        fig = plots.figure_a1_wins_bar(scores=scores, analysis=analysis,
                                       families_by_model=FAMILIES)
    elif name == "b1":
        fig = plots.figure_b1_per_dim_loss(scores=scores, analysis=analysis, dimensions=DIMS)
    elif name == "b2":
        fig = plots.figure_b2_wins_heatmap(scores=scores, analysis=analysis, dimensions=DIMS)
    elif name == "c1":
        fig = plots.figure_c1_murphy_components(murphy=murphy_report, dimensions=DIMS)
    else:
        fig = plots.figure_c2_reliability_diagrams(murphy=murphy_report, dimensions=DIMS)
    text = " || ".join(_all_text(fig))
    plt.close(fig)
    assert "glm5" in text, f"{name} never names the predictor: {text}"


def test_c2_legend_says_who_predicted_and_that_it_was_blind(murphy_report):
    fig = plots.figure_c2_reliability_diagrams(murphy=murphy_report, dimensions=DIMS)
    labels = [t.get_text() for lg in fig.legends for t in lg.get_texts()]
    plt.close(fig)
    assert "glm5 forecasting anonymous (no_thinking)" in labels
    assert "glm5 forecasting anonymous (thinking)" in labels
    assert "self (no_thinking)" in labels


def test_relabelling_follows_a_different_predictor_through_the_figure(murphy_report):
    murphy_report["cross_model"] = "moonshotai/kimi-k2.5"
    fig = plots.figure_c2_reliability_diagrams(murphy=murphy_report, dimensions=DIMS)
    text = " || ".join(_all_text(fig))
    plt.close(fig)
    assert "kimik2.5 forecasting anonymous (no_thinking)" in text
    assert "glm" not in text
