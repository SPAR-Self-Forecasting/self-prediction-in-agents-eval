"""Tests for plots.py -- the seven core figures (EVAL_SPEC_V1.md sec 15.6 / 15.7).

Every figure shows BOTH conditions together (no_thinking + thinking) and, with
the statistics tier pruned, renders point estimates only. ``main()`` / ``render_all``
write figures flat under ``figures_dir`` (no subfolders).

These tests build synthetic ``scores.json`` / ``analysis.json`` dicts in memory
and assert each figure's *structure* (right number of bars/markers/images, the
reference lines, etc). They do not lock visual choices.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from matplotlib.text import Text

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
BINARY_DIM_NAMES = [d[0] for d in DIMS if d[1] == "binary"]


def _model_score(mean_loss: float) -> dict:
    return {
        "mean_loss_per_dim": {
            d: mean_loss + 0.01 * (i - 4) for i, d in enumerate(DIM_NAMES)
        },
        "mean_loss": mean_loss,
        "n_tasks_used": {d: 30 for d in DIM_NAMES},
    }


def make_scores() -> dict:
    per_model_nt = {m: _model_score(0.18 + 0.01 * i) for i, m in enumerate(MODELS)}
    per_model_th = {m: _model_score(0.16 + 0.01 * i) for i, m in enumerate(MODELS)}
    return {
        "scores": {"no_thinking": per_model_nt, "thinking": per_model_th},
    }


def make_analysis() -> dict:
    per_model_loss = {m: 0.18 + 0.01 * i for i, m in enumerate(MODELS)}
    per_model_diff = {
        m: (0.16 + 0.01 * i) - (0.18 + 0.01 * i) for i, m in enumerate(MODELS)
    }
    solve_rates = {m: 0.2 + 0.1 * i for i, m in enumerate(MODELS)}
    return {
        "rq1_accuracy": {
            "estimand": "mean self-forecast loss across models",
            "condition": "no_thinking",
            "n_models": 4,
            "point_estimate": sum(per_model_loss.values()) / len(MODELS),
            "per_model_loss": per_model_loss,
        },
        "rq2_scaling": {
            "estimand": "Spearman rho of mean self loss vs empirical solve rate",
            "condition": "no_thinking",
            "n_models": 4,
            "spearman_rho": 0.8,
            "per_model_loss": per_model_loss,
            "solve_rates": solve_rates,
            "caveat": "descriptive monotonic association; not a significance test",
        },
        "rq3_elicitation": {
            "estimand": "mean paired Δloss(M) = loss(thinking) - loss(no_thinking)",
            "null": 0.0,
            "n_models": 4,
            "excluded_models": [],
            "point_estimate": sum(per_model_diff.values()) / len(MODELS),
            "per_model_difference": per_model_diff,
        },
        "exploratory": {
            "per_dimension_mean_loss": {d: 0.18 for d in DIM_NAMES},
            "partition_capability_mean_loss": 0.18,
            "partition_behavioral_mean_loss": 0.18,
        },
        "missingness": {},
        "solve_rates": solve_rates,
    }


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
        brier = 0.20 if cond == "no_thinking" else 0.18
        for name in BINARY_DIM_NAMES:
            per_model = {
                m: {"n_cells": 25, **_components(brier, 0.05, 0.09, 0.24)}
                for m in MODELS
            }
            mean = {k: v for k, v in _components(brier, 0.05, 0.09, 0.24).items()
                    if k != "binned"}
            mean["n_models"] = len(MODELS)
            conditions[cond][name] = {
                "per_model": per_model,
                "mean_across_models": mean,
                "pooled": {**_components(brier, 0.05, 0.09, 0.24),
                           "caveat": "pooled caveat"},
            }
    return {"method": {}, "n_bins": 10, "models": MODELS, "conditions": conditions}


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


@pytest.fixture
def murphy_report() -> dict:
    return make_murphy()


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _data_panels(fig):
    """Visible subplots that carry data -- excludes the C1 explainer panel."""
    return [ax for ax in fig.axes if ax.get_visible() and ax.has_data()
            and ax.get_label() != plots.MURPHY_KEY_LABEL]


def _key_panel(fig):
    return next(ax for ax in fig.axes if ax.get_label() == plots.MURPHY_KEY_LABEL)


# --------------------------------------------------------------------------
# A1 -- grouped mean self loss bars (2 per model: no_thinking + thinking)
# --------------------------------------------------------------------------
def test_a1_loss_bar_has_two_bars_per_model(scores, analysis, families_by_model):
    fig = plots.figure_a1_loss_bar(
        scores=scores, analysis=analysis, families_by_model=families_by_model,
    )
    ax = fig.axes[0]
    bar_containers = [c for c in ax.containers if hasattr(c, "patches")]
    assert len(bar_containers) == 2, "A1 must have two BarContainers -- one per condition"
    n_bars = sum(len(c) for c in bar_containers)
    assert n_bars == 2 * len(MODELS)


def test_a1_loss_bar_has_pooled_mean_reference(scores, analysis, families_by_model):
    fig = plots.figure_a1_loss_bar(
        scores=scores, analysis=analysis, families_by_model=families_by_model,
    )
    ax = fig.axes[0]
    pooled = analysis["rq1_accuracy"]["point_estimate"]
    found_ref = any(
        any(abs(v - pooled) < 1e-9 for v in line.get_xdata())
        or any(abs(v - pooled) < 1e-9 for v in line.get_ydata())
        for line in ax.get_lines()
    )
    assert found_ref, "A1 must draw a reference line at the RQ1 pooled mean loss"


# --------------------------------------------------------------------------
# A2 -- scaling scatter (2 markers per model: no_thinking + thinking)
# --------------------------------------------------------------------------
def test_a2_scaling_has_two_markers_per_model(scores, analysis, families_by_model):
    fig = plots.figure_a2_scaling(
        scores=scores, analysis=analysis, families_by_model=families_by_model,
    )
    ax = fig.axes[0]
    n_points = sum(len(coll.get_offsets()) for coll in ax.collections)
    assert n_points == 2 * len(MODELS)


def test_a2_scaling_has_no_spearman_annotation(scores, analysis, families_by_model):
    fig = plots.figure_a2_scaling(
        scores=scores, analysis=analysis, families_by_model=families_by_model,
    )
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert not any("ρ" in t or "Spearman" in t for t in texts), \
        "A2 should not annotate a Spearman rho on the plot"


def _crowded_a2_inputs() -> tuple[dict, dict, dict]:
    """A2 inputs with clustered points and long names -- the hard label case."""
    models = [f"vendor{i % 3}/model-{i}-long-enough-name" for i in range(14)]
    # Two tight clumps plus a couple of outliers, all on a shallow slope.
    xs = [0.05 + 0.004 * i for i in range(6)] + \
         [0.24 + 0.004 * i for i in range(6)] + [0.55, 0.60]
    losses = [0.62 - 0.004 * i for i in range(6)] + \
             [0.58 - 0.004 * i for i in range(6)] + [0.50, 0.47]
    scores_ = {"scores": {
        "no_thinking": {m: _model_score(v) for m, v in zip(models, losses)},
        "thinking": {m: _model_score(v - 0.004) for m, v in zip(models, losses)},
    }}
    solve_rates = dict(zip(models, xs))
    analysis_ = {
        "solve_rates": solve_rates,
        "rq2_scaling": {"solve_rates": solve_rates},
    }
    families = {m: m.split("/", 1)[0] for m in models}
    return scores_, analysis_, families


def _label_boxes(fig, ax, names) -> list[tuple[str, tuple]]:
    """Text-only pixel boxes of the model labels on ``ax``.

    ``Annotation.get_window_extent`` unions the text with its leader line, which
    would report overlaps for lines that legitimately pass under a neighbour, so
    the measurement goes through ``Text`` directly.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    return [
        (t.get_text(), tuple(Text.get_window_extent(t, renderer).extents))
        for t in ax.texts if t.get_text() in names
    ]


def test_a2_model_labels_never_overlap_each_other():
    scores_, analysis_, families = _crowded_a2_inputs()
    fig = plots.figure_a2_scaling(
        scores=scores_, analysis=analysis_, families_by_model=families,
    )
    wanted = {plots._short_label(m) for m in families}
    boxes = _label_boxes(fig, fig.axes[0], wanted)
    assert len(boxes) == len(wanted), "every model must be labelled exactly once"
    for i, (name_a, a) in enumerate(boxes):
        for name_b, b in boxes[i + 1:]:
            assert plots._box_overlap(a, b) == 0.0, \
                f"{name_a} label overlaps {name_b}"


def test_a2_model_labels_stay_inside_the_axes():
    scores_, analysis_, families = _crowded_a2_inputs()
    fig = plots.figure_a2_scaling(
        scores=scores_, analysis=analysis_, families_by_model=families,
    )
    ax = fig.axes[0]
    wanted = {plots._short_label(m) for m in families}
    frame = tuple(ax.get_window_extent().extents)
    for name, box in _label_boxes(fig, ax, wanted):
        assert plots._area_outside(box, frame) == 0.0, \
            f"{name} label spills outside the plot frame"


# --------------------------------------------------------------------------
# A3 -- Δloss forest (1 dot per model + pooled diamond)
# --------------------------------------------------------------------------
def test_a3_delta_loss_forest_has_one_row_per_model(scores, analysis, families_by_model):
    fig = plots.figure_a3_delta_loss_forest(
        scores=scores, analysis=analysis, families_by_model=families_by_model,
    )
    ax = fig.axes[0]
    n_points = sum(len(coll.get_offsets()) for coll in ax.collections)
    assert n_points >= len(MODELS)


def test_a3_delta_loss_forest_has_zero_reference(scores, analysis, families_by_model):
    fig = plots.figure_a3_delta_loss_forest(
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
# B1 -- per-dim 2-bar grouped (no_thinking, thinking)
# --------------------------------------------------------------------------
def test_b1_per_dim_loss_renders_one_subplot_per_dimension(scores, analysis, dims):
    fig = plots.figure_b1_per_dim_loss(scores=scores, analysis=analysis, dimensions=dims)
    visible = [ax for ax in fig.axes if ax.get_visible() and ax.has_data()]
    assert len(visible) == len(dims)


def test_b1_per_dim_loss_has_two_bars_per_subplot(scores, analysis, dims):
    fig = plots.figure_b1_per_dim_loss(scores=scores, analysis=analysis, dimensions=dims)
    visible = [ax for ax in fig.axes if ax.get_visible() and ax.has_data()]
    for ax in visible:
        bar_containers = [c for c in ax.containers if hasattr(c, "patches")]
        n_bars = sum(len(c) for c in bar_containers)
        assert n_bars == 2, "each B1 subplot must show 2 bars: no_thinking, thinking"


# --------------------------------------------------------------------------
# B2 -- two side-by-side per-(model, dim) loss heatmaps (one per condition)
# --------------------------------------------------------------------------
def test_b2_loss_heatmap_renders_two_panels(scores, analysis, dims):
    fig = plots.figure_b2_loss_heatmap(scores=scores, analysis=analysis, dimensions=dims)
    image_axes = [ax for ax in fig.axes if ax.get_images()]
    assert len(image_axes) == 2


def test_b2_loss_heatmap_cell_shape_is_models_by_dims(scores, analysis, dims):
    fig = plots.figure_b2_loss_heatmap(scores=scores, analysis=analysis, dimensions=dims)
    image_axes = [ax for ax in fig.axes if ax.get_images()]
    n_cols = len(dims) - len(plots._B2_HEATMAP_EXCLUDE)
    for ax in image_axes:
        arr = ax.get_images()[0].get_array()
        assert arr.shape == (len(MODELS), n_cols)


# --------------------------------------------------------------------------
# C1/C2 -- Murphy decomposition figures (render only when murphy.json exists)
# --------------------------------------------------------------------------
def test_c1_murphy_renders_one_subplot_per_binary_dim(murphy_report, dims):
    fig = plots.figure_c1_murphy_components(murphy=murphy_report, dimensions=dims)
    assert len(_data_panels(fig)) == len(BINARY_DIM_NAMES)


def test_c1_murphy_has_four_bars_and_unc_lines_per_subplot(murphy_report, dims):
    fig = plots.figure_c1_murphy_components(murphy=murphy_report, dimensions=dims)
    for ax in _data_panels(fig):
        n_bars = sum(len(c) for c in ax.containers if hasattr(c, "patches"))
        assert n_bars == 4, "REL + RES groups x [no_thinking, thinking]"
        hlines = [ln for ln in ax.get_lines()
                  if len(set(ln.get_ydata())) == 1 and len(ln.get_ydata()) > 0]
        assert len(hlines) >= 2


def test_c2_reliability_renders_curves_and_diagonal(murphy_report, dims):
    fig = plots.figure_c2_reliability_diagrams(murphy=murphy_report, dimensions=dims)
    visible = [ax for ax in fig.axes if ax.get_visible() and ax.has_data()]
    assert len(visible) == len(BINARY_DIM_NAMES)
    for ax in visible:
        assert len(ax.get_lines()) >= 3
        n_points = sum(len(coll.get_offsets()) for coll in ax.collections)
        assert n_points == 4


def test_c2_reliability_layouts_render_five_panels(murphy_report, dims):
    for layout in ("2x2+1", "1x5"):
        fig = plots.figure_c2_reliability_diagrams(
            murphy=murphy_report, dimensions=dims, layout=layout,
        )
        visible = [ax for ax in fig.axes if ax.get_visible() and ax.has_data()]
        assert len(visible) == len(BINARY_DIM_NAMES)
        plt.close(fig)


def test_c2_reliability_titles_explain_judge_dimensions(murphy_report, dims):
    fig = plots.figure_c2_reliability_diagrams(
        murphy=murphy_report, dimensions=dims, layout="1x5",
    )
    titles = [ax.get_title() for ax in fig.axes if ax.get_visible() and ax.has_data()]
    assert any("type hints" in t for t in titles)
    assert any("guard" in t or "validation" in t for t in titles)
    assert any("doubt" in t or "hedge" in t for t in titles)
    plt.close(fig)


def test_c1_key_panel_uses_the_spare_slot_and_defines_every_component(murphy_report, dims):
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
    fig = plots.figure_c1_murphy_components(murphy=murphy_report, dimensions=dims)
    try:
        data_axes = [ax for ax in fig.axes if ax.get_label() != plots.MURPHY_KEY_LABEL and ax.has_data()]
        labels = " ".join(t.get_text() for ax in data_axes for t in ax.get_xticklabels())
        assert "lower is better" in labels and "higher is better" in labels
        assert "\u2191" not in labels and "\u2193" not in labels
    finally:
        plt.close(fig)


def test_c1_key_panel_text_stays_inside_its_box(murphy_report, dims):
    fig = plots.figure_c1_murphy_components(murphy=murphy_report, dimensions=dims)
    try:
        key = _key_panel(fig)
        fig.canvas.draw()
        box = key.patches[0].get_window_extent()
        pad = 5.0
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
        blob = " ".join(
            t.get_text() for ax in fig.axes for t in ax.texts
        ).lower()
        blob += " ".join(t.get_text() for t in fig.texts).lower()
        assert "dot size" in blob or "larger dots" in blob
        assert "smaller dots" in blob or "thin bin" in blob
    finally:
        plt.close(fig)


def test_c2_draws_exactly_one_base_rate_line(murphy_report, dims):
    fig = plots.figure_c2_reliability_diagrams(murphy=murphy_report, dimensions=dims)
    for ax in [a for a in fig.axes if a.get_visible() and a.has_data()]:
        flat = [ln for ln in ax.get_lines()
                if len(ln.get_xdata()) == 2 and len(set(ln.get_ydata())) == 1]
        assert len(flat) == 1, f"expected 1 base-rate line, got {len(flat)}"


# --------------------------------------------------------------------------
# render_all / CLI smoke -- writes flat PNGs under figures_dir
# --------------------------------------------------------------------------
def test_render_all_writes_one_png_flat(tmp_path, scores, analysis):
    figures_dir = tmp_path / "figures"
    written = plots.render_all(scores, analysis, figures_dir)
    assert len(written) == 1
    assert not any(p.parent != figures_dir for p in written)
    pngs = sorted(p.name for p in figures_dir.glob("*.png"))
    assert pngs == ["a2_scaling.png"]
    for p in figures_dir.glob("*.png"):
        assert p.stat().st_size > 0


def test_render_all_adds_murphy_figures_when_report_supplied(
    tmp_path, scores, analysis, murphy_report,
):
    figures_dir = tmp_path / "figures"
    written = plots.render_all(scores, analysis, figures_dir, murphy=murphy_report)
    assert len(written) == 3
    pngs = sorted(p.name for p in figures_dir.glob("*.png"))
    assert pngs == sorted([
        "a2_scaling.png",
        "c2_reliability_diagrams.png",
        "c2_reliability_diagrams_2x2_plus_1.png",
    ])


def test_main_writes_one_png_flat(tmp_path, scores, analysis):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "scores.json").write_text(json.dumps(scores), encoding="utf-8")
    (results_dir / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
    figures_dir = tmp_path / "figures"

    rc = plots.main(["--results-dir", str(results_dir), "--figures-dir", str(figures_dir)])
    assert rc == 0
    assert not any(p.is_dir() for p in figures_dir.iterdir())
    pngs = sorted(p.name for p in figures_dir.glob("*.png"))
    assert pngs == ["a2_scaling.png"]


def test_main_adds_murphy_figures_when_report_present(
    tmp_path, scores, analysis, murphy_report,
):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "scores.json").write_text(json.dumps(scores), encoding="utf-8")
    (results_dir / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
    (results_dir / "murphy.json").write_text(json.dumps(murphy_report), encoding="utf-8")
    figures_dir = tmp_path / "figures"

    rc = plots.main(["--results-dir", str(results_dir), "--figures-dir", str(figures_dir)])
    assert rc == 0
    pngs = sorted(p.name for p in figures_dir.glob("*.png"))
    assert pngs == sorted([
        "a2_scaling.png",
        "c2_reliability_diagrams.png",
        "c2_reliability_diagrams_2x2_plus_1.png",
    ])


def test_main_errors_loudly_on_missing_results(tmp_path):
    missing = tmp_path / "nowhere"
    with pytest.raises(SystemExit):
        plots.main(["--results-dir", str(missing), "--figures-dir", str(tmp_path / "out")])
