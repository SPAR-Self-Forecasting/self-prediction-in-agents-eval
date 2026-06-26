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
