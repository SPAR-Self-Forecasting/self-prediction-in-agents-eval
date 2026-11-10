"""Tests for make_plots.py -- the flat headline ``plots/`` set.

Focused on RQ1--RQ3 loss readouts and the per-dimension breakdown. The risks
worth pinning are semantic, not visual: wrong metric keys, missing bars, and
jargon in the caveat panel that goes unexplained.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import config
import make_plots


MODELS = ["a/high", "b/mid", "c/low", "d/lowest"]
LOSSES = {"a/high": 0.22, "b/mid": 0.28, "c/low": 0.35, "d/lowest": 0.42}


def _analysis() -> dict:
    return {
        "rq1_accuracy": {
            "condition": "no_thinking",
            "point_estimate": sum(LOSSES.values()) / len(LOSSES),
            "per_model_loss": dict(LOSSES),
        },
        "rq2_scaling": {
            "per_model_loss": dict(LOSSES),
            "solve_rates": {m: 0.1 * (i + 1) for i, m in enumerate(MODELS)},
            "spearman_rho": 0.8,
        },
        "rq3_elicitation": {
            "per_model_difference": {
                m: -0.02 if i % 2 == 0 else 0.03 for i, m in enumerate(MODELS)
            },
            "point_estimate": 0.0025,
        },
    }


def _bars(ax):
    return [p for p in ax.patches if p.get_width() > 0]


def _signed_bars(ax):
    return [p for p in ax.patches if p.get_width() != 0]


# ------------------------------------------------------------------ RQ1
def test_one_bar_per_model():
    fig = make_plots.figure_rq1(_analysis())
    try:
        ax = fig.axes[0]
        assert len(_bars(ax)) == len(LOSSES)
    finally:
        plt.close(fig)


def test_loss_values_are_printed_next_to_bars():
    fig = make_plots.figure_rq1(_analysis())
    try:
        texts = {t.get_text() for t in fig.axes[0].texts}
        for v in LOSSES.values():
            assert f"{v:.4g}" in texts
    finally:
        plt.close(fig)


def test_no_wins_null_reference_line():
    """RQ1 is loss-based now -- the old 0.5 win-rate null must not appear."""
    fig = make_plots.figure_rq1(_analysis())
    try:
        ax = fig.axes[0]
        verticals = [round(ln.get_xdata()[0], 3) for ln in ax.get_lines()
                     if len(set(ln.get_xdata())) == 1]
        assert 0.5 not in verticals
        blob = " ".join(t.get_text() for t in ax.texts) + ax.get_title()
        assert "win" not in blob.lower()
    finally:
        plt.close(fig)


def test_caveat_panel_explains_the_terms_it_uses():
    """"Point estimates" and "W2" are jargon unless the chart unpacks them, and
    the panel must point at where the intervals actually live."""
    fig = make_plots.figure_rq1(_analysis())
    try:
        blob = " ".join(t.get_text() for t in fig.texts).lower()
        for phrase in ("point estimates", "significance.py", "95% ci", "benjamini",
                       "exchangeable", "w2", "forecast variance", "lower bound"):
            assert phrase in blob, f"the caveat never explains {phrase!r}"
    finally:
        plt.close(fig)


def test_caveat_panel_text_stays_inside_its_box():
    """Prose in a fixed-height box: an added line has to fail loudly, not spill."""
    fig = make_plots.figure_rq1(_analysis())
    try:
        fig.canvas.draw()
        box = fig.patches[0].get_window_extent()
        pad = 4.0
        inside = [t for t in fig.texts
                  if t.get_window_extent().y0 < box.y1]
        assert inside, "expected the caveat text to live in the box"
        for text in inside:
            extent = text.get_window_extent()
            label = text.get_text()[:40]
            assert extent.y0 >= box.y0 + pad, f"caveat text spills below its box: {label!r}"
            assert extent.x1 <= box.x1 - pad, f"caveat text spills right of its box: {label!r}"
    finally:
        plt.close(fig)


# ------------------------------------------------------------------ RQ2
def test_rq2_one_point_per_model():
    fig = make_plots.figure_rq2(_analysis())
    try:
        ax = fig.axes[0]
        n_points = sum(len(coll.get_offsets()) for coll in ax.collections)
        assert n_points == len(LOSSES)
    finally:
        plt.close(fig)


def test_rq2_annotates_spearman_rho():
    fig = make_plots.figure_rq2(_analysis())
    try:
        assert "0.800" in fig.axes[0].get_title()
    finally:
        plt.close(fig)


# ------------------------------------------------------------------ RQ3
def test_rq3_has_zero_reference():
    fig = make_plots.figure_rq3(_analysis())
    try:
        ax = fig.axes[0]
        verticals = [round(ln.get_xdata()[0], 3) for ln in ax.get_lines()
                     if len(set(ln.get_xdata())) == 1]
        assert 0.0 in verticals
    finally:
        plt.close(fig)


def test_rq3_negative_delta_coloured_as_thinking_helped():
    fig = make_plots.figure_rq3(_analysis())
    try:
        ax = fig.axes[0]
        helped = matplotlib.colors.to_rgba(make_plots._THINKING_HELPED)
        hurt = matplotlib.colors.to_rgba(make_plots._THINKING_HURT)
        by_width = {round(p.get_width(), 3): p.get_facecolor() for p in _signed_bars(ax)}
        assert by_width[-0.02] == helped
        assert by_width[0.03] == hurt
    finally:
        plt.close(fig)


# ------------------------------------------------------------------ dimensions
def _scores() -> dict:
    """Two models over the real 10-dimension battery, with separable values."""
    dims = [d.name for d in config.DIMENSIONS]
    per_model = {}
    for i, m in enumerate(MODELS[:2]):
        per_model[m] = {
            "mean_loss_per_dim": {
                d: (0.3 + 0.1 * i) if config.DIMENSION_BY_NAME[d].kind == "binary"
                else (100.0 + 10 * i) for d in dims},
            "n_tasks_used": {d: 30 for d in dims},
        }
    return {"scores": {"no_thinking": per_model}}


def _dim_axes(fig):
    """(binary-loss, continuous-loss) axes, in creation order."""
    return fig.axes[0], fig.axes[1]


def _dim_bars(ax):
    return [p for c in ax.containers for p in c.patches]


def test_dimension_figure_covers_every_dimension():
    fig = make_plots.figure_dimensions(_scores())
    try:
        ax_bin, ax_cont = _dim_axes(fig)
        n = len(config.DIMENSIONS)
        n_binary = sum(d.kind == "binary" for d in config.DIMENSIONS)
        assert len(_dim_bars(ax_bin)) == n_binary
        assert len(_dim_bars(ax_cont)) == n - n_binary
        labels = [t.get_text() for t in ax_bin.get_xticklabels()]
        labels += [t.get_text() for t in ax_cont.get_xticklabels()]
        for d in config.DIMENSIONS:
            assert any(d.name in lab for lab in labels)
        for ax, n_ticks in ((ax_bin, n_binary), (ax_cont, n - n_binary)):
            top_labels = [t.get_text() for t in ax.get_xticklabels()]
            assert len(top_labels) == n_ticks
            assert all(lab.strip() for lab in top_labels)
    finally:
        plt.close(fig)


def test_incommensurable_units_never_share_a_scale():
    """Brier and 'tokens' on one axis would bury every binary bar."""
    fig = make_plots.figure_dimensions(_scores())
    try:
        ax_bin, ax_cont = _dim_axes(fig)
        assert ax_bin.get_yscale() == "linear"
        assert ax_cont.get_yscale() == "log"
        assert ax_bin.get_ylim()[1] < 10, "binary panel must stay on a Brier scale"
        assert ax_cont.get_ylim()[1] > 10, "continuous panel must span native units"
    finally:
        plt.close(fig)


def test_dimension_values_are_averaged_across_models():
    """Two models at 0.3 and 0.4 self-loss must print 0.35, not either one."""
    fig = make_plots.figure_dimensions(_scores())
    try:
        ax_bin, _ = _dim_axes(fig)
        printed = {t.get_text() for t in ax_bin.texts}
        assert "0.35" in printed
    finally:
        plt.close(fig)


# ------------------------------------------------------------------ render
def test_render_headline_plots_adds_the_dimension_figure_when_scores_have_detail(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "analysis.json").write_text(json.dumps(_analysis()), encoding="utf-8")
    (results / "scores.json").write_text(json.dumps(_scores()), encoding="utf-8")
    written = make_plots.render_headline_plots(results)
    assert (results / "plots" / "dimension_accuracy.png") in written


def test_render_headline_plots_writes_every_figure(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "analysis.json").write_text(json.dumps(_analysis()), encoding="utf-8")
    (results / "scores.json").write_text(json.dumps({"scores": {}}), encoding="utf-8")

    written = make_plots.render_headline_plots(results)
    names = {p.name for p in written}
    assert names == {"rq1_loss_by_model.png", "rq2_loss_vs_solverate.png",
                     "rq3_thinking_delta.png"}
    assert all(p.exists() for p in written)


def test_render_headline_plots_survives_missing_optional_files(tmp_path):
    """scores.json is optional -- RQ figures still render."""
    results = tmp_path / "results"
    results.mkdir()
    (results / "analysis.json").write_text(json.dumps(_analysis()), encoding="utf-8")
    written = make_plots.render_headline_plots(results)
    assert (results / "plots" / "rq1_loss_by_model.png").exists()
    assert len(written) == 3
