"""Tests for make_plots.py -- the flat headline ``plots/`` set.

Focused on RQ1, whose bars are annotated with numbers derived from several
files. The risks worth pinning are semantic, not visual: colouring a tie as a
win, and printing a cell denominator inferred from an assumed task count.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import config
import make_plots


MODELS = ["a/high", "b/tied", "c/low", "d/lowest"]
WINS = {"a/high": 0.62, "b/tied": 0.50, "c/low": 0.45, "d/lowest": 0.38}
DIMS = ["p_solve", "documentation"]


def _analysis(*, dropped: dict | None = None, broken: dict | None = None) -> dict:
    dropped = dropped or {m: {d: 0.0 for d in DIMS} for m in MODELS}
    broken = broken or {m: {"no_thinking": 0.0, "thinking": 0.0} for m in MODELS}
    return {
        "rq1_existence": {
            "null": 0.5, "condition": "no_thinking",
            "point_estimate": sum(WINS.values()) / len(WINS),
            "per_model_wins": dict(WINS),
        },
        "missingness": {"dropped_cell_rate": dropped, "broken_forecast_rate": broken},
    }


def _bars(ax):
    return [p for p in ax.patches if p.get_width() > 0]


def test_tie_is_coloured_as_a_tie_not_a_win():
    """A model exactly at the null has not beaten the peer -- it drew."""
    fig = make_plots.figure_rq1(_analysis())
    try:
        ax = fig.axes[0]
        by_width = {round(p.get_width(), 3): p.get_facecolor() for p in _bars(ax)}
        tie_color = by_width[0.500]
        assert tie_color != by_width[0.620], "tie must not share the self-better colour"
        assert tie_color != by_width[0.450], "tie must not share the peer-better colour"
    finally:
        plt.close(fig)


def test_one_bar_per_model_and_chance_is_the_only_reference_line():
    """The across-model mean was removed: unequal denominators make it unreadable."""
    fig = make_plots.figure_rq1(_analysis())
    try:
        ax = fig.axes[0]
        assert len(_bars(ax)) == len(WINS)
        verticals = [round(ln.get_xdata()[0], 3) for ln in ax.get_lines()
                     if len(set(ln.get_xdata())) == 1]
        assert 0.5 in verticals, "the null must still be drawn"
        mean = round(sum(WINS.values()) / len(WINS), 3)
        assert mean not in verticals, "the across-model mean must not be drawn"
        blob = " ".join(t.get_text() for t in ax.texts) + ax.get_title()
        assert "mean" not in blob.lower()
    finally:
        plt.close(fig)


def test_cell_counts_need_a_task_count():
    """Without n_tasks the denominators are unknowable, so none are printed."""
    analysis = _analysis()
    assert make_plots._scored_cell_counts(analysis, None) == {}
    counts = make_plots._scored_cell_counts(analysis, 30)
    assert counts == {m: 30 * len(DIMS) for m in MODELS}


def test_cell_counts_subtract_dropped_cells():
    dropped = {m: {d: 0.0 for d in DIMS} for m in MODELS}
    dropped["c/low"]["p_solve"] = 0.10          # 3 of 30 tasks lost on one dim
    counts = make_plots._scored_cell_counts(_analysis(dropped=dropped), 30)
    assert counts["c/low"] == 57                # 27 + 30
    assert counts["a/high"] == 60


def test_broken_forecasts_mark_the_count_as_an_upper_bound():
    broken = {m: {"no_thinking": 0.0, "thinking": 0.0} for m in MODELS}
    broken["c/low"]["no_thinking"] = 0.2
    fig = make_plots.figure_rq1(_analysis(broken=broken), n_tasks=30)
    try:
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert any(t.startswith("\u2264") for t in texts), \
            "a partially-broken model's cell count must be flagged as a bound"
        assert sum(t.startswith("\u2264") for t in texts) == 1
    finally:
        plt.close(fig)


def test_peer_is_named_when_known():
    fig = make_plots.figure_rq1(_analysis(), cross_model="z-ai/glm-5")
    try:
        blob = " ".join(t.get_text() for t in fig.texts)
        assert "glm-5" in blob
        assert "glm5 forecasting anonymous" in blob
    finally:
        plt.close(fig)


def test_caveat_panel_explains_the_terms_it_uses():
    """"Point estimates only" and "W2" are jargon unless the chart unpacks them."""
    fig = make_plots.figure_rq1(_analysis(), cross_model="z-ai/glm-5")
    try:
        blob = " ".join(t.get_text() for t in fig.texts).lower()
        for phrase in ("point estimates only", "bootstrap", "significance",
                       "standard error", "w2", "forecast variance", "lower bound"):
            assert phrase in blob, f"the caveat never explains {phrase!r}"
    finally:
        plt.close(fig)


def test_caveat_panel_text_stays_inside_its_box():
    """Prose in a fixed-height box: an added line has to fail loudly, not spill."""
    fig = make_plots.figure_rq1(_analysis(), cross_model="z-ai/glm-5")
    try:
        fig.canvas.draw()
        box = fig.patches[0].get_window_extent()
        pad = 4.0
        inside = [t for t in fig.texts
                  if t.get_window_extent().y0 < box.y1]   # the panel's own texts
        assert inside, "expected the caveat text to live in the box"
        for text in inside:
            extent = text.get_window_extent()
            label = text.get_text()[:40]
            assert extent.y0 >= box.y0 + pad, f"caveat text spills below its box: {label!r}"
            assert extent.x1 <= box.x1 - pad, f"caveat text spills right of its box: {label!r}"
    finally:
        plt.close(fig)


# -------------------------------------------------------------- dimensions
def _scores(*, wins_override: dict | None = None) -> dict:
    """Two models over the real 10-dimension battery, with separable values."""
    dims = [d.name for d in config.DIMENSIONS]
    per_model = {}
    for i, m in enumerate(MODELS[:2]):
        per_model[m] = {
            "wins": 0.5,
            "wins_per_dim": {d: (wins_override or {}).get(d, 0.4 + 0.02 * j + 0.1 * i)
                             for j, d in enumerate(dims)},
            # continuous dims get large native-unit losses; binary stay in [0, 1]
            "mean_loss_self_per_dim": {
                d: (0.3 + 0.1 * i) if config.DIMENSION_BY_NAME[d].kind == "binary"
                else (100.0 + 10 * i) for d in dims},
            "mean_loss_cross_per_dim": {
                d: (0.4 + 0.1 * i) if config.DIMENSION_BY_NAME[d].kind == "binary"
                else (200.0 + 10 * i) for d in dims},
            "n_tasks_used": {d: 30 for d in dims},
        }
    return {"cross_model": "z-ai/glm-5", "scores": {"no_thinking": per_model}}


def _dim_axes(fig):
    """(binary-loss, continuous-loss, wins) axes, in creation order."""
    return fig.axes[0], fig.axes[1], fig.axes[2]


def _dim_bars(ax):
    """Bars only -- ``_bars`` would also catch the axvspan shading the split."""
    return [p for c in ax.containers for p in c.patches]


def test_dimension_figure_covers_every_dimension():
    fig = make_plots.figure_dimensions(_scores())
    try:
        ax_bin, ax_cont, ax_win = _dim_axes(fig)
        n = len(config.DIMENSIONS)
        n_binary = sum(d.kind == "binary" for d in config.DIMENSIONS)
        assert len(_dim_bars(ax_bin)) == 2 * n_binary        # self + peer per dim
        assert len(_dim_bars(ax_cont)) == 2 * (n - n_binary)
        assert len(_dim_bars(ax_win)) == n
        labels = [t.get_text() for t in ax_win.get_xticklabels()]
        for d in config.DIMENSIONS:
            assert any(d.name in lab for lab in labels)
        # Top panels used to leave x ticks blank — each bar group must name its dim.
        for ax, n in ((ax_bin, n_binary), (ax_cont, n - n_binary)):
            top_labels = [t.get_text() for t in ax.get_xticklabels()]
            assert len(top_labels) == n
            assert all(lab.strip() for lab in top_labels)
    finally:
        plt.close(fig)


def test_incommensurable_units_never_share_a_scale():
    """Brier and 'tokens' on one axis would bury every binary bar."""
    fig = make_plots.figure_dimensions(_scores())
    try:
        ax_bin, ax_cont, _ = _dim_axes(fig)
        assert ax_bin.get_yscale() == "linear"
        assert ax_cont.get_yscale() == "log"
        assert ax_bin.get_ylim()[1] < 10, "binary panel must stay on a Brier scale"
        assert ax_cont.get_ylim()[1] > 10, "continuous panel must span native units"
    finally:
        plt.close(fig)


def test_dimension_wins_tie_is_not_coloured_as_a_win():
    dims = [d.name for d in config.DIMENSIONS]
    fig = make_plots.figure_dimensions(_scores(wins_override={d: 0.5 for d in dims}))
    try:
        _, _, ax_win = _dim_axes(fig)
        colors = {p.get_facecolor() for p in _dim_bars(ax_win)}
        assert len(colors) == 1
        tie = colors.pop()
        assert tie != matplotlib.colors.to_rgba(make_plots._SELF_BETTER)
        assert tie != matplotlib.colors.to_rgba(make_plots._PEER_BETTER)
    finally:
        plt.close(fig)


def test_dimension_values_are_averaged_across_models():
    """Two models at 0.3 and 0.4 self-loss must print 0.35, not either one."""
    fig = make_plots.figure_dimensions(_scores())
    try:
        ax_bin, _, _ = _dim_axes(fig)
        printed = {t.get_text() for t in ax_bin.texts}
        assert "0.35" in printed and "0.45" in printed
    finally:
        plt.close(fig)


def test_render_headline_plots_adds_the_dimension_figure_when_scores_have_detail(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "analysis.json").write_text(json.dumps({
        **_analysis(),
        "rq2_scaling": {"per_model_wins": dict(WINS),
                        "solve_rates": {m: 0.1 * i for i, m in enumerate(MODELS)},
                        "spearman_rho": 0.8},
        "rq3_elicitation": {"per_model_difference": {m: 0.01 for m in MODELS},
                            "point_estimate": 0.01},
    }), encoding="utf-8")
    (results / "scores.json").write_text(json.dumps(_scores()), encoding="utf-8")
    written = make_plots.render_headline_plots(results)
    assert (results / "plots" / "dimension_accuracy.png") in written


def test_render_headline_plots_writes_every_figure(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "analysis.json").write_text(json.dumps({
        **_analysis(),
        "rq2_scaling": {"per_model_wins": dict(WINS),
                        "solve_rates": {m: 0.1 * i for i, m in enumerate(MODELS)},
                        "spearman_rho": 0.8},
        "rq3_elicitation": {"per_model_difference": {m: 0.01 for m in MODELS},
                            "point_estimate": 0.01},
    }), encoding="utf-8")
    (results / "scores.json").write_text(json.dumps({"cross_model": "z-ai/glm-5"}),
                                         encoding="utf-8")
    (results / "run_summary.json").write_text(json.dumps({"n_tasks": 30}), encoding="utf-8")

    written = make_plots.render_headline_plots(results)
    names = {p.name for p in written}
    assert names == {"rq1_wins_by_model.png", "rq2_wins_vs_solverate.png",
                     "rq3_thinking_delta.png"}
    assert all(p.exists() for p in written)


def test_render_headline_plots_survives_missing_optional_files(tmp_path):
    """scores.json / run_summary.json are optional -- RQ1 degrades, never crashes."""
    results = tmp_path / "results"
    results.mkdir()
    (results / "analysis.json").write_text(json.dumps({
        **_analysis(),
        "rq2_scaling": {"per_model_wins": dict(WINS),
                        "solve_rates": {m: 0.1 * i for i, m in enumerate(MODELS)},
                        "spearman_rho": 0.8},
        "rq3_elicitation": {"per_model_difference": {m: 0.01 for m in MODELS},
                            "point_estimate": 0.01},
    }), encoding="utf-8")
    written = make_plots.render_headline_plots(results)
    assert (results / "plots" / "rq1_wins_by_model.png").exists()
    assert len(written) == 3
