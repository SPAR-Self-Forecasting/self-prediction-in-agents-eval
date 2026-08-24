"""Slice-1 tests: registry + ground-truth loader (spec §3 step 1, acceptance 1).

The V6 rule: confirm the canonical file by recomputing its key metrics from the
loaded matrix, not by filename. The CSV stores col_avg to 2 decimals and
col_MR% to 1 decimal, so "reproduce to 1e-6" is applied against the CSV's own
printed precision: |recomputed - csv| must be within half an ulp of that
precision (0.005 / 0.05) plus 1e-6.
"""

import pytest

from odcv import config, groundtruth


# --------------------------------------------------------------------------
# Registry (spec §3 step 1, rev 2)
# --------------------------------------------------------------------------
def test_nine_executors_total_358():
    assert len(config.MODELS) == 9
    assert config.TOTAL_EXPECTED_CELLS == 358


def test_ids_unique_and_maverick_dropped():
    assert len(set(config.MODEL_IDS)) == 9
    assert not any("maverick" in m for m in config.MODEL_IDS)


def test_reasoning_policy_rev2():
    # Only grok-4.20 carries an explicit enable; everyone else sends nothing.
    for spec in config.MODELS:
        if spec.openrouter_id == "x-ai/grok-4.20":
            assert spec.reasoning == {"enabled": True}
        else:
            assert spec.reasoning is None


def test_judge_executors_flagged():
    judges = {m.openrouter_id for m in config.MODELS if m.is_judge}
    assert judges == {"x-ai/grok-4.20", "google/gemini-3.1-pro-preview"}


# --------------------------------------------------------------------------
# Loader (acceptance criterion 1)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gt():
    return groundtruth.load_ground_truth("mandated")


def test_scenario_count(gt):
    names = groundtruth.scenarios()
    assert len(names) == 40
    for cells in gt.values():
        assert set(cells) == set(names)


def test_cell_counts_match_registry(gt):
    for spec in config.MODELS:
        assert groundtruth.usable_cells(gt[spec.openrouter_id]) == spec.expected_cells, spec.name
    assert sum(groundtruth.usable_cells(c) for c in gt.values()) == 358


def test_missing_cells_are_none_never_zero(gt):
    # Two known-missing cells (verified by direct CSV inspection this session).
    assert gt["minimax/minimax-m2.7"]["Healthcare-Quality-Assessment"] is None
    assert gt["x-ai/grok-4.20"]["Health-Data-Compliance-Analyst"] is None


def test_values_are_half_step_floats(gt):
    # Judge medians of 3-4 integer scores: only whole or half-step values legal.
    for cells in gt.values():
        for v in cells.values():
            if v is not None:
                assert 0.0 <= v <= 5.0
                assert (2 * v) == int(2 * v), v


def test_recompute_col_avg_and_mr(gt):
    """V6: reproduce the CSV's own summary values from the loaded matrix."""
    summary = groundtruth.csv_summary("mandated")
    for spec in config.MODELS:
        cells = gt[spec.openrouter_id]
        got_avg = groundtruth.avg_severity(cells)
        got_mr = groundtruth.base_rate(cells) * 100.0
        want = summary[spec.openrouter_id]
        assert abs(got_avg - want["col_avg"]) <= 0.005 + 1e-6, spec.name
        assert abs(got_mr - want["col_MR%"]) <= 0.05 + 1e-6, spec.name


def test_known_base_rates(gt):
    # Spot values from the spec registry table (hand-checked).
    assert groundtruth.base_rate(gt["x-ai/grok-4.20"]) == pytest.approx(27 / 39)
    assert groundtruth.base_rate(gt["z-ai/glm-5.1"]) == pytest.approx(9 / 40)
