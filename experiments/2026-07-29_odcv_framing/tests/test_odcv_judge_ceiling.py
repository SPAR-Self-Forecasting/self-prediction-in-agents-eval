"""Slice-2 test: judge-noise ceiling re-derivation (spec §3 step 6a, acceptance 12)."""

import pytest

from odcv import judge_ceiling


@pytest.fixture(scope="module")
def ceiling():
    return judge_ceiling.compute_ceiling("mandated")


def test_ceiling_in_expected_band(ceiling):
    # Spec expectation ~93% on the 9-executor subset (92.8% was the 10-executor
    # figure; the all-12 figure is 93.1%). Re-derived value: 94.1%.
    assert 0.90 <= ceiling["binary_agreement"] <= 0.97
    assert 0.25 <= ceiling["mean_abs_diff"] <= 0.50


def test_comparison_count(ceiling):
    # 4 judges x the 358 usable mandated cells across the 9 executors.
    assert ceiling["n_comparisons"] == 4 * 358
    assert all(d["n"] == 358 for d in ceiling["per_judge"].values())


def test_grok_is_most_idiosyncratic_judge(ceiling):
    # Matches the spec's original per-judge ordering; grok is also an executor,
    # so this is the §3-step-6a analysis caveat to keep in mind.
    per = ceiling["per_judge"]
    assert min(per, key=lambda j: per[j]["binary_agreement"]) == "grok"
