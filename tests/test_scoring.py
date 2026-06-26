"""Tests for src/scoring.py -- generic loss / standardisation / hierarchical bootstrap."""

import numpy as np
import pytest

from src.scoring import (
    Resample,
    hierarchical_bootstrap,
    identity_resample,
    per_forecast_loss,
    pooled_sd,
    standardize,
    transform,
)
from src.types import Dimension

BINARY = Dimension("p_solve", "binary")        # kind=binary -> Brier
CONTINUOUS = Dimension("avg_tokens_step", "continuous")  # kind=continuous -> MAE


# --------------------------------------------------------------------------
# transform
# --------------------------------------------------------------------------
def test_transform_raw_is_identity():
    np.testing.assert_array_equal(transform([0.0, 1.0, 5.0], "raw"), [0.0, 1.0, 5.0])


def test_transform_log1p_handles_zero():
    # log1p(0) == 0 -- the spec's zero / short-rollout handling (sec 3.1).
    np.testing.assert_allclose(transform([0.0], "log1p"), [0.0])
    np.testing.assert_allclose(transform([np.e - 1.0], "log1p"), [1.0])


def test_transform_log1p_rejects_negative():
    # Counts are non-negative by construction; a negative reaching here is a bug.
    with pytest.raises(ValueError):
        transform([-1.0], "log1p")


# --------------------------------------------------------------------------
# per_forecast_loss
# --------------------------------------------------------------------------
def test_brier_loss_known_value():
    # forecast 0.5 vs outcomes {1, 0}: mean(0.25, 0.25) = 0.25
    assert per_forecast_loss(0.5, [1.0, 0.0], BINARY) == pytest.approx(0.25)


def test_brier_loss_perfect():
    assert per_forecast_loss(1.0, [1.0, 1.0], BINARY) == pytest.approx(0.0)


def test_mae_for_continuous():
    # Continuous dim -> MAE on raw scale (sec 15.7): mean(|2-1|, |2-3|) = 1.0
    assert per_forecast_loss(2.0, [1.0, 3.0], CONTINUOUS) == pytest.approx(1.0)


def test_mae_continuous_with_count_outcomes():
    # MAE in native units: predicting 100 against outcomes {50, 200}
    # gives mean(|100-50|, |100-200|) = mean(50, 100) = 75.0 -- raw, no log.
    assert per_forecast_loss(100.0, [50.0, 200.0], CONTINUOUS) == pytest.approx(75.0)


def test_per_forecast_loss_rejects_empty_outcomes():
    with pytest.raises(ValueError):
        per_forecast_loss(0.5, [], BINARY)


# --------------------------------------------------------------------------
# pooled_sd / standardize
# --------------------------------------------------------------------------
def test_pooled_sd_binary():
    # Sample SD (ddof=1) of {0,0,1,1}
    assert pooled_sd([0.0, 0.0, 1.0, 1.0], "raw") == pytest.approx(0.5773502691896257)


def test_pooled_sd_log_scale():
    vals = [10.0, 100.0, 1000.0]
    assert pooled_sd(vals, "log1p") == pytest.approx(np.std(np.log1p(vals), ddof=1))


def test_pooled_sd_needs_two_values():
    with pytest.raises(ValueError):
        pooled_sd([1.0], "raw")


def test_standardize_divides_by_variance():
    # W1: standardize uses s_d**2 in the denominator -> 0.4 / (0.2**2) = 10.0
    assert standardize(0.4, 0.2) == pytest.approx(10.0)


def test_standardize_rejects_zero_scale():
    # s_d == 0 means a globally-degenerate dimension -- a loud, human-inspectable error.
    with pytest.raises(ValueError):
        standardize(0.1, 0.0)


# --------------------------------------------------------------------------
# hierarchical bootstrap
# --------------------------------------------------------------------------
FAMILIES = {"A": ["a1", "a2"], "B": ["b1"], "C": ["c1", "c2", "c3"]}


def test_identity_resample_structure():
    r = identity_resample(FAMILIES, n_tasks=5, n_rollouts=8)
    assert r.models == ["a1", "a2", "b1", "c1", "c2", "c3"]
    assert r.families == ["A", "A", "B", "C", "C", "C"]
    np.testing.assert_array_equal(r.task_indices, np.arange(5))
    # identity rollout indices: every cell maps 0..n_rollouts-1
    assert r.rollout_indices.shape == (6, 5, 8)
    np.testing.assert_array_equal(r.rollout_indices[3, 2], np.arange(8))


def test_hierarchical_bootstrap_returns_B_values():
    boot = hierarchical_bootstrap(
        FAMILIES, n_tasks=5, n_rollouts=8,
        statistic=lambda r: float(len(r.models)),
        B=200, seed=42,
    )
    assert boot.shape == (200,)


def test_hierarchical_bootstrap_is_deterministic():
    kw = dict(families=FAMILIES, n_tasks=5, n_rollouts=8,
              statistic=lambda r: float(np.sum(r.task_indices)), B=100, seed=42)
    np.testing.assert_array_equal(hierarchical_bootstrap(**kw), hierarchical_bootstrap(**kw))


def test_hierarchical_bootstrap_different_seed_differs():
    kw = dict(families=FAMILIES, n_tasks=5, n_rollouts=8,
              statistic=lambda r: float(np.sum(r.task_indices)), B=100)
    assert not np.array_equal(
        hierarchical_bootstrap(**kw, seed=1), hierarchical_bootstrap(**kw, seed=2)
    )


def test_hierarchical_bootstrap_resample_shapes_and_ranges():
    """Every level is resampled within valid index ranges."""
    seen = {}

    def stat(r: Resample) -> float:
        # 6 families resampled with replacement -> 6 drawn families.
        assert 0 < len(r.models) <= sum(len(v) for v in FAMILIES.values()) * 3
        assert r.task_indices.shape == (5,)
        assert r.task_indices.min() >= 0 and r.task_indices.max() < 5
        assert r.rollout_indices.shape == (len(r.models), 5, 8)
        assert r.rollout_indices.min() >= 0 and r.rollout_indices.max() < 8
        # drawn models always belong to the resampled families
        for fam, mdl in zip(r.families, r.models):
            assert mdl in FAMILIES[fam]
        seen["ok"] = True
        return 0.0

    hierarchical_bootstrap(FAMILIES, 5, 8, stat, B=50, seed=7)
    assert seen.get("ok")


def test_hierarchical_bootstrap_resamples_families_with_replacement():
    """The number of drawn families equals the number of original families."""
    def stat(r: Resample) -> float:
        return float(len(set(r.families)))  # distinct families <= 3 (some dropped/dup)

    boot = hierarchical_bootstrap(FAMILIES, 5, 8, stat, B=300, seed=3)
    # With replacement over 3 families, at least one resample drops a family.
    assert boot.min() < 3
