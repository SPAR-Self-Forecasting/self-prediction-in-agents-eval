"""Tests for src/stats.py -- generic inference primitives."""

import numpy as np
import pytest

from src import stats


# --------------------------------------------------------------------------
# Confidence intervals
# --------------------------------------------------------------------------
def test_percentile_ci():
    boot = np.arange(0, 101, dtype=float)  # 0..100
    lo, hi = stats.percentile_ci(boot, alpha=0.05)
    assert lo == pytest.approx(2.5)
    assert hi == pytest.approx(97.5)


def test_bca_falls_back_to_percentile_without_jackknife():
    boot = np.arange(0, 101, dtype=float)
    assert stats.bca_ci(boot, theta_hat=50.0, jackknife=None) == pytest.approx(
        stats.percentile_ci(boot, alpha=0.05)
    )


def test_bca_symmetric_matches_percentile():
    # Symmetric bootstrap + symmetric jackknife -> z0 = a = 0 -> BCa == percentile.
    rng = np.random.default_rng(0)
    boot = rng.normal(0.0, 1.0, 5000)
    jack = np.linspace(-1.0, 1.0, 21)  # symmetric -> acceleration 0
    bca = stats.bca_ci(boot, theta_hat=float(np.median(boot)), jackknife=jack)
    pct = stats.percentile_ci(boot)
    assert bca[0] == pytest.approx(pct[0], abs=0.15)
    assert bca[1] == pytest.approx(pct[1], abs=0.15)


def test_bca_degenerate_bootstrap_falls_back():
    boot = np.full(100, 3.0)  # all identical -> z0 undefined
    lo, hi = stats.bca_ci(boot, theta_hat=3.0, jackknife=np.array([3.0, 3.1, 2.9]))
    assert lo == pytest.approx(3.0) and hi == pytest.approx(3.0)


# --------------------------------------------------------------------------
# Bootstrap p-value
# --------------------------------------------------------------------------
def test_bootstrap_p_value_all_positive():
    boot = np.arange(1.0, 101.0)  # entirely > 0
    assert stats.bootstrap_p_value(boot, null=0.0) == pytest.approx(0.0)


def test_bootstrap_p_value_centered_is_near_one():
    boot = np.arange(-50.0, 51.0)  # symmetric about 0
    assert stats.bootstrap_p_value(boot, null=0.0) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Correlation & t / Wilcoxon
# --------------------------------------------------------------------------
def test_spearman_monotonic():
    rho, p = stats.spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
    assert rho == pytest.approx(1.0)
    assert p < 0.05


def test_one_sample_t_positive_mean():
    t, p = stats.one_sample_t([1.0, 2.0, 3.0, 4.0, 5.0], popmean=0.0)
    assert t > 0
    assert p < 0.05


def test_wilcoxon_signed_rank_all_positive():
    stat, p = stats.wilcoxon_signed_rank([0.3, 0.5, 0.7, 0.9, 1.1, 1.3])
    assert p < 0.05


# --------------------------------------------------------------------------
# Mixed model (family random intercept)
# --------------------------------------------------------------------------
def test_mixed_model_intercept_balanced_equals_grand_mean():
    # Balanced design -> the fixed intercept is the grand mean.
    values, groups = [], []
    for g, eff in enumerate([0.1, -0.1, 0.2, -0.2, 0.05, -0.05]):
        for delta in (-0.01, 0.0, 0.01):
            values.append(1.0 + eff + delta)
            groups.append(f"fam{g}")
    res = stats.mixed_model_intercept(values, groups)
    assert res["estimate"] == pytest.approx(np.mean(values), abs=1e-6)
    assert res["ci_lo"] < res["estimate"] < res["ci_hi"]


# --------------------------------------------------------------------------
# OLS / quadratic regression (RQ2)
# --------------------------------------------------------------------------
def test_ols_recovers_known_coefficients():
    x = np.linspace(0, 1, 50)
    y = 2.0 + 3.0 * x
    X = np.column_stack([np.ones_like(x), x])
    coef = stats.ols_coefficients(y, X)
    np.testing.assert_allclose(coef, [2.0, 3.0], atol=1e-8)


def test_ols_quadratic_recovers_linear_term():
    x = np.linspace(-1, 1, 80)
    y = 0.5 + 1.5 * x + 4.0 * x**2
    X = np.column_stack([np.ones_like(x), x, x**2])
    coef = stats.ols_coefficients(y, X)
    assert coef[1] == pytest.approx(1.5, abs=1e-8)  # the clean linear term


# --------------------------------------------------------------------------
# Webb weights / wild cluster bootstrap
# --------------------------------------------------------------------------
def test_webb_weights_in_six_point_support():
    rng = np.random.default_rng(1)
    w = stats.webb_weights(rng, 10000)
    support = {-(1.5**0.5), -1.0, -(0.5**0.5), 0.5**0.5, 1.0, 1.5**0.5}
    assert all(any(abs(v - s) < 1e-9 for s in support) for v in w)
    assert w.shape == (10000,)


def test_wild_cluster_bootstrap_mean_deterministic_and_sized():
    values = [0.2, 0.25, -0.1, 0.4, 0.35, 0.1]
    clusters = ["A", "A", "B", "C", "C", "B"]
    b1 = stats.wild_cluster_bootstrap_mean(values, clusters, B=500, seed=42)
    b2 = stats.wild_cluster_bootstrap_mean(values, clusters, B=500, seed=42)
    np.testing.assert_array_equal(b1, b2)
    assert b1.shape == (500,)
    # Bootstrap distribution centred on the observed mean.
    assert np.mean(b1) == pytest.approx(np.mean(values), abs=0.1)


# --------------------------------------------------------------------------
# Benjamini-Hochberg FDR
# --------------------------------------------------------------------------
def test_benjamini_hochberg_uniform_case():
    adj = stats.benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05])
    np.testing.assert_allclose(adj, [0.05] * 5)


def test_benjamini_hochberg_preserves_input_order():
    adj = stats.benjamini_hochberg([0.04, 0.01])
    np.testing.assert_allclose(adj, [0.04, 0.02])


def test_benjamini_hochberg_clamps_to_one():
    adj = stats.benjamini_hochberg([0.9, 0.95])
    assert all(a <= 1.0 for a in adj)
