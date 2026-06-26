"""Generic statistical inference for self-prediction experiments.

Implements the inference toolbox the pre-registered plan (EVAL_SPEC_V1.md sec 9)
draws on:

- **Bootstrap CIs** -- BCa with a percentile fallback (sec 9.3).
- **Bootstrap p-values** -- two-sided, from a resampled distribution.
- **Mixed model** -- a family random-intercept estimate of a mean (sec 9.2).
- **Spearman / Wilcoxon / one-sample t** -- the convergent secondary tests.
- **OLS** -- coefficient recovery for the RQ2 quadratic regression.
- **Wild cluster bootstrap** with Webb 6-point weights -- the sec 9.3 fallback
  for <~12 clusters.
- **Benjamini-Hochberg** FDR for the exploratory family (sec 9.4).

Nothing here is experiment-specific; ``analysis.py`` wires these to the data.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import scipy.stats as sps
import statsmodels.api as sm
from scipy.stats import norm

# Webb (2014) 6-point distribution -- the recommended weights for the wild
# cluster bootstrap with few clusters (sec 9.3).
_WEBB_POINTS = np.array(
    [-(1.5**0.5), -1.0, -(0.5**0.5), 0.5**0.5, 1.0, 1.5**0.5], dtype=float
)


# --------------------------------------------------------------------------
# Confidence intervals
# --------------------------------------------------------------------------
def percentile_ci(boot: Sequence[float], alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI at confidence ``1 - alpha``."""
    arr = np.asarray(boot, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        raise ValueError("percentile_ci received an all-NaN / empty bootstrap array")
    lo = float(np.percentile(arr, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def bca_ci(
    boot: Sequence[float],
    theta_hat: float,
    jackknife: Sequence[float] | None = None,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bias-corrected and accelerated (BCa) bootstrap CI.

    Falls back to :func:`percentile_ci` (sec 9.3) when the BCa adjustment is
    undefined: no jackknife supplied, a degenerate bootstrap distribution, or a
    zero jackknife spread.
    """
    arr = np.asarray(boot, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        raise ValueError("bca_ci received an all-NaN / empty bootstrap array")

    if jackknife is None:
        return percentile_ci(arr, alpha)

    # Bias-correction z0 from the fraction of resamples below the point estimate.
    prop_below = float(np.mean(arr < theta_hat))
    if prop_below <= 0.0 or prop_below >= 1.0:
        return percentile_ci(arr, alpha)
    z0 = norm.ppf(prop_below)

    # Acceleration from the jackknife skewness.
    jk = np.asarray(jackknife, dtype=float)
    jk = jk[~np.isnan(jk)]
    if jk.size < 2:
        return percentile_ci(arr, alpha)
    jk_dev = jk.mean() - jk
    denom = 6.0 * (np.sum(jk_dev**2) ** 1.5)
    if denom == 0.0:
        return percentile_ci(arr, alpha)
    a = float(np.sum(jk_dev**3) / denom)

    z_lo, z_hi = norm.ppf(alpha / 2.0), norm.ppf(1.0 - alpha / 2.0)
    p_lo = norm.cdf(z0 + (z0 + z_lo) / (1.0 - a * (z0 + z_lo)))
    p_hi = norm.cdf(z0 + (z0 + z_hi) / (1.0 - a * (z0 + z_hi)))
    if not (np.isfinite(p_lo) and np.isfinite(p_hi)):
        return percentile_ci(arr, alpha)
    return float(np.percentile(arr, 100.0 * p_lo)), float(np.percentile(arr, 100.0 * p_hi))


def bootstrap_p_value(boot: Sequence[float], null: float = 0.0) -> float:
    """Two-sided bootstrap p-value for ``statistic == null``."""
    arr = np.asarray(boot, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        raise ValueError("bootstrap_p_value received an all-NaN / empty array")
    below = float(np.mean(arr <= null))
    above = float(np.mean(arr >= null))
    return min(1.0, 2.0 * min(below, above))


# --------------------------------------------------------------------------
# Correlation & univariate tests
# --------------------------------------------------------------------------
def spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Spearman rank correlation; returns (rho, p-value)."""
    res = sps.spearmanr(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return float(res.statistic), float(res.pvalue)


def one_sample_t(x: Sequence[float], popmean: float = 0.0) -> tuple[float, float]:
    """One-sample (= paired-difference) t-test against ``popmean``; (t, p-value)."""
    res = sps.ttest_1samp(np.asarray(x, dtype=float), popmean=popmean)
    return float(res.statistic), float(res.pvalue)


def wilcoxon_signed_rank(x: Sequence[float]) -> tuple[float, float]:
    """Wilcoxon signed-rank test that the median of ``x`` is 0; (statistic, p-value)."""
    res = sps.wilcoxon(np.asarray(x, dtype=float))
    return float(res.statistic), float(res.pvalue)


# --------------------------------------------------------------------------
# Mixed model: family random intercept
# --------------------------------------------------------------------------
def mixed_model_intercept(values: Sequence[float], groups: Sequence) -> dict:
    """Estimate a population mean with a per-group random intercept (sec 9.2).

    Intercept-only fixed effect + random intercept by ``groups`` (the families).
    Returns {estimate, ci_lo, ci_hi, pvalue}. Raises loudly on non-convergence.
    """
    y = np.asarray(values, dtype=float)
    exog = np.ones((y.size, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence chatter -- handled below
        model = sm.MixedLM(endog=y, exog=exog, groups=np.asarray(groups))
        try:
            result = model.fit(reml=True)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"MixedLM failed to fit: {exc}") from exc
    if not getattr(result, "converged", True):
        raise RuntimeError("MixedLM did not converge")
    ci = np.asarray(result.conf_int())[0]
    return {
        "estimate": float(np.asarray(result.fe_params)[0]),
        "ci_lo": float(ci[0]),
        "ci_hi": float(ci[1]),
        "pvalue": float(np.asarray(result.pvalues)[0]),
    }


# --------------------------------------------------------------------------
# OLS
# --------------------------------------------------------------------------
def ols_coefficients(y: Sequence[float], X: np.ndarray) -> np.ndarray:
    """Ordinary least squares coefficients for design matrix ``X`` (with intercept column)."""
    y_arr = np.asarray(y, dtype=float)
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2 or X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"X shape {X_arr.shape} incompatible with y length {y_arr.shape[0]}")
    coef, *_ = np.linalg.lstsq(X_arr, y_arr, rcond=None)
    return coef


# --------------------------------------------------------------------------
# Wild cluster bootstrap (Webb 6-point) -- sec 9.3 fallback
# --------------------------------------------------------------------------
def webb_weights(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw ``n`` weights from Webb's 6-point distribution."""
    return rng.choice(_WEBB_POINTS, size=n, replace=True)


def wild_cluster_bootstrap_mean(
    values: Sequence[float],
    clusters: Sequence,
    B: int,
    seed: int,
) -> np.ndarray:
    """Wild cluster bootstrap distribution of a mean (sec 9.3 fallback).

    Residuals about the observed mean are reweighted, one shared Webb weight per
    cluster, so within-cluster correlation is respected with few clusters.
    Returns ``B`` resampled means.
    """
    if B <= 0:
        raise ValueError(f"B must be positive, got {B}")
    y = np.asarray(values, dtype=float)
    cl = np.asarray(clusters)
    grand = float(np.mean(y))
    resid = y - grand
    unique = list(dict.fromkeys(cl.tolist()))
    cluster_idx = np.array([unique.index(c) for c in cl.tolist()])
    rng = np.random.default_rng(seed)
    out = np.empty(B, dtype=float)
    for b in range(B):
        w = webb_weights(rng, len(unique))
        out[b] = grand + float(np.mean(resid * w[cluster_idx]))
    return out


# --------------------------------------------------------------------------
# Benjamini-Hochberg FDR (sec 9.4)
# --------------------------------------------------------------------------
def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, returned in the input order."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity from the largest p downward, then clamp
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = adj
    return out.tolist()
