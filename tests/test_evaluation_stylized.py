"""Unit tests for src/evaluation/stylized_facts.py.

Tests on synthetic data with known properties:
- Student-t with df=3 → known kurtosis, tail index.
- White noise → ACF ~ 0.
- GARCH-like → ACF(|r|) > 0.
"""
import numpy as np
from scipy import stats as sp_stats

from src.evaluation.stylized_facts import (
    kurtosis,
    skewness,
    tail_index_hill,
    acf_returns,
    acf_abs_returns,
    leverage_effect,
    drawdown_distribution,
    realized_vol_distribution,
    var_es,
    all_stylized_facts,
)


RNG = np.random.default_rng(42)


def test_kurtosis_normal():
    """Normal returns → excess kurtosis ~ 0."""
    r = RNG.normal(0, 1, 10000)
    k = kurtosis(r)
    assert abs(k) < 0.15


def test_kurtosis_student_t():
    """Student-t(df=3) → excess kurtosis > 0 (heavy-tailed)."""
    r = sp_stats.t.rvs(df=3, size=20000, random_state=42)
    k = kurtosis(r)
    assert k > 2.0  # definitely heavy-tailed


def test_skewness_normal():
    """Normal returns → skewness ~ 0."""
    r = RNG.normal(0, 1, 10000)
    s = skewness(r)
    assert abs(s) < 0.1


def test_tail_index_hill():
    """Hill estimate for Student-t(df=3) should be in the ballpark of ~3."""
    r = sp_stats.t.rvs(df=3, size=50000, random_state=42)
    hill = tail_index_hill(r, k_frac=0.05)
    # Should be broadly consistent with df=3 (rough range)
    assert 2.0 < hill < 6.0


def test_acf_returns_white_noise():
    """ACF of white noise should be ~0 at all lags."""
    r = RNG.normal(0, 1, 5000)
    acf = acf_returns(r, lags=10)
    # All within 2*sigma ~ 2/sqrt(n) ≈ 0.028
    assert np.all(np.abs(acf) < 0.1)


def test_acf_abs_returns_garch():
    """ACF(|r|) for GARCH-like data should show persistence."""
    # Generate GARCH(1,1)-like data
    n = 5000
    omega, alpha, beta = 0.1, 0.15, 0.8
    sigma2 = np.ones(n)
    r = np.zeros(n)
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
        r[t] = np.sqrt(sigma2[t]) * RNG.normal()
    acf = acf_abs_returns(r, lags=10)
    # At least one lag should show positive autocorrelation
    assert np.any(acf > 0.05)


def test_leverage_effect():
    """Leverage effect for synthetic data with negative asymmetry."""
    n = 5000
    r = RNG.normal(0, 1, n)
    # Inject leverage: large negative returns → larger subsequent |returns|
    for i in range(1, n):
        if r[i - 1] < -2:
            r[i] *= 2.0
    lev = leverage_effect(r, lags=5)
    assert np.any(lev < -0.01)  # at least some negative


def test_drawdown_distribution():
    """Drawdown stats are reasonable."""
    r = np.cumsum(RNG.normal(0, 0.01, 1000))  # random walk returns
    r_ret = np.diff(r) / 100  # approximate returns
    dd = drawdown_distribution(r_ret)
    assert 'max_drawdown' in dd
    assert 'mean_drawdown' in dd
    assert 'max_drawdown_duration' in dd
    assert dd['max_drawdown'] <= 0


def test_realized_vol_distribution():
    """Realized vol produces annualized values."""
    r = RNG.normal(0, 0.01, 500)  # 1% daily vol
    rv = realized_vol_distribution(r, window=21)
    expected_annualised = 0.01 * np.sqrt(252)  # ≈ 0.1587 or ~16%
    mean_rv = np.mean(rv)
    assert 0.10 < mean_rv < 0.25


def test_var_es():
    """VaR and ES at 95%: ES should be more extreme than VaR."""
    r = RNG.normal(0, 1, 10000)
    var95, es95 = var_es(r, 0.95)
    var99, es99 = var_es(r, 0.99)
    assert var95 < 0
    assert es95 < var95  # ES more extreme
    assert es99 < var99
    assert var99 < var95  # 99% VaR more extreme than 95%


def test_all_stylized_facts():
    """All stylized facts return scalar dict."""
    r = RNG.normal(0, 0.01, 500)
    facts = all_stylized_facts(r)
    assert isinstance(facts, dict)
    for k in ['kurtosis', 'skewness', 'var_95', 'es_95', 'var_99', 'es_99']:
        assert k in facts
        assert isinstance(facts[k], float), f"{k} not float"