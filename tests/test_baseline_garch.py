"""Unit tests for src/baselines/garch_baseline.py.

Key acceptance test (per MASTER_SPEC §3.3):
  Simulated paths match the fitted unconditional variance in expectation.
"""
import numpy as np
import pytest

from src.baselines.garch_baseline import GARCHBaseline

RNG = np.random.default_rng(42)


def _garch_like_returns(n: int) -> np.ndarray:
    omega, alpha, beta = 0.02, 0.12, 0.84
    sigma2 = np.ones(n) * omega / (1 - alpha - beta)
    r = np.zeros(n)
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
        r[t] = np.sqrt(sigma2[t] / 252) * RNG.normal()
    return r


def test_fit_and_sample_shape():
    train = _garch_like_returns(2000)
    g = GARCHBaseline(dist="t", seed=42)
    g.fit(train)
    history = train[-50:].reshape(1, -1)  # 1 path, 50 history steps
    samples = g.generate_samples(n_samples=8, n_paths=1, horizon=5,
                                 history_returns=history)
    assert samples.shape == (8, 1, 5)


def test_unconditional_variance():
    """Simulated variance should approach theoretical unconditional variance.

    Generate a large number of long paths; the empirical variance over all
    steps should be close to the fitted unconditional variance.
    """
    train = _garch_like_returns(5000)
    g = GARCHBaseline(dist="normal", seed=42)
    g.fit(train)
    theory_var = g.unconditional_variance
    assert theory_var > 0
    assert np.isfinite(theory_var)

    samples = g.generate_samples(n_samples=1, n_paths=500, horizon=100,
                                 history_returns=train[-20:].reshape(1, -1).repeat(500, axis=0))
    emp_var = float(np.var(samples))
    rel_err = abs(emp_var - theory_var) / theory_var
    assert rel_err < 0.25, (
        f"Empirical variance {emp_var:.8f} deviates from "
        f"theoretical {theory_var:.8f} by {rel_err:.2%}"
    )


def test_t_distinction():
    """Student-t GARCH should produce more extreme returns than normal."""
    train = _garch_like_returns(3000)
    g_t = GARCHBaseline(dist="t", seed=42)
    g_t.fit(train)
    g_n = GARCHBaseline(dist="normal", seed=42)
    g_n.fit(train)

    history = train[-30:].reshape(1, -1).repeat(100, axis=0)
    s_t = g_t.generate_samples(n_samples=1, n_paths=100, horizon=50,
                               history_returns=history)
    s_n = g_n.generate_samples(n_samples=1, n_paths=100, horizon=50,
                               history_returns=history)
    k_t = float(np.mean(s_t ** 4) / (np.var(s_t) ** 2))
    k_n = float(np.mean(s_n ** 4) / (np.var(s_n) ** 2))
    assert k_t > k_n, f"t-GARCH kurtosis {k_t:.2f} should exceed normal {k_n:.2f}"


def test_not_fitted_raises():
    g = GARCHBaseline()
    with pytest.raises(RuntimeError, match="not fitted"):
        g.generate_samples(n_samples=2, n_paths=1, horizon=3,
                           history_returns=np.zeros((1, 10)))


def test_wrong_history_rows_raises():
    train = _garch_like_returns(2000)
    g = GARCHBaseline(dist="normal", seed=42)
    g.fit(train)
    with pytest.raises(ValueError, match="history_returns must have"):
        g.generate_samples(n_samples=1, n_paths=5, horizon=3,
                           history_returns=np.zeros((3, 10)))


def test_history_1d_reshaped():
    train = _garch_like_returns(2000)
    g = GARCHBaseline(dist="normal", seed=42)
    g.fit(train)
    samples = g.generate_samples(n_samples=3, n_paths=1, horizon=4,
                                 history_returns=train[-30:])
    assert samples.shape == (3, 1, 4)


def test_invalid_dist_raises():
    with pytest.raises(ValueError, match="dist must be"):
        GARCHBaseline(dist="cauchy")