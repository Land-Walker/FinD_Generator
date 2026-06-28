"""Unit tests for src/baselines/block_bootstrap.py.

Key acceptance test (per MASTER_SPEC §3.2):
  ACF(|r|) of block-bootstrap samples should be closer to real than
  i.i.d. bootstrap — i.e. block bootstrap preserves volatility clustering.
"""
import numpy as np
import pytest

from src.baselines.block_bootstrap import generate_samples
from src.evaluation.stylized_facts import acf_abs_returns
from src.baselines.historical_bootstrap import generate_samples as hist_bootstrap

RNG = np.random.default_rng(42)


def _garch_like_returns(n: int) -> np.ndarray:
    """Generate returns with volatility clustering."""
    omega, alpha, beta = 0.02, 0.12, 0.84
    sigma2 = np.ones(n) * omega / (1 - alpha - beta)
    r = np.zeros(n)
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
        r[t] = np.sqrt(sigma2[t]) * RNG.normal()
    return r


def test_shape():
    train = _garch_like_returns(2000)
    samples = generate_samples(train, n_samples=4, n_paths=30, horizon=8,
                               block_length=10, rng=RNG)
    assert samples.shape == (4, 30, 8)


def test_block_bootstrap_preserves_vol_clustering():
    """ACF(|r|) of block bootstrap is closer to real than i.i.d. bootstrap."""
    train = _garch_like_returns(5000)
    real_acf_abs = acf_abs_returns(train, lags=10)
    real_acf_abs_mean10 = float(np.mean(real_acf_abs[:10]))

    # i.i.d. bootstrap — should lose volatility clustering
    iid_samples = hist_bootstrap(train, n_samples=1, n_paths=1,
                                 horizon=5000, rng=RNG)
    iid_flat = iid_samples.ravel()
    iid_acf_abs = acf_abs_returns(iid_flat, lags=10)
    iid_acf_abs_mean10 = float(np.mean(iid_acf_abs[:10]))

    # Block bootstrap — should preserve it
    block_samples = generate_samples(train, n_samples=1, n_paths=1,
                                     horizon=5000, block_length=10, rng=RNG)
    block_flat = block_samples.ravel()
    block_acf_abs = acf_abs_returns(block_flat, lags=10)
    block_acf_abs_mean10 = float(np.mean(block_acf_abs[:10]))

    iid_err = abs(iid_acf_abs_mean10 - real_acf_abs_mean10)
    block_err = abs(block_acf_abs_mean10 - real_acf_abs_mean10)
    assert block_err < iid_err, (
        f"Block ACF error {block_err:.4f} should be less than "
        f"i.i.d. error {iid_err:.4f}"
    )


def test_block_length_matters():
    """Block_length=1 → i.i.d. equivalent; larger blocks → longer dependencies."""
    train = _garch_like_returns(2000)
    short = generate_samples(train, n_samples=1, n_paths=1, horizon=1000,
                             block_length=1, rng=RNG)
    long = generate_samples(train, n_samples=1, n_paths=1, horizon=1000,
                            block_length=20, rng=RNG)
    short_acf_abs = acf_abs_returns(short.ravel(), lags=10)
    long_acf_abs = acf_abs_returns(long.ravel(), lags=10)
    # Longer blocks should produce higher ACF(|r|) on average
    assert np.mean(long_acf_abs[:5]) >= np.mean(short_acf_abs[:5])


def test_deterministic_with_same_rng():
    train = _garch_like_returns(1000)
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    s1 = generate_samples(train, n_samples=3, n_paths=10, horizon=6,
                          block_length=5, rng=rng1)
    s2 = generate_samples(train, n_samples=3, n_paths=10, horizon=6,
                          block_length=5, rng=rng2)
    assert np.array_equal(s1, s2)


def test_empty_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        generate_samples(np.array([]), n_samples=2, n_paths=3, horizon=5,
                         rng=RNG)


def test_block_length_zero_raises():
    train = _garch_like_returns(100)
    with pytest.raises(ValueError, match="block_length must be >= 1"):
        generate_samples(train, n_samples=1, n_paths=1, horizon=5,
                         block_length=0, rng=RNG)