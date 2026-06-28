"""Unit tests for src/baselines/historical_bootstrap.py."""
import numpy as np
import pytest

from src.baselines.historical_bootstrap import generate_samples


RNG = np.random.default_rng(42)


def test_shape():
    train = RNG.normal(0, 0.01, 1000)
    samples = generate_samples(train, n_samples=8, n_paths=50, horizon=5, rng=RNG)
    assert samples.shape == (8, 50, 5)


def test_mean_unbiased():
    """Bootstrap mean should approximately match training mean."""
    train = RNG.normal(0.0005, 0.01, 5000)
    samples = generate_samples(train, n_samples=32, n_paths=100, horizon=10, rng=RNG)
    assert np.abs(np.mean(samples) - np.mean(train)) < 0.002


def test_no_out_of_sample_values():
    """Every bootstrapped return should be present in training data."""
    train = RNG.normal(0, 0.01, 200)
    unique_train = set(np.round(train, 8))
    samples = generate_samples(train, n_samples=4, n_paths=10, horizon=7, rng=RNG)
    for val in np.round(samples.ravel(), 8):
        assert val in unique_train


def test_deterministic_with_same_rng():
    train = RNG.normal(0, 0.01, 500)
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    s1 = generate_samples(train, n_samples=5, n_paths=20, horizon=6, rng=rng1)
    s2 = generate_samples(train, n_samples=5, n_paths=20, horizon=6, rng=rng2)
    assert np.array_equal(s1, s2)


def test_empty_train_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        generate_samples(np.array([]), n_samples=2, n_paths=3, horizon=5)