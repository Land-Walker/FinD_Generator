"""Unit tests for src/evaluation/regime_validation.py.

Tests on synthetic regime-labelled data where the regime shift is known:
- Regime A: high vol (sigma=2)
- Regime B: low vol (sigma=0.5)
The KS test should detect this shift.
"""
import numpy as np
from scipy import stats as sp_stats

from src.evaluation.regime_validation import (
    cohens_d,
    energy_distance,
    conditional_distribution_shift_test,
    roundtrip_identifiability,
    regime_validation_report,
)


RNG = np.random.default_rng(42)


def make_synthetic_regime_samples(n_high: int = 200, n_low: int = 500, n_steps: int = 50):
    """Generate synthetic returns for two regimes with known vol difference."""
    high_vol = RNG.normal(0, 0.02, (n_high, n_steps))
    low_vol = RNG.normal(0, 0.005, (n_low, n_steps))
    return high_vol, low_vol


def test_cohens_d_known_difference():
    """Cohen's d should be large for two clearly different samples."""
    x = RNG.normal(0, 1, 1000)
    y = RNG.normal(1, 1, 1000)  # mean shift of 1 sigma
    d = cohens_d(x, y)
    assert abs(d) > 0.8  # large effect


def test_cohens_d_same_distribution():
    """Cohen's d should be ~0 for samples from the same distribution."""
    x = RNG.normal(0, 1, 1000)
    y = RNG.normal(0, 1, 1000)
    d = cohens_d(x, y)
    assert abs(d) < 0.15


def test_energy_distance_separated():
    """Energy distance between separated distributions should be positive."""
    x = RNG.normal(0, 1, 100)
    y = RNG.normal(2, 1, 100)
    ed = energy_distance(x, y)
    assert ed > 0.5


def test_energy_distance_same():
    """Energy distance between samples from the same distribution ~ 0."""
    x1 = RNG.normal(0, 1, 200)
    x2 = RNG.normal(0, 1, 200)
    ed = energy_distance(x1, x2)
    assert abs(ed) < 1.0


def test_conditional_shift_detects_vol_regime():
    """KS test should detect the vol difference between high/low regimes."""
    high_vol, low_vol = make_synthetic_regime_samples(200, 500)
    result = conditional_distribution_shift_test(
        high_vol, low_vol, statistic='realized_vol', n_samples=100, rng=RNG
    )
    assert result['ks_pvalue'] < 0.01
    assert result['cohens_d'] > 1.0
    assert result['ks_stat'] > 0.3


def test_conditional_shift_same_regime():
    """KS test should NOT detect a shift when regimes are truly identical.
    Both samples drawn from N(0, sigma^2) with the same sigma."""
    rng = np.random.default_rng(42)
    sigma = 0.01
    n_steps = 80
    x = rng.normal(0, sigma, (400, n_steps))
    y = rng.normal(0, sigma, (400, n_steps))
    result = conditional_distribution_shift_test(
        x, y, statistic='realized_vol', n_samples=400, rng=rng
    )
    assert result['ks_pvalue'] > 0.05


def test_conditional_shift_mean_return():
    """Shift test works with mean_return statistic."""
    pos = RNG.normal(0.001, 0.01, (200, 50))
    neg = RNG.normal(-0.001, 0.01, (200, 50))
    result = conditional_distribution_shift_test(
        pos, neg, statistic='mean_return', n_samples=100, rng=RNG
    )
    assert result['cohens_d'] > 0.2


def test_conditional_shift_small_sample_warns():
    """Small samples should produce a warning."""
    tiny_g = RNG.normal(0, 1, (5, 10))
    tiny_ng = RNG.normal(0, 1, (5, 10))
    result = conditional_distribution_shift_test(
        tiny_g, tiny_ng, rng=RNG
    )
    assert 'warning' in result
    assert result['warning'] == 'insufficient support — validation underpowered'


def test_roundtrip_identifiability_perfect():
    """Classifier should perfectly recover labels when features are separable."""
    rng = np.random.default_rng(42)
    n = 300
    half = n // 2
    # Regime 1: high vol, positive mean
    feat_high = rng.normal(0.05, 0.02, (half, 3))
    # Regime 2: low vol, near-zero mean
    feat_low = rng.normal(0.00, 0.005, (half, 3))
    features = np.vstack([feat_high, feat_low])
    labels = np.array([0] * half + [1] * half)
    # Shuffle
    idx = rng.permutation(n)
    features, labels = features[idx], labels[idx]
    result = roundtrip_identifiability(
        features, labels, features, labels, classifier_type='logistic', rng=rng
    )
    assert result['test_accuracy'] > result['test_chance']
    assert result['test_accuracy'] > 0.7


def test_roundtrip_chance_level():
    """Classifier on random features should not beat chance."""
    rng = np.random.default_rng(42)
    n = 200
    features = rng.normal(0, 1, (n, 3))
    labels = rng.integers(0, 4, n)
    result = roundtrip_identifiability(
        features, labels, features, labels, classifier_type='logistic', rng=rng
    )
    assert result['test_accuracy'] < result['test_chance'] + 0.15


def test_regime_validation_report():
    """End-to-end regime validation report works."""
    rng = np.random.default_rng(42)
    high_vol = RNG.normal(0, 0.02, (100, 30))
    low_vol = RNG.normal(0, 0.005, (200, 30))
    samples = {
        'vol_regime': {
            'high_vol': high_vol,
            'normal_vol': low_vol,
        },
    }
    results = regime_validation_report(samples, ['vol_regime'], rng=rng)
    assert 'vol_regime' in results
    assert 'high_vol' in results['vol_regime']
    assert 'normal_vol' in results['vol_regime']
    high_result = results['vol_regime']['high_vol']
    assert high_result['ks_pvalue'] < 0.01


def test_regime_validation_small_support():
    """Regime with <50 samples produces warning."""
    rng = np.random.default_rng(42)
    tiny = RNG.normal(0, 0.02, (10, 30))
    medium = RNG.normal(0, 0.005, (100, 30))
    samples = {
        'macro_regime': {
            'stagflation': tiny,
            'expansion': medium,
        },
    }
    results = regime_validation_report(samples, ['macro_regime'], rng=rng)
    stag = results['macro_regime']['stagflation']
    assert 'warning' in stag
    assert 'insufficient support' in stag['warning']
    assert stag['n_g'] == 10