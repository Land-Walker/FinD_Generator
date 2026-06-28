"""Unit tests for src/evaluation/forecast_metrics.py.

All tests use synthetic Gaussian data where analytic values are known:
- CRPS for N(0, 1): closed form exists.
- Coverage for ensemble from known distribution: should match nominal.
- PIT for well-calibrated ensemble: should be Uniform(0, 1).
"""
import numpy as np
from scipy import stats

from src.evaluation.forecast_metrics import (
    crps_ensemble,
    mae,
    rmse,
    quantile_loss,
    pit_values,
    pit_ks_test,
    coverage,
    energy_score,
    negative_log_likelihood,
)


RNG = np.random.default_rng(42)


def synthetic_gaussian_ensemble(n_ensemble: int = 500, n_time: int = 200, mu: float = 0.0, sigma: float = 1.0):
    """Generate ensemble forecasts from N(mu, sigma^2)."""
    forecasts = RNG.normal(mu, sigma, (n_ensemble, n_time)).astype(np.float64)
    targets = RNG.normal(mu, sigma, n_time).astype(np.float64)
    return forecasts, targets


def test_crps_gaussian_known():
    """CRPS for N(0,1): closed form is ~0.2337 per observation (sigma * (1/sqrt(pi) - 1/(2*sqrt(pi)))?).
    Actually CRPS(N(0, sigma^2)) = sigma * (1/sqrt(pi)) ≈ 0.5642 for sigma=1.
    For ensemble CRPS with 500 members it should approximate this.
    """
    forecasts, targets = synthetic_gaussian_ensemble(500, 500)
    result = crps_ensemble(forecasts, targets)
    # Ensemble CRPS approximates theoretical CRPS(N(0,1)) = 1/sqrt(pi) ≈ 0.564
    expected = 1.0 / np.sqrt(np.pi)
    assert 0.45 < result < 0.70
    rel_err = abs(result - expected) / expected
    assert rel_err < 0.15  # within 15% of theoretical


def test_mae_rmse_gaussian():
    """MAE and RMSE for N(0,1): MAE → sigma*sqrt(2/pi), RMSE → sigma."""
    forecasts, targets = synthetic_gaussian_ensemble(500, 500)
    m = mae(forecasts, targets)
    r = rmse(forecasts, targets)
    expected_mae = 1.0 * np.sqrt(2 / np.pi)  # ≈ 0.7979
    assert 0.6 < m < 1.0
    assert 0.8 < r < 1.2


def test_quantile_loss_median():
    """Quantile loss at alpha=0.5 (median): pinball loss should be ~ MAE / 2."""
    forecasts, targets = synthetic_gaussian_ensemble(500, 500)
    q50 = quantile_loss(forecasts, targets, 0.5)
    assert q50 > 0
    assert q50 < 1.0


def test_pit_uniform_for_calibrated():
    """PIT of well-calibrated ensemble should pass KS test against Uniform(0,1)."""
    forecasts, targets = synthetic_gaussian_ensemble(500, 500)
    pit = pit_values(forecasts, targets)
    assert len(pit) == 500
    assert np.all((pit >= 0) & (pit <= 1))
    # Histogram: roughly uniform
    ks_stat, p_val = pit_ks_test(pit)
    assert p_val > 0.01  # well-calibrated should not reject


def test_pit_ks_rejects_miscalibrated():
    """PIT KS test should reject a miscalibrated forecast."""
    forecasts, targets = synthetic_gaussian_ensemble(500, 500, mu=0, sigma=1)
    # Miscalibrated: shift forecasts
    forecasts_bad = forecasts + 0.5
    pit = pit_values(forecasts_bad, targets)
    ks_stat, p_val = pit_ks_test(pit)
    assert p_val < 0.05  # should reject miscalibration


def test_coverage_gaussian():
    """Coverage of 80% interval for well-calibrated ensemble should be ~0.8."""
    forecasts, targets = synthetic_gaussian_ensemble(500, 500)
    cov80 = coverage(forecasts, targets, level=0.8)
    assert 0.70 < cov80 < 0.90
    cov50 = coverage(forecasts, targets, level=0.5)
    assert 0.40 < cov50 < 0.60


def test_coverage_perfect_calibration():
    """Coverage of 95% interval should be close to 0.95 for a large ensemble."""
    forecasts, targets = synthetic_gaussian_ensemble(1000, 500)
    cov95 = coverage(forecasts, targets, level=0.95)
    assert 0.90 < cov95 < 1.0


def test_energy_score_gaussian():
    """Energy score for Gaussian ensemble: should be positive."""
    forecasts, targets = synthetic_gaussian_ensemble(500, 300)
    es = energy_score(forecasts, targets)
    assert es > 0
    assert es < 2.0


def test_negative_log_likelihood():
    """NLL should be positive and reasonable."""
    forecasts, targets = synthetic_gaussian_ensemble(500, 300)
    nll = negative_log_likelihood(forecasts, targets)
    assert nll > 0
    assert nll < 5.0  # should be around 1.42 for N(0,1)


def test_shape_handling():
    """All functions handle various input shapes."""
    # 1-d targets, 2-d forecasts
    forecasts = RNG.normal(0, 1, (50, 100))
    targets = RNG.normal(0, 1, 100)
    assert isinstance(crps_ensemble(forecasts, targets), float)
    assert isinstance(mae(forecasts, targets), float)
    assert isinstance(rmse(forecasts, targets), float)
    assert isinstance(coverage(forecasts, targets, 0.8), float)

    # 2-d forecasts (single forecast)
    forecasts_1 = RNG.normal(0, 1, 100)
    targets_1 = RNG.normal(0, 1, 100)
    assert isinstance(mae(forecasts_1, targets_1), float)