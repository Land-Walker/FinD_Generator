"""
forecast_metrics.py — point and probabilistic forecast evaluation.

Every function accepts (forecasts, targets):
- forecasts: np.ndarray of shape (n_ensemble, n_time, n_features) or (n_time, n_features)
  for ensemble forecasts; the ensemble axis is the first axis.
- targets: np.ndarray of shape (n_time, n_features).

All functions operate on 1-d series internally (flatten or aggregate).
"""

import numpy as np
from scipy import stats
from typing import Optional


def _ensure_1d(arr: np.ndarray) -> np.ndarray:
    return arr.ravel()


def _ensemble_1d(forecasts: np.ndarray) -> np.ndarray:
    """Reshape ensemble forecasts to (n_ensemble, n_samples) 1-d."""
    if forecasts.ndim == 1:
        return forecasts.reshape(1, -1)
    if forecasts.ndim == 2:
        return forecasts
    return forecasts.reshape(forecasts.shape[0], -1)


def _target_1d(targets: np.ndarray) -> np.ndarray:
    return targets.ravel()


def crps_ensemble(forecasts: np.ndarray, targets: np.ndarray) -> float:
    """Continuous Ranked Probability Score for ensemble forecasts.

    Uses the ensemble (empirical CDF) formulation:
    CRPS = E|X - y| - 0.5 * E|X - X'|
    where X, X' are independent draws from the ensemble.

    Parameters
    ----------
    forecasts : np.ndarray, shape (n_ensemble, n_time) or (n_ensemble, n_time, 1)
    targets : np.ndarray, shape (n_time,) or (n_time, 1)
    """
    y = _target_1d(targets)
    ens = _ensemble_1d(forecasts)  # (n_ensemble, n_samples)
    n_ens, n_samples = ens.shape
    # Per sample-point CRPS
    abs_err = np.mean(np.abs(ens - y[np.newaxis, :]), axis=0)  # (n_samples,)
    # Mean pairwise absolute difference within ensemble
    if n_ens > 1:
        pw = np.zeros(n_samples)
        for i in range(n_ens):
            for j in range(i + 1, n_ens):
                pw += np.abs(ens[i] - ens[j])
        pw *= 2.0 / (n_ens * (n_ens - 1))
    else:
        pw = 0.0
    crps_per_sample = abs_err - 0.5 * pw
    return float(np.mean(crps_per_sample))


def mae(forecasts: np.ndarray, targets: np.ndarray) -> float:
    """Mean Absolute Error. For ensembles, uses the ensemble mean."""
    y = _target_1d(targets)
    if forecasts.ndim >= 2 and forecasts.shape[0] > 1:
        pred = np.mean(_ensemble_1d(forecasts), axis=0)
    else:
        pred = _ensure_1d(forecasts)
    return float(np.mean(np.abs(pred - y)))


def rmse(forecasts: np.ndarray, targets: np.ndarray) -> float:
    """Root Mean Squared Error. For ensembles, uses the ensemble mean."""
    y = _target_1d(targets)
    if forecasts.ndim >= 2 and forecasts.shape[0] > 1:
        pred = np.mean(_ensemble_1d(forecasts), axis=0)
    else:
        pred = _ensure_1d(forecasts)
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def quantile_loss(forecasts: np.ndarray, targets: np.ndarray, alpha: float) -> float:
    """Pinball / quantile loss at quantile alpha.

    Parameters
    ----------
    forecasts : np.ndarray
        Ensemble forecasts. Quantile is taken from the empirical distribution.
    targets : np.ndarray
    alpha : float, in (0, 1)
    """
    y = _target_1d(targets)
    ens = _ensemble_1d(forecasts)  # (n_ensemble, n_samples)
    q_hat = np.quantile(ens, alpha, axis=0)
    err = y - q_hat
    return float(np.mean(np.where(err >= 0, alpha * err, (alpha - 1) * err)))


def pit_values(forecasts: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Probability Integral Transform values.

    PIT_i = F_i(y_i) where F_i is the empirical CDF of the ensemble at step i.

    Under a well-calibrated forecast, PIT values should be Uniform(0, 1).
    """
    y = _target_1d(targets)
    ens = _ensemble_1d(forecasts)  # (n_ensemble, n_samples)
    n_ens, n_samples = ens.shape
    pit = np.zeros(n_samples)
    for t in range(n_samples):
        pit[t] = np.mean(ens[:, t] <= y[t])
    return pit


def pit_ks_test(pit: np.ndarray) -> tuple:
    """KS test of PIT values against Uniform(0, 1).

    Returns (ks_statistic, p_value).
    """
    ks_stat, p_value = stats.kstest(pit, 'uniform', args=(0, 1))
    return float(ks_stat), float(p_value)


def coverage(forecasts: np.ndarray, targets: np.ndarray, level: float = 0.8) -> float:
    """Empirical coverage of the central prediction interval at the given level.

    Parameters
    ----------
    level : float, in (0, 1). E.g. 0.8 → 80 % central interval.
    """
    y = _target_1d(targets)
    ens = _ensemble_1d(forecasts)  # (n_ensemble, n_samples)
    alpha = (1 - level) / 2
    lower = np.quantile(ens, alpha, axis=0)
    upper = np.quantile(ens, 1 - alpha, axis=0)
    return float(np.mean((y >= lower) & (y <= upper)))


def energy_score(forecasts: np.ndarray, targets: np.ndarray, beta: float = 1.0) -> float:
    """Energy score — multivariate proper scoring rule.

    ES = E||X - y||^beta - 0.5 * E||X - X'||^beta

    Parameters
    ----------
    beta : float, default 1.0 for 1-dimensional; use 0.5 for higher dimensions.
    """
    y = _target_1d(targets)
    ens = _ensemble_1d(forecasts)  # (n_ensemble, n_samples)
    n_ens, n_samples = ens.shape
    # Pairwise: O(n_ens^2 * n_samples)
    d_xy = np.abs(ens - y[np.newaxis, :]) ** beta  # (n_ens, n_samples)
    e_d_xy = np.mean(d_xy, axis=0)  # (n_samples,)
    if n_ens > 1:
        d_xx = np.zeros(n_samples)
        for i in range(n_ens):
            for j in range(i + 1, n_ens):
                d_xx += np.abs(ens[i] - ens[j]) ** beta
        d_xx *= 2.0 / (n_ens * (n_ens - 1))
    else:
        d_xx = 0.0
    es_per_sample = e_d_xy - 0.5 * d_xx
    return float(np.mean(es_per_sample))


def negative_log_likelihood(forecasts: np.ndarray, targets: np.ndarray, bandwidth: Optional[float] = None) -> float:
    """Negative log-likelihood via kernel density estimation of the ensemble.

    Uses Gaussian KDE with Silverman's rule (or given bandwidth).
    """
    y = _target_1d(targets)
    ens = _ensemble_1d(forecasts)  # (n_ensemble, n_samples)
    n_ens, n_samples = ens.shape
    if bandwidth is None:
        sigma = np.std(ens)
        bandwidth = 1.06 * sigma * n_ens ** (-0.2) if sigma > 0 else 0.1
    nll = 0.0
    for t in range(n_samples):
        kernels = np.exp(-0.5 * ((ens[:, t] - y[t]) / bandwidth) ** 2)
        density = np.mean(kernels) / (bandwidth * np.sqrt(2 * np.pi))
        nll -= np.log(max(density, 1e-15))
    return float(nll / n_samples)