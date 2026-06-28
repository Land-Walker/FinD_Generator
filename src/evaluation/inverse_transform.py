"""
inverse_transform.py — canonical evaluation space builder.

Maps generated target_pca samples back to denoised-close log returns:
  PCA^{-1} → StandardScaler^{-1} → denoised close → log returns.

The reconstruction is lossy (PCA at 95 % variance discards ~5 %). The error
is measured and reported honestly. All model samples and baselines are then
compared in this single canonical space.
"""

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from typing import Tuple


TARGET_COLS = ["open_den", "high_den", "low_den", "close_den"]
CLOSE_INDEX = 3  # "close_den" position in TARGET_COLS


def target_pca_to_log_returns(
    samples: np.ndarray,
    pca: PCA,
    scaler: StandardScaler,
) -> np.ndarray:
    """Inverse-transform PCA samples to denoised-close log returns.

    Parameters
    ----------
    samples : np.ndarray, shape (..., n_components)
        PCA-transformed target samples.  If the last-but-one axis represents
        time (e.g. (..., horizon, n_components)), log-return differencing is
        applied ALONG that axis, reducing its length by 1.
    pca : sklearn.decomposition.PCA
        Fitted PCA on the train-split denoised OHLC.
    scaler : sklearn.preprocessing.StandardScaler
        Fitted StandardScaler on the train-split denoised OHLC.

    Returns
    -------
    log_returns : np.ndarray
        For input shape (..., horizon, n_components) the output is
        (..., horizon-1).  For 2-d input (n_points, n_components) the
        output is 1-d with length n_points-1.
    """
    orig_shape = samples.shape
    n_components = samples.shape[-1]
    if samples.ndim == 2:
        flat = samples
    else:
        flat = samples.reshape(-1, n_components)
    denoised_scaled = pca.inverse_transform(flat)
    denoised_ohlc = scaler.inverse_transform(denoised_scaled)  # (n_total, 4)
    close = denoised_ohlc[:, CLOSE_INDEX]                       # (n_total,)
    if samples.ndim > 2:
        close = close.reshape(orig_shape[:-1])                  # (..., horizon)
        log_returns = np.log(close[..., 1:] / close[..., :-1])  # (..., horizon-1)
    else:
        log_returns = np.log(close[1:] / close[:-1])
    return log_returns


def pca_to_denoised_ohlc(
    samples: np.ndarray,
    pca: PCA,
    scaler: StandardScaler,
) -> np.ndarray:
    """Inverse-transform PCA samples to denoised OHLC levels (close, high, low, open).

    Returns shape (..., 4) — same order as TARGET_COLS.
    """
    flat = samples.reshape(-1, samples.shape[-1]) if samples.ndim > 2 else samples
    denoised = scaler.inverse_transform(pca.inverse_transform(flat))
    if samples.ndim > 2:
        denoised = denoised.reshape(samples.shape[:-1] + (4,))
    return denoised


def reconstruction_error(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> Tuple[float, float]:
    """Report PCA reconstruction error.

    Parameters
    ----------
    original : np.ndarray, shape (n, 4)
        Original denoised OHLC features.
    reconstructed : np.ndarray, shape (n, 4)
        PCA-reconstructed (round-tripped) denoised OHLC features.

    Returns
    -------
    rmse, mape : float
        Root-mean-square reconstruction error and mean absolute percentage error.
    """
    rmse = float(np.sqrt(np.mean((original - reconstructed) ** 2)))
    mape = float(np.mean(np.abs((original - reconstructed) / (np.abs(original) + 1e-10))) * 100)
    return rmse, mape