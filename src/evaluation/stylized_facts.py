"""
stylized_facts.py — financial stylized-fact metrics for evaluating scenario quality.

Compares generated returns against real (test-set) returns on standard facts.
All functions accept 1-d return series as np.ndarray.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple


def kurtosis(returns: np.ndarray) -> float:
    """Excess kurtosis (Fisher). Normal = 0."""
    return float(stats.kurtosis(returns, fisher=True))


def skewness(returns: np.ndarray) -> float:
    """Sample skewness."""
    return float(stats.skew(returns))


def tail_index_hill(returns: np.ndarray, k_frac: float = 0.05) -> float:
    """Hill estimator of the tail index for the RIGHT tail.

    Uses the top k_frac fraction of absolute returns (heavy-tail focus).
    Returns the Hill tail index α. Larger α = thinner tails.
    """
    abs_r = np.abs(returns)
    n = len(abs_r)
    k = max(int(n * k_frac), 10)
    sorted_tail = np.sort(abs_r)[-k:]
    threshold = sorted_tail[0]
    exceedances = sorted_tail - threshold
    if np.all(exceedances <= 0):
        return float('inf')
    hill_alpha = 1.0 / (np.mean(np.log(exceedances / threshold + 1e-15)) + 1e-15)
    # Hill estimator: α̂ = 1 / ( (1/k) * Σ log(X_i / X_{(k)}) )
    log_exceed = np.log(sorted_tail / threshold)
    log_exceed = log_exceed[log_exceed > 0]
    if len(log_exceed) < 2:
        return float('inf')
    hill_alpha = k / np.sum(log_exceed)
    return float(hill_alpha)


def acf_returns(returns: np.ndarray, lags: int = 20) -> np.ndarray:
    """Autocorrelation of returns (should be ~0 for efficient markets)."""
    r = returns - np.mean(returns)
    n = len(r)
    result = np.zeros(lags)
    var = np.var(r) * n
    if var == 0:
        return result
    for lag in range(1, lags + 1):
        result[lag - 1] = np.sum(r[lag:] * r[:-lag]) / var
    return result


def acf_abs_returns(returns: np.ndarray, lags: int = 20) -> np.ndarray:
    """Autocorrelation of absolute returns (volatility clustering)."""
    return acf_returns(np.abs(returns), lags)


def leverage_effect(returns: np.ndarray, lags: int = 20) -> np.ndarray:
    """Leverage effect: corr(r_t, |r|_{t+k}) for k = 1..lags.
    Negative values indicate that down moves predict higher future volatility.
    """
    r = returns - np.mean(returns)
    abs_r = np.abs(returns) - np.mean(np.abs(returns))
    n = len(r)
    result = np.zeros(lags)
    for lag in range(1, lags + 1):
        if lag < n:
            num = np.sum(r[:n - lag] * abs_r[lag:])
            den = np.sqrt(np.sum(r[:n - lag] ** 2) * np.sum(abs_r[lag:] ** 2))
            result[lag - 1] = num / den if den > 0 else 0.0
    return result


def _drawdown_single_path(returns: np.ndarray) -> Tuple[float, float, int]:
    """Compute drawdown stats for a single price path.

    Pre-pends starting capital 1.0 so that an initial drop from par is captured.
    """
    cumulative = np.cumprod(1 + returns)
    prepended = np.concatenate([[1.0], cumulative])
    peak = np.maximum.accumulate(prepended)
    drawdown = (cumulative - peak[1:]) / peak[1:]
    max_dd = float(np.min(drawdown))
    if np.any(drawdown < 0):
        mean_dd = float(np.mean(drawdown[drawdown < 0]))
    else:
        mean_dd = 0.0
    in_dd = drawdown < 0
    max_duration = 0
    current = 0
    for v in in_dd:
        if v:
            current += 1
            max_duration = max(max_duration, current)
        else:
            current = 0
    return max_dd, mean_dd, max_duration


def drawdown_distribution(returns: np.ndarray) -> Dict[str, float]:
    """Drawdown statistics: max, mean, and duration statistics.

    Parameters
    ----------
    returns : np.ndarray, shape (n_steps,) or (n_paths, n_steps)
        If 1-d: single continuous price path.
        If 2-d: independent paths (rows); stats are aggregated across paths
        returning the worst max_drawdown, mean of mean drawdowns, and
        longest drawdown duration.

    Returns dict with 'max_drawdown', 'mean_drawdown', 'max_drawdown_duration'.
    """
    if returns.ndim == 2:
        max_dds = []
        mean_dds = []
        max_durations = []
        for r in returns:
            max_dd, mean_dd, max_dur = _drawdown_single_path(r)
            max_dds.append(max_dd)
            if mean_dd < 0:
                mean_dds.append(mean_dd)
            max_durations.append(max_dur)
        return {
            'max_drawdown': float(np.min(max_dds)) if max_dds else 0.0,
            'mean_drawdown': float(np.mean(mean_dds)) if mean_dds else 0.0,
            'max_drawdown_duration': int(np.max(max_durations)) if max_durations else 0,
        }
    max_dd, mean_dd, max_duration = _drawdown_single_path(returns)
    return {
        'max_drawdown': max_dd,
        'mean_drawdown': mean_dd,
        'max_drawdown_duration': int(max_duration),
    }


def realized_vol_distribution(returns: np.ndarray, window: int = 21) -> np.ndarray:
    """Rolling realized volatility (annualised).

    Returns the full series of rolling vol values.
    """
    rv = np.array([np.nan] * len(returns))
    for i in range(window - 1, len(returns)):
        rv[i] = np.std(returns[i - window + 1:i + 1]) * np.sqrt(252)
    return rv[~np.isnan(rv)]


def var_es(returns: np.ndarray, alpha: float) -> Tuple[float, float]:
    """Value at Risk and Expected Shortfall at confidence level alpha.

    Parameters
    ----------
    alpha : float, in (0, 1). e.g. 0.95 for 95 % VaR.

    Returns
    -------
    var, es : float
    """
    var = float(np.quantile(returns, 1 - alpha))
    es = float(np.mean(returns[returns <= var]))
    return var, es


def all_stylized_facts(returns: np.ndarray, lags: int = 20) -> Dict[str, float]:
    """Compute all stylized facts for a return series.

    Returns a flat dict with scalar metrics.
    """
    facts = {}
    facts['kurtosis'] = kurtosis(returns)
    facts['skewness'] = skewness(returns)
    facts['tail_index_hill'] = tail_index_hill(returns)
    acf_r = acf_returns(returns, lags)
    facts['acf_returns_lag1'] = float(acf_r[0])
    facts['acf_returns_maxabs'] = float(np.max(np.abs(acf_r)))
    acf_ar = acf_abs_returns(returns, lags)
    facts['acf_abs_returns_lag1'] = float(acf_ar[0])
    # Mean ACF of abs returns over first 10 lags (vol clustering summary)
    facts['acf_abs_returns_mean10'] = float(np.mean(acf_ar[:10]))
    lev = leverage_effect(returns, lags)
    facts['leverage_lag1'] = float(lev[0])
    facts['leverage_min'] = float(np.min(lev))
    dd = drawdown_distribution(returns)
    facts.update({f'drawdown_{k}': v for k, v in dd.items()})
    rv = realized_vol_distribution(returns)
    facts['realized_vol_mean'] = float(np.mean(rv)) if len(rv) > 0 else 0.0
    var95, es95 = var_es(returns, 0.95)
    facts['var_95'] = var95
    facts['es_95'] = es95
    var99, es99 = var_es(returns, 0.99)
    facts['var_99'] = var99
    facts['es_99'] = es99
    return facts