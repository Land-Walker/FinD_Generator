"""
block_bootstrap.py — Stationary (Politis-Romano) block bootstrap of log returns.

Unlike i.i.d. resampling, block bootstrap preserves serial dependence
(volatility clustering) by resampling contiguous blocks.  The expected block
length is configurable; actual block lengths are drawn from a geometric
distribution with mean = block_length.
"""

import numpy as np
from typing import Optional


def _geometric_block_lengths(total: int, mean_len: float, rng: np.random.Generator) -> np.ndarray:
    """Sample block lengths from geometric(p=1/mean_len) truncated to [1, total]."""
    p = 1.0 / mean_len
    lengths = []
    remaining = total
    while remaining > 0:
        bl = rng.geometric(p)
        bl = min(bl, remaining)
        lengths.append(bl)
        remaining -= bl
    return np.array(lengths, dtype=int)


def generate_samples(
    train_returns: np.ndarray,
    n_samples: int,
    n_paths: int,
    horizon: int,
    block_length: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Stationary (Politis-Romano) block bootstrap.

    For each path of length `horizon`, the function draws blocks with
    wrap-around from `train_returns` and concatenates them.

    Parameters
    ----------
    train_returns : np.ndarray, shape (n_train,)
        Training-period log returns.
    n_samples : int
        Ensemble members.
    n_paths : int
        Independent paths.
    horizon : int
        Return steps per path.
    block_length : int
        Expected block length (default 10).
    rng : np.random.Generator, optional

    Returns
    -------
    samples : np.ndarray, shape (n_samples, n_paths, horizon)
    """
    if rng is None:
        rng = np.random.default_rng()
    n_train = len(train_returns)
    if n_train == 0:
        raise ValueError("train_returns must not be empty")
    if block_length < 1:
        raise ValueError("block_length must be >= 1")

    samples = np.empty((n_samples, n_paths, horizon), dtype=train_returns.dtype)
    for s in range(n_samples):
        for p in range(n_paths):
            bls = _geometric_block_lengths(horizon, block_length, rng)
            pos = 0
            for bl in bls:
                start = rng.integers(0, n_train)
                if start + bl <= n_train:
                    block = train_returns[start : start + bl]
                else:
                    block = np.concatenate([
                        train_returns[start:],
                        train_returns[: bl - (n_train - start)],
                    ])
                samples[s, p, pos : pos + bl] = block
                pos += bl
    return samples