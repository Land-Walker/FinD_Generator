"""
historical_bootstrap.py — i.i.d. resample from training-set log returns.

Produces sample paths in canonical space (denoised-close log returns).
Output shape: (n_samples, n_paths, horizon) — compatible with Phase 2 evaluation
once the canonical-space path is used (no PCA inverse-transform needed for baselines).
"""

import numpy as np
from typing import Optional


def generate_samples(
    train_returns: np.ndarray,
    n_samples: int,
    n_paths: int,
    horizon: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate i.i.d. bootstrap returns by resampling from train_returns.

    Parameters
    ----------
    train_returns : np.ndarray, shape (n_train,)
        1-d array of training-period log returns.
    n_samples : int
        Number of ensemble members (first axis, like diffusion num_samples).
    n_paths : int
        Number of independent paths (e.g. test windows).
    horizon : int
        Number of return steps per path.
    rng : np.random.Generator, optional
        Seeded generator for reproducibility.

    Returns
    -------
    samples : np.ndarray, shape (n_samples, n_paths, horizon)
        Bootstrapped log returns.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_train = len(train_returns)
    if n_train == 0:
        raise ValueError("train_returns must not be empty")
    indices = rng.integers(0, n_train, size=(n_samples, n_paths, horizon))
    return train_returns[indices]