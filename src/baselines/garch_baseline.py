"""
garch_baseline.py — GARCH(1,1) with Student-t innovations using the `arch` package.

Fits on training returns only, then simulates prediction_length steps forward
from test-period history. Produces sample paths in canonical space (log returns).
"""

import numpy as np
from typing import Optional, Tuple
from scipy import stats as sp_stats


class GARCHBaseline:
    """GARCH(1,1) with Student-t innovations.

    Parameters
    ----------
    dist : str
        Innovation distribution: 't' (Student-t) or 'normal'.
    seed : int, optional
        Seed for the RNG used in simulation.
    """

    def __init__(self, dist: str = "t", seed: Optional[int] = None):
        if dist not in ("t", "normal"):
            raise ValueError("dist must be 't' or 'normal'")
        self.dist = dist
        self._rng = np.random.default_rng(seed)
        self._params: Optional[Tuple[float, float, float, float, float]] = None
        self._fitted = False

    def fit(self, train_returns: np.ndarray) -> "GARCHBaseline":
        """Fit GARCH(1,1) on training returns using MLE via the `arch` package.

        Must be called before `generate_samples`.
        """
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError(
                "The `arch` package is required for GARCH baselines. "
                "Install with: pip install arch"
            )
        returns = np.asarray(train_returns, dtype=float) * 100.0
        dist_map = {"t": "t", "normal": "normal"}
        model = arch_model(returns, mean="Zero", vol="GARCH", p=1, q=1, dist=dist_map[self.dist])
        fitted = model.fit(disp="off")
        omega = float(fitted.params["omega"]) / 10000.0
        alpha = float(fitted.params["alpha[1]"])
        beta = float(fitted.params["beta[1]"])
        if self.dist == "t":
            nu = float(fitted.params["nu"])
        else:
            nu = float("inf")
        self._params = (omega, alpha, beta, nu, float(np.var(train_returns)))
        self._fitted = True
        return self

    @property
    def unconditional_variance(self) -> float:
        """Theoretical unconditional variance: omega / (1 - alpha - beta)."""
        if self._params is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        omega, alpha, beta, nu, _ = self._params
        denom = 1.0 - alpha - beta
        if denom <= 0:
            return float("inf")
        return omega / denom

    def generate_samples(
        self,
        n_samples: int,
        n_paths: int,
        horizon: int,
        history_returns: np.ndarray,
    ) -> np.ndarray:
        """Simulate log returns from the fitted GARCH(1,1).

        Each path starts from the most recent residual and variance implied
        by `history_returns`.

        Parameters
        ----------
        n_samples : int
            Ensemble members (first axis).
        n_paths : int
            Number of independent paths (e.g. test windows).
        horizon : int
            Return steps per path.
        history_returns : np.ndarray, shape (n_paths, history_len)
            Per-path history rows used to seed the GARCH variance and
            last innovation.  The last element of each row seeds the
            initial residual.

        Returns
        -------
        samples : np.ndarray, shape (n_samples, n_paths, horizon)
            Simulated log returns.
        """
        if not self._fitted or self._params is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        omega, alpha, beta, nu, ret_var = self._params

        history = np.asarray(history_returns, dtype=float)
        if history.ndim == 1:
            history = history.reshape(1, -1)
        if history.shape[0] != n_paths:
            raise ValueError(
                f"history_returns must have {n_paths} rows, got {history.shape[0]}"
            )

        last_ret = history[:, -1]
        last_var = np.clip(np.var(history, axis=1), omega, None)

        samples = np.empty((n_samples, n_paths, horizon), dtype=np.float64)
        for s in range(n_samples):
            if self.dist == "t":
                z = sp_stats.t.rvs(df=nu, size=(n_paths, horizon), random_state=self._rng)
                scale = np.sqrt((nu - 2) / nu) if nu > 2 else 1.0
            else:
                z = self._rng.normal(size=(n_paths, horizon))
                scale = 1.0
            var = last_var.copy()
            resid = last_ret.copy()
            for t in range(horizon):
                var = omega + alpha * resid ** 2 + beta * var
                resid = np.sqrt(np.maximum(var, 0)) * z[:, t] / scale
                samples[s, :, t] = resid
        return samples