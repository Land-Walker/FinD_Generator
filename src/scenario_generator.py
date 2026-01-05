"""Scenario feature override utility for conditional TimeGrad models.

The generator optionally overrides pre-computed regime one-hot features at
inference time for stress testing or counterfactual analysis. It never
modifies targets, recomputes regimes, or mutates the input DataFrame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class ScenarioSpec:
    """Optional scenario definition for regime overrides.

    Attributes
    ----------
    market_regime, vol_regime, macro_regime
        Optional regime names (without prefix) to force during the window.
    start_t, duration
        Start index and length of the scenario window (required when used).
    transition
        "hard" for abrupt switch, "soft" for smoothed interpolation.
    """

    market_regime: Optional[str] = None
    vol_regime: Optional[str] = None
    macro_regime: Optional[str] = None
    start_t: int = 0
    duration: int = 0
    transition: str = "hard"


def _extract_prefix_columns(cond_df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in cond_df.columns if c.startswith(f"{prefix}_")]
    if not cols:
        raise ValueError(f"No columns found with prefix '{prefix}_' in conditioning DataFrame")
    return cols


def _clip_window(start: int, duration: int, horizon: int, length: int) -> Optional[range]:
    start_idx = max(0, start)
    end_idx = min(start_idx + duration, horizon, length)
    if end_idx <= start_idx:
        return None
    return range(start_idx, end_idx)


class ScenarioFeatureGenerator:
    """Applies optional regime overrides to conditioning features.

    The generator only edits regime one-hot columns and leaves all other
    conditioning signals untouched. Passing ``scenario=None`` is a strict
    no-op that returns a copy of the input DataFrame.
    """

    def __init__(self, regime_prefixes: Dict[str, str], smoothing_window: int = 5):
        if smoothing_window < 1:
            raise ValueError("smoothing_window must be >= 1")
        self.regime_prefixes = regime_prefixes
        self.smoothing_window = smoothing_window

    def _coerce_spec(self, scenario: Union[ScenarioSpec, Dict[str, object]]) -> ScenarioSpec:
        if isinstance(scenario, ScenarioSpec):
            spec = scenario
        elif isinstance(scenario, dict):
            spec = ScenarioSpec(**scenario)
        else:
            raise TypeError("scenario must be a ScenarioSpec, dict, or None")

        if spec.duration <= 0:
            raise ValueError("scenario.duration must be positive")
        if spec.start_t < 0:
            raise ValueError("scenario.start_t must be non-negative")
        if spec.transition not in {"hard", "soft"}:
            raise ValueError("scenario.transition must be 'hard' or 'soft'")
        return spec

    def _apply_soft_transition(
        self,
        base_window: np.ndarray,
        target_idx: int,
        ramp: int,
    ) -> np.ndarray:
        length = base_window.shape[0]
        ramp = max(1, min(ramp, length))
        blended = base_window.copy()

        for pos in range(length):
            left = min(1.0, (pos + 1) / ramp)
            right = min(1.0, (length - pos) / ramp)
            weight = min(left, right)
            row = base_window[pos] * (1.0 - weight)
            row[target_idx] += weight
            total = row.sum()
            blended[pos] = row if total <= 0 else row / total
        return blended

    def _apply_group_override(
        self,
        df: pd.DataFrame,
        columns: List[str],
        target_col: str,
        window: range,
        transition: str,
    ) -> None:
        target_idx = columns.index(target_col)
        base = df.loc[df.index[list(window)], columns].to_numpy(copy=True)

        if transition == "hard":
            base[:] = 0.0
            base[:, target_idx] = 1.0
        else:
            base = self._apply_soft_transition(base, target_idx, self.smoothing_window)

        df.loc[df.index[list(window)], columns] = base

    def apply_scenario(
        self,
        cond_df: pd.DataFrame,
        scenario: Optional[Union[ScenarioSpec, Dict[str, object]]],
        horizon: int,
    ) -> pd.DataFrame:
        """Optionally override regime one-hot features for stress testing.

        Parameters
        ----------
        cond_df
            Conditioning DataFrame containing regime one-hot columns.
        scenario
            Optional ScenarioSpec or dict. If None, returns ``cond_df.copy()``.
        horizon
            Forecast horizon to clip the scenario window to ``[0, horizon)``.
        """

        if scenario is None:
            return cond_df.copy()

        spec = self._coerce_spec(scenario)
        window = _clip_window(spec.start_t, spec.duration, horizon, len(cond_df))
        if window is None:
            return cond_df.copy()

        out_df = cond_df.copy()
        for group_key, prefix in self.regime_prefixes.items():
            attr_name = f"{group_key}_regime"
            target_value = getattr(spec, attr_name, None)
            if not target_value:
                continue

            cols = _extract_prefix_columns(out_df, prefix)
            target_col = f"{prefix}_{target_value}"
            if target_col not in cols:
                # Allow stress/counterfactual regimes that were not present in the
                # original dataset by introducing a zero-initialized column. This
                # preserves non-regime features and avoids mutating the input.
                out_df[target_col] = 0.0
                cols = sorted([*cols, target_col])

            self._apply_group_override(
                out_df,
                columns=cols,
                target_col=target_col,
                window=window,
                transition=spec.transition,
            )

        return out_df