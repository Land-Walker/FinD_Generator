"""Unit tests for the causal rolling-window wavelet denoiser (Phase 1.1)."""
import numpy as np
import pandas as pd
import pytest

from src.preprocessor.data_loader import _denoise_window, wavelet_denoise_series

WINDOW = 64


def _series(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100.0, index=idx)


def test_warmup_is_nan_then_finite():
    s = _series()
    out = wavelet_denoise_series(s, window=WINDOW)
    assert out.iloc[: WINDOW - 1].isna().all(), "warmup positions must be NaN"
    assert out.iloc[WINDOW - 1 :].notna().all(), "post-warmup positions must be finite"
    assert (out.index == s.index).all()


def test_causality_future_changes_do_not_affect_past():
    s = _series()
    out_full = wavelet_denoise_series(s, window=WINDOW)

    t = 250
    s_mod = s.copy()
    s_mod.iloc[t + 1 :] += 1000.0  # violently change the future
    out_mod = wavelet_denoise_series(s_mod, window=WINDOW)

    past_full = out_full.iloc[WINDOW - 1 : t + 1].to_numpy()
    past_mod = out_mod.iloc[WINDOW - 1 : t + 1].to_numpy()
    np.testing.assert_array_equal(past_full, past_mod)


def test_matches_single_window_kernel():
    s = _series()
    out = wavelet_denoise_series(s, window=WINDOW)
    t = 300
    w = s.to_numpy()[t - WINDOW + 1 : t + 1].astype(float)
    expected = _denoise_window(w, wavelet="db4", level=3)
    assert out.iloc[t] == pytest.approx(expected, rel=0, abs=0)


def test_constant_series_reconstructed():
    idx = pd.bdate_range("2010-01-01", periods=200)
    s = pd.Series(42.0, index=idx)
    out = wavelet_denoise_series(s, window=WINDOW)
    np.testing.assert_allclose(out.iloc[WINDOW - 1 :], 42.0, rtol=1e-10)


def test_window_too_small_raises():
    s = _series()
    with pytest.raises(ValueError):
        wavelet_denoise_series(s, level=3, window=16)  # min is 2**3*4 = 32


def test_leading_nans_propagate_not_backfilled():
    s = _series()
    s.iloc[:10] = np.nan  # ffill cannot resolve leading NaNs
    out = wavelet_denoise_series(s, window=WINDOW)
    # windows touching the unresolved leading NaNs must be NaN, never backfilled
    assert out.iloc[: 10 + WINDOW - 1].isna().all()
    assert out.iloc[10 + WINDOW - 1 :].notna().all()
