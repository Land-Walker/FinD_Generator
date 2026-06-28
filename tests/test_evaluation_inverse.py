"""Unit tests for src/evaluation/inverse_transform.py."""
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.evaluation.inverse_transform import (
    target_pca_to_log_returns,
    pca_to_denoised_ohlc,
    reconstruction_error,
    TARGET_COLS,
    CLOSE_INDEX,
)


def make_synthetic_ohlc(n_samples: int = 200) -> np.ndarray:
    """Synthetic OHLC data with realistic structure."""
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n_samples))
    high = close + np.abs(rng.normal(0, 0.5, n_samples))
    low = close - np.abs(rng.normal(0, 0.5, n_samples))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return np.column_stack([open_, high, low, close])


def test_target_pca_to_log_returns_shape():
    """Inverse transform produces correct output shape."""
    ohlc = make_synthetic_ohlc(200)
    scaler = StandardScaler().fit(ohlc)
    scaled = scaler.transform(ohlc)
    pca = PCA(n_components=0.95).fit(scaled)
    pca_out = pca.transform(scaled)

    n_components = pca_out.shape[1]
    samples = np.random.default_rng(0).normal(0, 1, (50, n_components))
    log_rets = target_pca_to_log_returns(samples, pca, scaler)
    assert log_rets.ndim == 1
    assert len(log_rets) == 49  # 50 samples → 49 returns


def test_pca_to_denoised_ohlc_shape():
    """Inverse PCA returns 4-column denoised OHLC."""
    ohlc = make_synthetic_ohlc(200)
    scaler = StandardScaler().fit(ohlc)
    scaled = scaler.transform(ohlc)
    pca = PCA(n_components=0.95).fit(scaled)
    pca_out = pca.transform(scaled)

    n_components = pca_out.shape[1]
    samples = np.random.default_rng(0).normal(0, 1, (50, n_components))
    result = pca_to_denoised_ohlc(samples, pca, scaler)
    assert result.shape == (50, 4)
    # Check order: open, high, low, close
    # close should be non-negative
    assert np.all(result[:, 3] > 0)


def test_reconstruction_error_is_positive():
    """Reconstruction error is measured correctly."""
    ohlc = make_synthetic_ohlc(200)
    scaler = StandardScaler().fit(ohlc)
    scaled = scaler.transform(ohlc)
    pca = PCA(n_components=0.95).fit(scaled)

    reconstructed = scaler.inverse_transform(
        pca.inverse_transform(pca.transform(scaled))
    )
    rmse, mape = reconstruction_error(ohlc, reconstructed)
    assert rmse > 0
    assert mape > 0
    assert rmse < 10.0  # should be reasonable


def test_roundtrip_identity_low_error():
    """Round-trip preserves close well enough for evaluation."""
    ohlc = make_synthetic_ohlc(500)
    scaler = StandardScaler().fit(ohlc)
    scaled = scaler.transform(ohlc)
    pca = PCA(n_components=0.99).fit(scaled)
    # High-variance PCA → near-identity roundtrip
    pca_out = pca.transform(scaled)
    reconstructed = scaler.inverse_transform(pca.inverse_transform(pca_out))
    close_orig = ohlc[:, CLOSE_INDEX]
    close_recon = reconstructed[:, CLOSE_INDEX]
    rmse = float(np.sqrt(np.mean((close_orig - close_recon) ** 2)))
    assert rmse < 0.5  # near-perfect with 99% variance retained


def test_reconstruction_error_report_on_real_data():
    """Compute and report the reconstruction error on the full train set.
    This is an informational test that outputs the actual error number.
    """
    import sys
    sys.path.insert(0, '.')
    from run import _load_local_data
    from src.preprocessor.data_loader import TimeGradDataModule

    raw = _load_local_data()
    dm = TimeGradDataModule(data_dict=raw, device='cpu')
    dm.preprocess_and_split()

    scaler = dm.scalers.get('target_scaler')
    pca = dm.pcas.get('target_pca')
    if scaler is None or pca is None:
        pytest.skip("Scaler/PCA not fitted")
    train = dm.train_df
    target_cols = [c for c in train.columns if c.endswith('_den')]
    ohlc_den = train[target_cols].values
    scaled = scaler.transform(ohlc_den)
    pca_out = pca.transform(scaled)
    reconstructed = scaler.inverse_transform(pca.inverse_transform(pca_out))

    rmse, mape = reconstruction_error(ohlc_den, reconstructed)
    var_explained = np.sum(pca.explained_variance_ratio_)
    print(f"\n  PCA components: {pca.n_components_} (variance: {var_explained:.4f})")
    print(f"  Reconstruction RMSE: {rmse:.6f}")
    print(f"  Reconstruction MAPE: {mape:.2f}%")
    assert rmse < 100.0  # sanity bound