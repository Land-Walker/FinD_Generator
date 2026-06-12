"""Phase 1.4 (MASTER_SPEC): no-look-ahead audit of EVERY engineered feature.

For 100 random timestamps t drawn from the val and test splits, every
feature is recomputed INDEPENDENTLY using only raw data with date <= t
(truncated inputs) and compared to the pipeline's value at t.

If any pipeline feature used information from after t, the recomputed value
would differ and the assertion would fail. The full feature list covered
here is enumerated in docs/data_integrity.md and must be kept in sync.

Implementation notes:
- The recomputations below mirror the pipeline's documented alignment rules
  (business-day grid, ffill-from-past, month-end mapping for monthly/
  quarterly data) but are written independently of the pipeline functions,
  except for the single-window wavelet kernel `_denoise_window`, which is
  fed ONLY truncated data.
- Scaled/PCA features are recomputed by applying the train-fitted
  scalers/PCA (past data relative to any val/test t) to the independently
  recomputed raw features.
"""
import numpy as np
import pandas as pd
import pytest

from run import _load_local_data
from src import config
from src.preprocessor.data_loader import TimeGradDataModule, _denoise_window

N_TIMESTAMPS = 100
RNG_SEED = 0
WINDOW = config.WAVELET_WINDOW
ROLL_WINDOW = 30  # market regime roll-stat window (pipeline default)

RTOL, ATOL = 1e-7, 1e-10


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pipeline():
    raw = _load_local_data()
    base_keys = ["target", "market", "daily_macro", "monthly_macro", "quarterly_macro"]

    def indexed(name):
        df = raw[name].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        return df.set_index("Date").sort_index()

    # build the truncation copies BEFORE the DataModule mutates data_dict
    # (it inserts already-indexed *_aligned frames into the same dict)
    rawi = {k: indexed(k) for k in base_keys}

    dm = TimeGradDataModule(data_dict=raw, device="cpu")
    dm.preprocess_and_split()
    return dm, rawi


@pytest.fixture(scope="module")
def sampled_timestamps(pipeline):
    dm, _ = pipeline
    rng = np.random.default_rng(RNG_SEED)
    val_idx = dm.val_df.index
    test_idx = dm.test_df.index
    n_val = N_TIMESTAMPS // 2
    ts_val = list(pd.DatetimeIndex(rng.choice(val_idx, size=n_val, replace=False)))
    ts_test = list(pd.DatetimeIndex(rng.choice(test_idx, size=N_TIMESTAMPS - n_val, replace=False)))
    return ts_val + ts_test


# --------------------------------------------------------------------------
# independent recomputation helpers (all take data truncated at t)
# --------------------------------------------------------------------------
def _last_at_or_before(series: pd.Series, t) -> float:
    s = series.loc[:t].dropna()
    assert len(s) > 0, f"no observation at or before {t}"
    return float(s.iloc[-1])


def _recompute_target_den(rawi, t, col) -> float:
    s = rawi["target"][col].loc[:t].ffill()
    w = s.to_numpy(dtype=float)[-WINDOW:]
    assert len(w) == WINDOW and not np.isnan(w).any()
    return _denoise_window(w, wavelet=config.WAVELET, level=config.WAVELET_LEVEL)


def _recompute_market_ret(rawi, t, col) -> float:
    s = np.log(rawi["market"][col]).diff().loc[:t].dropna()
    return float(s.iloc[-1])


def _monthly_transforms(monthly: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=monthly.index)
    out["cpi_mom"] = monthly["cpi"].pct_change().fillna(0)
    out["unemployment_detrend"] = (
        monthly["unemployment"] - monthly["unemployment"].rolling(12, min_periods=1).mean()
    ).fillna(0)
    out["interest_rate_diff"] = monthly["interest_rate"].diff().fillna(0)
    out["trade_balance_seasdiff"] = (
        monthly["trade_balance"] - monthly["trade_balance"].shift(12)
    ).fillna(0)
    out.index = out.index.to_period("M").to_timestamp("M")  # month-end mapping (merge step)
    return out


def _quarterly_transforms(quarterly: pd.DataFrame, t) -> pd.DataFrame:
    q = quarterly.resample("MS").ffill()  # alignment step (no suffix, post-fix)
    # Extend the monthly grid up to the month containing t: months with no new
    # quarterly publication carry the previous value (pure ffill of data <= t,
    # still causal) — the pipeline materializes those rows, so must we.
    full_ms = pd.date_range(q.index[0], pd.Timestamp(t).normalize().replace(day=1), freq="MS")
    q = q.reindex(full_ms).ffill()
    out = pd.DataFrame(index=q.index)
    out["gdp_yoy"] = np.log(q["gdp"] / q["gdp"].shift(1))
    out["gov_fiscal_balance_to_gdp"] = q["gov_fiscal_balance"] / q["gdp"].replace({0: np.nan})
    for col in ["gov_debt", "tax_receipts", "gov_spending"]:
        out[col] = q[col]
    out.index = out.index.to_period("M").to_timestamp("M")
    return out


def _recompute_market_close_series(rawi, merged_index, t) -> pd.Series:
    """market_close on the merged grid, using only raw data <= t."""
    grid = merged_index[merged_index <= t]
    return rawi["market"]["close"].loc[:t].reindex(grid).ffill()


# --------------------------------------------------------------------------
# the audit
# --------------------------------------------------------------------------
def test_every_feature_is_causal(pipeline, sampled_timestamps):
    dm, rawi = pipeline
    merged = dm.merged_raw
    thresholds = dm.regime_thresholds

    failures = []

    def check(t, name, recomputed, pipeline_value):
        if not np.isclose(recomputed, pipeline_value, rtol=RTOL, atol=ATOL, equal_nan=True):
            failures.append(f"{t.date()} {name}: recomputed={recomputed!r} pipeline={pipeline_value!r}")

    for t in sampled_timestamps:
        row = merged.loc[t]
        in_val = t in dm.val_df.index
        trans_row = (dm.val_transformed_full if in_val else dm.test_transformed_full).loc[t]

        # --- target block: causal wavelet + volume passthrough -------------
        target_den = {}
        for c in ["open", "high", "low", "close"]:
            target_den[f"{c}_den"] = _recompute_target_den(rawi, t, c)
            check(t, f"{c}_den", target_den[f"{c}_den"], row[f"{c}_den"])
        check(t, "target_volume_raw", _last_at_or_before(rawi["target"]["volume"], t), row["target_volume_raw"])

        # --- market block: log returns + log1p volume ----------------------
        market_ret = {}
        for c in ["open", "high", "low", "close"]:
            market_ret[f"{c}_ret"] = _recompute_market_ret(rawi, t, c)
            check(t, f"{c}_ret", market_ret[f"{c}_ret"], row[f"{c}_ret"])
        mkt_vol = np.log1p(_last_at_or_before(rawi["market"]["volume"], t))
        check(t, "market_volume_raw", mkt_vol, row["market_volume_raw"])

        # --- daily macro ----------------------------------------------------
        vix = _last_at_or_before(rawi["daily_macro"]["vix"], t)
        ycurve = _last_at_or_before(rawi["daily_macro"]["yield_curve"], t)
        check(t, "vix_daily", vix, row["vix_daily"])
        check(t, "yield_curve_daily", ycurve, row["yield_curve_daily"])

        # --- monthly macro (truncated raw -> transforms -> month-end map) ---
        monthly_trunc = rawi["monthly_macro"].loc[:t]
        mt = _monthly_transforms(monthly_trunc)
        monthly_vals = {}
        for c in ["cpi_mom", "unemployment_detrend", "interest_rate_diff", "trade_balance_seasdiff"]:
            monthly_vals[c] = _last_at_or_before(mt[c], t)
            check(t, c, monthly_vals[c], row[c])

        # --- quarterly macro -------------------------------------------------
        quarterly_trunc = rawi["quarterly_macro"].loc[:t]
        qt = _quarterly_transforms(quarterly_trunc, t)
        quarterly_vals = {}
        for c in ["gdp_yoy", "gov_fiscal_balance_to_gdp", "gov_debt", "tax_receipts", "gov_spending"]:
            quarterly_vals[c] = _last_at_or_before(qt[c], t)
            check(t, c, quarterly_vals[c], row[c])

        # --- calendar features (pure functions of t) ------------------------
        check(t, "day_of_week", t.dayofweek, row["day_of_week"])
        check(t, "month", t.month, row["month"])
        check(t, "quarter", t.quarter, row["quarter"])
        check(t, "year", t.year, row["year"])
        # the pipeline index carries freq='B', so is_month_end/is_quarter_end
        # mark the last BUSINESS day of the period — a pure function of the
        # date (causal); replicate that convention here
        next_b = t + pd.offsets.BDay(1)
        check(t, "is_month_end", int(next_b.month != t.month), row["is_month_end"])
        check(t, "is_quarter_end", int(next_b.quarter != t.quarter), row["is_quarter_end"])
        check(t, "month_sin", np.sin(2 * np.pi * t.month / 12), row["month_sin"])
        check(t, "month_cos", np.cos(2 * np.pi * t.month / 12), row["month_cos"])
        check(t, "dow_sin", np.sin(2 * np.pi * t.dayofweek / 7), row["dow_sin"])
        check(t, "dow_cos", np.cos(2 * np.pi * t.dayofweek / 7), row["dow_cos"])
        check(t, "quarter_sin", np.sin(2 * np.pi * t.quarter / 4), row["quarter_sin"])
        check(t, "quarter_cos", np.cos(2 * np.pi * t.quarter / 4), row["quarter_cos"])

        # --- market_close + regime labels (frozen train-only threshold) -----
        mc = _recompute_market_close_series(rawi, merged.index, t)
        check(t, "market_close", float(mc.iloc[-1]), row["market_close"])

        roll_return = float(np.log(mc).diff(ROLL_WINDOW).iloc[-1])
        roll_vol = float(mc.pct_change().rolling(ROLL_WINDOW, min_periods=1).std().iloc[-1])
        vm = thresholds["roll_vol_median_train"]
        if roll_return > 0.02 and roll_vol < vm:
            market_regime = "bull"
        elif roll_return < -0.02 and roll_vol > vm:
            market_regime = "bear"
        else:
            market_regime = "sideways"
        for reg in ["bear", "bull", "sideways"]:
            check(t, f"market_regime_{reg}", float(market_regime == reg), float(row[f"market_regime_{reg}"]))

        vol_regime = "high_vol" if vix > 20 else "normal_vol"
        for reg in ["high_vol", "normal_vol"]:
            check(t, f"vol_regime_{reg}", float(vol_regime == reg), float(row[f"vol_regime_{reg}"]))

        infl, gdp = monthly_vals["cpi_mom"], quarterly_vals["gdp_yoy"]
        if infl > 0.03 and gdp < 0.0:
            macro = "stagflation"
        elif infl > 0.03:
            macro = "high_inflation"
        elif gdp < 0.0:
            macro = "recession"
        else:
            macro = "expansion"
        for reg in ["expansion", "high_inflation", "normal", "recession", "stagflation"]:
            check(t, f"macro_regime_{reg}", float(macro == reg), float(row[f"macro_regime_{reg}"]))

        # --- scaled / PCA features (train-fitted transforms on recomputed raw)
        tvec = np.array([[target_den[c] for c in ["open_den", "high_den", "low_den", "close_den"]]])
        t_pca = dm.pcas["target_pca"].transform(dm.scalers["target_scaler"].transform(tvec))
        check(t, "target_pca_1", float(t_pca[0, 0]), trans_row["target_pca_1"])

        mvec = np.array([[market_ret[c] for c in ["open_ret", "high_ret", "low_ret", "close_ret"]]])
        m_pca = dm.pcas["market_pca"].transform(dm.scalers["market_scaler"].transform(mvec))
        for i in range(m_pca.shape[1]):
            check(t, f"market_pca_{i+1}", float(m_pca[0, i]), trans_row[f"market_pca_{i+1}"])

        dvec = np.array([[vix, ycurve]])
        d_scaled = dm.scalers["daily_macro_scaler"].transform(dvec)
        check(t, "daily_vix_daily_scaled", float(d_scaled[0, 0]), trans_row["daily_vix_daily_scaled"])
        check(t, "daily_yield_curve_daily_scaled", float(d_scaled[0, 1]), trans_row["daily_yield_curve_daily_scaled"])

        v_scaled = dm.scalers["volume_scaler"].transform([[mkt_vol]])
        check(t, "volume_scaled", float(v_scaled[0, 0]), trans_row["volume_scaled"])

        mmvec = np.array([[monthly_vals[c] for c in ["cpi_mom", "unemployment_detrend", "interest_rate_diff", "trade_balance_seasdiff"]]])
        mm_pca = dm.pcas["monthly_pca"].transform(dm.scalers["monthly_macro_scaler"].transform(mmvec))
        for i in range(mm_pca.shape[1]):
            check(t, f"monthly_pca_{i+1}", float(mm_pca[0, i]), trans_row[f"monthly_pca_{i+1}"])

        qqvec = np.array([[quarterly_vals[c] for c in ["gdp_yoy", "gov_fiscal_balance_to_gdp", "gov_debt", "tax_receipts", "gov_spending"]]])
        qq_pca = dm.pcas["quarterly_pca"].transform(dm.scalers["quarterly_macro_scaler"].transform(qqvec))
        for i in range(qq_pca.shape[1]):
            check(t, f"quarterly_pca_{i+1}", float(qq_pca[0, i]), trans_row[f"quarterly_pca_{i+1}"])

    assert not failures, (
        f"{len(failures)} causality violations detected:\n" + "\n".join(failures[:50])
    )


def test_threshold_is_train_only(pipeline):
    dm, _ = pipeline
    th = dm.regime_thresholds
    train_end = th["train_end_row"]
    expected = float(
        dm.merged_unlabeled.iloc[:train_end]["roll_vol"].median()
    )
    assert th["roll_vol_median_train"] == expected
    # and it must differ from the leaky full-sample median in this dataset
    assert th["roll_vol_median_train"] != th["roll_vol_median_full_sample_FOR_AUDIT_ONLY"]


def test_no_bfill_in_pipeline_source():
    """Source-level guard (W2.2): no backward-fill anywhere in the data pipeline."""
    import inspect
    import src.preprocessor.data_loader as dl

    src = inspect.getsource(dl)
    code_lines = [
        l for l in src.splitlines() if not l.strip().startswith("#") and "bfill" in l
    ]
    # allow mentions inside docstrings/comments describing the legacy bug;
    # forbid actual calls
    offending = [l for l in code_lines if ".bfill(" in l or "method='bfill'" in l or 'method="bfill"' in l]
    assert not offending, f"bfill found in pipeline source: {offending}"
