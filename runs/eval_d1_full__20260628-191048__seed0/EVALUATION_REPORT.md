# EVALUATION_REPORT — Phase 2

## Canonical Evaluation Space
- All methods evaluated in **denoised-close log returns**.
- PCA 1 components, reconstruction RMSE = 0.3525, MAPE = 0.25%.
- `real_raw_un_denoised` = un-denoised log returns (honest kurtosis/tails reference).

## Forecast Metrics

| Metric | Value |
|--------|-------|
| crps | 0.004331 |
| mae | 0.005979 |
| rmse | 0.007888 |
| quantile_loss_0.05 | 0.001045 |
| quantile_loss_0.1 | 0.001562 |
| quantile_loss_0.5 | 0.002892 |
| quantile_loss_0.9 | 0.001508 |
| quantile_loss_0.95 | 0.001021 |
| coverage_0.5 | 0.363041 |
| coverage_0.8 | 0.622437 |
| coverage_0.9 | 0.725513 |
| coverage_0.95 | 0.792426 |
| pit_ks_stat | 0.184374 |
| pit_ks_pvalue | 0.000000 |
| energy_score | 0.004331 |
| nll | -3.387004 |

## Stylized Facts

| Fact | Real (raw) | Real (denoised) | Generated |
|------|-----------|-----------------|-----------|
| acf_abs_returns_lag1 | 0.1392 | 0.2897 | 0.2720 |
| acf_abs_returns_mean10 | 0.1740 | 0.3340 | 0.1852 |
| acf_returns_lag1 | 0.0177 | 0.4317 | 0.1990 |
| acf_returns_maxabs | 0.0951 | 0.7836 | 0.3001 |
| drawdown_max_drawdown | -0.2622 | -0.6794 | -1.0000 |
| drawdown_max_drawdown_duration | 533 | 2044 | 351009 |
| drawdown_mean_drawdown | -0.0894 | -0.2809 | -0.9976 |
| es_95 | -0.0243 | -0.0157 | -0.0212 |
| es_99 | -0.0359 | -0.0248 | -0.0309 |
| kurtosis | 2.1744 | 3.7880 | 3.3813 |
| leverage_lag1 | -0.0556 | -0.0736 | -0.1310 |
| leverage_min | -0.1373 | -0.1592 | -0.1933 |
| realized_vol_mean | 0.1483 | 0.0687 | 0.1008 |
| skewness | -0.2503 | -0.6737 | -0.5356 |
| tail_index_hill | 4.0331 | 3.7681 | 3.6846 |
| var_95 | -0.0166 | -0.0101 | -0.0151 |
| var_99 | -0.0298 | -0.0189 | -0.0249 |

## Regime Validation

| Dimension | Label | KS p-value | Cohen's d | Energy Dist | N (g/¬g) |
|-----------|-------|-----------|-----------|-------------|-----------|
| vol_regime | regime_high_vol | 0.0009 | 0.148 | 0.0001 | 1000/1000 |
| vol_regime | regime_normal_vol | 0.0001 | -0.211 | 0.0001 | 1000/1000 |
| market_regime | regime_bear | 0.0000 | 0.728 | 0.0011 | 1000/1000 |
| market_regime | regime_bull | 0.0000 | -0.522 | 0.0006 | 1000/1000 |
| market_regime | regime_sideways | 0.0004 | -0.190 | 0.0001 | 1000/1000 |
| macro_regime | regime_expansion | 0.0000 | -0.472 | 0.0005 | 1000/1000 |
| macro_regime | regime_high_inflation | 0.0000 | -0.377 | 0.0003 | 1000/1000 |
| macro_regime | regime_recession | 0.6855 | 0.018 | -0.0000 | 1000/1000 |
| macro_regime | regime_stagflation | 0.0000 | 0.836 | 0.0015 | 1000/1000 |

## Sanity Check
- Reconstruction error (0.3525) is negligible relative to price scale.
- stagflation (42 rows, 0.67%) is underpowered — see KNOWN_ISSUES #9.
- Baseline columns (hist_bootstrap, block_bootstrap, garch_t, vanilla_timegrad) are TODO — Phase 3.
- Full regime-conditional evaluation on all test windows requires GPU (HOST_TASKS.md).