# COMPARISON_TABLE — Canonical Method Comparison

Assembled from committed run folders:
- conditional: `retrain_d1__20260706-212108__seed0`
- vanilla: `vanilla_eval__20260706-214104__seed0`
- CFG w=2: `cfg_eval_w2__20260713-212242__seed0`
- CFG w=4: `cfg_eval_w4__20260713-214944__seed0`

| Method | CRPS | coverage_0.8 | PIT_KS_p | kurtosis | skewness | |r|_ACF1 | leverage_lag1 | VaR_99 | ES_99 | Hill_idx |
|--------|------|--------------|----------|----------|----------|----------|---------------|--------|-------|----------|
| conditional | 0.0043 | 0.6296 | 0.0000 | 3.3576 | -0.5282 | 0.2741 | -0.1334 | -0.0248 | -0.0308 | 3.6792 |
| conditional (CFG w=2) | 0.0039 | 0.6580 | 0.0000 | 4.5756 | -0.2758 | 0.2322 | -0.0702 | -0.0216 | -0.0276 | 3.4927 |
| conditional (CFG w=4) | 0.0042 | 0.5376 | 0.0000 | 8.2035 | -0.9151 | 0.3286 | -0.1159 | -0.0239 | -0.0314 | 2.9731 |
| vanilla | 0.0035 | 0.7062 | 0.0000 | 14.8279 | 0.1603 | 0.2214 | -0.0253 | -0.0240 | -0.0338 | 2.4989 |
| hist_boot | 0.0034 | 0.8024 | 0.0875 | 12.7256 | -0.4943 | -0.0003 | -0.0014 | -0.0229 | -0.0351 | 2.7675 |
| block_boot | 0.0034 | 0.8069 | 0.0933 | 12.2566 | -0.5278 | 0.2355 | -0.1003 | -0.0229 | -0.0350 | 2.8250 |
| garch_t | 0.0039 | 0.9655 | 0.0000 | 327.0903 | -2.7794 | 0.3007 | -0.0076 | -0.0434 | -0.0657 | 2.6385 |
| **real (test)** | TBD | TBD | TBD | 2.1744 | -0.2503 | 0.1392 | -0.0556 | -0.0298 | -0.0359 | 4.0331 |

## Regime Validation (conditional, w=1.0)

| Dimension | Label | KS p-value | Cohen's d |
|-----------|-------|-----------|-----------|
| macro_regime | regime_expansion | 0.0000 | -0.422 |
| macro_regime | regime_high_inflation | 0.0000 | -0.447 |
| macro_regime | regime_recession | 0.1641 | 0.064 (not sig.) |
| macro_regime | regime_stagflation | 0.0000 | 0.739 |
| market_regime | regime_bear | 0.0000 | 0.697 |
| market_regime | regime_bull | 0.0000 | -0.460 |
| market_regime | regime_sideways | 0.0000 | -0.284 |
| vol_regime | regime_high_vol | 0.0006 | 0.195 |
| vol_regime | regime_normal_vol | 0.0000 | -0.220 |

## CFG Regime Effect (bear cohens_d)

| CFG Scale | Cohen's d (bear) |
|-----------|-----------------|
| w=1.0 (conditional) | 0.697 |
| w=2 | 0.872 |
| w=4 | 1.206 |

## Notes
- All metrics in canonical denoised-close log-return space.
- **Real (test)**: un-denoised raw S&P 500 log returns on the test period.
- **Vanilla**: unconditional TimeGrad (no regime conditioning).
- **hist_boot / block_boot / garch_t**: CPU baselines (Phase 3).
- CFG w=2/w=4 rows are full-eval runs (cfg_eval_w2, cfg_eval_w4) —
  not plumbing-scale. Regime d-values above are from those full evals.
- Bear d in the cfg_sweep table (32-window plumbing) differs from the
  full-eval d-values above because of window-count differences.
  The sweep is only used to show the monotonic trend across w.