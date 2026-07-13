# CFG Sweep — Classifier-Free Guidance

Sweeps cfg_scale w ∈ {0.0, 0.5, 1.0, 2.0, 4.0} and records
regime-conditioned effect sizes (Cohen's d) and coverage.

## Summary Table

| CFG Scale w | Regime Dimension | Label | Cohen's d | KS p-value |
|-------------|-----------------|-------|-----------|------------|
| w=0.0 | vol_regime | regime_high_vol | -0.035 | 0.1205 |
| w=0.0 | vol_regime | regime_normal_vol | -0.073 | 0.0484 |
| w=0.0 | market_regime | regime_bear | -0.031 | 0.1205 |
| w=0.0 | market_regime | regime_bull | -0.021 | 0.6478 |
| w=0.0 | market_regime | regime_sideways | -0.088 | 0.4660 |
| w=0.0 | macro_regime | regime_expansion | 0.029 | 0.7228 |
| w=0.0 | macro_regime | regime_high_inflation | 0.019 | 0.9987 |
| w=0.0 | macro_regime | regime_recession | 0.033 | 0.5006 |
| w=0.0 | macro_regime | regime_stagflation | 0.021 | 0.5006 |
| w=0.5 | vol_regime | regime_high_vol | 0.063 | 0.4660 |
| w=0.5 | vol_regime | regime_normal_vol | -0.035 | 0.5006 |
| w=0.5 | market_regime | regime_bear | 0.162 | 0.0002 |
| w=0.5 | market_regime | regime_bull | -0.219 | 0.0000 |
| w=0.5 | market_regime | regime_sideways | -0.087 | 0.0971 |
| w=0.5 | macro_regime | regime_expansion | -0.062 | 0.0028 |
| w=0.5 | macro_regime | regime_high_inflation | -0.049 | 0.2878 |
| w=0.5 | macro_regime | regime_recession | -0.029 | 0.7228 |
| w=0.5 | macro_regime | regime_stagflation | 0.140 | 0.0256 |
| w=1.0 | vol_regime | regime_high_vol | 0.148 | 0.0039 |
| w=1.0 | vol_regime | regime_normal_vol | -0.119 | 0.0484 |
| w=1.0 | market_regime | regime_bear | 0.521 | 0.0000 |
| w=1.0 | market_regime | regime_bull | -0.386 | 0.0000 |
| w=1.0 | market_regime | regime_sideways | -0.213 | 0.0000 |
| w=1.0 | macro_regime | regime_expansion | -0.154 | 0.0014 |
| w=1.0 | macro_regime | regime_high_inflation | -0.120 | 0.0971 |
| w=1.0 | macro_regime | regime_recession | -0.058 | 0.0224 |
| w=1.0 | macro_regime | regime_stagflation | 0.336 | 0.0000 |
| w=2.0 | vol_regime | regime_high_vol | 0.329 | 0.0000 |
| w=2.0 | vol_regime | regime_normal_vol | -0.231 | 0.0000 |
| w=2.0 | market_regime | regime_bear | 0.947 | 0.0000 |
| w=2.0 | market_regime | regime_bull | -0.638 | 0.0000 |
| w=2.0 | market_regime | regime_sideways | -0.308 | 0.0000 |
| w=2.0 | macro_regime | regime_expansion | -0.235 | 0.0000 |
| w=2.0 | macro_regime | regime_high_inflation | -0.251 | 0.0000 |
| w=2.0 | macro_regime | regime_recession | 0.020 | 0.1812 |
| w=2.0 | macro_regime | regime_stagflation | 0.594 | 0.0000 |
| w=4.0 | vol_regime | regime_high_vol | 0.421 | 0.0000 |
| w=4.0 | vol_regime | regime_normal_vol | -0.437 | 0.0000 |
| w=4.0 | market_regime | regime_bear | 1.083 | 0.0000 |
| w=4.0 | market_regime | regime_bull | -0.688 | 0.0000 |
| w=4.0 | market_regime | regime_sideways | -0.419 | 0.0000 |
| w=4.0 | macro_regime | regime_expansion | -0.340 | 0.0000 |
| w=4.0 | macro_regime | regime_high_inflation | -0.437 | 0.0000 |
| w=4.0 | macro_regime | regime_recession | 0.075 | 0.0224 |
| w=4.0 | macro_regime | regime_stagflation | 0.708 | 0.0000 |

## Interpretation

- w=1.0 = pure conditional (default, current behavior)
- w=0.0 = unconditional (regime conditioning zeroed)
- w>1.0 = amplified conditioning (stress-test mode)
- The default w should maximize regime-control effect size
  without pushing CRPS up by more than 10%.

**Note:** This report was generated on CPU with limited test windows.
Full sweep requires GPU host — see HOST_TASKS.md.