# COMPARISON_TABLE — Phase 3 Baseline Battery

Run: `cfg_eval_w4__20260713-214944__seed0`

| Method | CRPS | coverage_0.8 | PIT_KS_p | kurtosis | skewness | |r|_ACF1 | leverage_lag1 | VaR_99 | ES_99 | Hill_idx |
|--------|------|--------------|----------|----------|----------|----------|---------------|--------|-------|----------|
| conditional | 0.0042 | 0.5376 | 0.0000 | 8.2035 | -0.9151 | 0.3286 | -0.1159 | -0.0239 | -0.0314 | 2.9731 |
| vanilla | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| hist_boot | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| block_boot | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| garch_t | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| **real (test)** | TBD | TBD | TBD | 2.1744 | -0.2503 | 0.1392 | -0.0556 | -0.0298 | -0.0359 | 4.0331 |

## Notes
- **Real (test)**: un-denoised raw S&P 500 log returns on the test period.
- **Vanilla**: placeholder row — requires host GPU training (see HOST_TASKS.md).
- **hist_boot / block_boot / garch_t**: CPU baselines (Phase 3).
- All metrics computed in canonical space (denoised-close log returns).
- Numbers shown are **plumbing-test only** unless generated on the host GPU with full training.