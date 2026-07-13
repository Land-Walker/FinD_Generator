# COMPARISON_TABLE — Phase 3 Baseline Battery

Run: `cfg_eval_w2__20260713-212242__seed0`

| Method | CRPS | coverage_0.8 | PIT_KS_p | kurtosis | skewness | |r|_ACF1 | leverage_lag1 | VaR_99 | ES_99 | Hill_idx |
|--------|------|--------------|----------|----------|----------|----------|---------------|--------|-------|----------|
| conditional | 0.0039 | 0.6580 | 0.0000 | 4.5756 | -0.2758 | 0.2322 | -0.0702 | -0.0216 | -0.0276 | 3.4927 |
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