# COMPARISON_TABLE — Phase 3 Baseline Battery

Run: `retrain_d1__20260706-212108__seed0`

| Method | CRPS | coverage_0.8 | PIT_KS_p | kurtosis | skewness | |r|_ACF1 | leverage_lag1 | VaR_99 | ES_99 | Hill_idx |
|--------|------|--------------|----------|----------|----------|----------|---------------|--------|-------|----------|
| conditional | 0.0043 | 0.6296 | 0.0000 | 3.3576 | -0.5282 | 0.2741 | -0.1334 | -0.0248 | -0.0308 | 3.6792 |
| vanilla | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| hist_boot | 0.0034 | 0.8024 | 0.0875 | 12.7256 | -0.4943 | -0.0003 | -0.0014 | -0.0229 | -0.0351 | 2.7675 |
| block_boot | 0.0034 | 0.8069 | 0.0933 | 12.2566 | -0.5278 | 0.2355 | -0.1003 | -0.0229 | -0.0350 | 2.8250 |
| garch_t | 0.0039 | 0.9655 | 0.0000 | 327.0903 | -2.7794 | 0.3007 | -0.0076 | -0.0434 | -0.0657 | 2.6385 |
| **real (test)** | TBD | TBD | TBD | 2.1744 | -0.2503 | 0.1392 | -0.0556 | -0.0298 | -0.0359 | 4.0331 |

## Notes
- **Real (test)**: un-denoised raw S&P 500 log returns on the test period.
- **Vanilla**: placeholder row — requires host GPU training (see HOST_TASKS.md).
- **hist_boot / block_boot / garch_t**: CPU baselines (Phase 3).
- All metrics computed in canonical space (denoised-close log returns).
- Numbers shown are **plumbing-test only** unless generated on the host GPU with full training.