# FinD Generator — DRAFT (owner writes final)

**Regime-conditional stress scenario generator** built on a conditional
TimeGrad diffusion model with Student-t marginals and classifier-free
guidance (CFG). Generates multi-step return scenarios conditioned on
macro/market/volatility regimes, for counterfactual stress testing,
portfolio risk analysis, and RL-agent training environments.

---

## Headline: regime conditioning works + CFG is a controllable dial

**Regime conditioning is real.** Forcing a specific regime label at
inference time produces a significantly different forecast distribution
(KS test, p < 0.01) in 8 of 9 regime labels. Only recession (108 rows)
fails to separate — stagflation (42 rows) is significant despite severe
underpowering. Source: `runs/retrain_d1__20260706-212108__seed0/metrics/regime_validation.json`.

**CFG scale w is a monotonic stress dial.** Increasing w amplifies the
regime-conditional distributional shift while trading off unconditional
calibration. At w=0 (conditioning zeroed), all Cohen's d values are ~0
(no regime signal). As w rises, bear-market separation grows
monotonically: d ≈ 0.5 (w=1), 0.9 (w=2), 1.1 (w=4). Recession becomes
marginally significant only at w=4 (d=0.08, p=0.022). Source:
`runs/cfg_sweep__20260713-205210__seed0/metrics/cfg_sweep.json`
(32-window plumbing sweep — d-values here differ from the full-eval runs
below because window counts differ; the sweep is used ONLY for the
monotonic-trend story, not for headline effect sizes).

| w | bear d (sweep) | recession d (sweep) | recession p |
|---|---------------|---------------------|-------------|
| 0.0 | −0.03 (ns) | +0.03 (ns) | 0.50 |
| 0.5 | +0.16 | −0.03 (ns) | 0.72 |
| 1.0 | +0.52 | −0.06 (ns) | 0.02 |
| 2.0 | +0.95 | +0.02 (ns) | 0.18 |
| 4.0 | +1.08 | +0.08 | 0.022 |

---

## Canonical Comparison Table

All methods evaluated in the same canonical space (denoised-close log
returns). Every number in this table traces to a committed metrics JSON
under `runs/`. See `COMPARISON_TABLE.md` for source folders.

| Method | CRPS | cov.₈₀ | kurtosis | |r| ACF₁ | VaR₉₉ | Hill |
|--------|------|--------|----------|--------|-------|------|
| conditional | 0.0043 | 0.630 | 3.36 | 0.274 | −0.0248 | 3.68 |
| conditional (CFG w=2) | 0.0039 | 0.658 | 4.58 | 0.232 | −0.0216 | 3.49 |
| conditional (CFG w=4) | 0.0042 | 0.538 | 8.20 | 0.329 | −0.0239 | 2.97 |
| vanilla | 0.0035 | 0.706 | 14.83 | 0.221 | −0.0240 | 2.50 |
| hist_boot | 0.0034 | 0.802 | 12.73 | 0.000 | −0.0229 | 2.77 |
| block_boot | 0.0034 | 0.807 | 12.26 | 0.236 | −0.0229 | 2.83 |
| garch_t | 0.0039 | 0.966 | 327.09 | 0.301 | −0.0434 | 2.64 |
| **real (test)** | — | — | 2.17 | 0.139 | −0.0298 | 4.03 |

- CFG w=2 (bear d=0.87) is Pareto-optimal: it dominates the
  unconditional conditional row on CRPS, coverage, AND regime control.
- CFG w=4 (bear d=1.21) is extreme-stress mode: maximum regime
  separation, recession finally separates (p=0.005), but coverage drops
  from 0.66 → 0.54 and kurtosis overshoots real.
- The CFG bear-d headline values (0.87, 1.21) come from the full-eval
  runs `cfg_eval_w2` / `cfg_eval_w4`; the sweep table above uses
  plumbing-scale runs with fewer windows — values are NOT interchangeable.

**Regime Validation at w=1.0** (conditional, from
`runs/retrain_d1__20260706-212108__seed0/metrics/regime_validation.json`):

| Dimension | Label | KS p | d | Sig? |
|-----------|-------|------|---|------|
| vol | high_vol | <0.001 | 0.20 | Yes |
| vol | normal_vol | <0.001 | −0.22 | Yes |
| market | bear | <0.001 | 0.70 | Yes |
| market | bull | <0.001 | −0.46 | Yes |
| market | sideways | <0.001 | −0.28 | Yes |
| macro | expansion | <0.001 | −0.42 | Yes |
| macro | high_inflation | <0.001 | −0.45 | Yes |
| macro | recession | 0.164 | 0.06 | **No** |
| macro | stagflation | <0.001 | 0.74 | Yes |

---

## Stress Demo

Generate regime-conditioned scenario paths, apply them to an example
portfolio, and compare risk metrics (VaR, ES, max drawdown) against
unconditional and historical returns.

### Quick usage (CPU, test scale)

```bash
# Generate scenario returns under forced high-inflation regime at w=2
python -m src.stress_demo.scenario_run \
  --checkpoint runs/cfg_smoke_cond__20260708-124243__seed0/checkpoints/model_best.pt \
  --regime '{"macro_regime":"high_inflation"}' \
  --cfg-scale 2.0 --num-scenarios 50 --num-windows 8 \
  --run-dir runs/stress_demo_test --seed 0

# Compute VaR/ES/drawdown and produce plots
python -c "
from pathlib import Path
from src.stress_demo.scenario_run import run_scenario
from src.stress_demo.portfolio_stress import (
    ExamplePortfolio, compute_stress_comparison,
    write_stress_var_table, generate_stress_fan_chart,
)
import numpy as np

data = np.load('runs/stress_demo_test/samples/scenario_returns.npz')
scenario_results = {
    'scenario_returns': data['scenario'],
    'unconditional_returns': data['unconditional'],
    'historical_returns': data['historical'],
    'cfg_scale': 2.0,
    'regime_spec': {'macro_regime': 'high_inflation'},
}
results = compute_stress_comparison(scenario_results)
write_stress_var_table(results, Path('runs/stress_demo_test/stress_var_table.md'))
generate_stress_fan_chart(scenario_results, Path('runs/stress_demo_test/stress_fan_chart.png'))
print('Done — see runs/stress_demo_test/')
"
```

Full-GPU stress runs: see `docs/HOST_TASKS.md` §5b–5c.

---

## Honest Limitations

1. **Bootstrap and vanilla beat this model on unconditional metrics.**
   The historical bootstrap achieves CRPS 0.0034 and coverage 0.80
   versus 0.0043/0.63 for the conditional model. GARCH-t has better
   coverage (0.97). Vanilla TimeGrad has better CRPS (0.0035) and
   coverage (0.71). The conditional model exists to provide targeted
   regime control, not to win on unconditional sharpness.

2. **Recession is underpowered** (108 rows, 1.7% of data). It is the
   only non-significant regime at w=1.0 (p=0.16) and only reaches
   marginal significance at the extreme w=4 setting (p=0.022, d=0.08).

3. **Stagflation is severely underpowered** (42 rows, 0.67% of data).
   While it is statistically significant in KS tests, the tiny training
   support means the regime embedding is noisy and unreliable.

4. **The model learns denoised targets**, so its generated-return
   kurtosis differs from raw-return kurtosis. The "generated" column
   should be compared against `real_denoised` (kurtosis 3.79) for
   calibration, and against `real_raw_un_denoised` (kurtosis 2.17) for
   practical tail-risk assessment. CFG w=4 overshoots even the denoised
   reference (kurtosis 8.20).

5. **CFG w=4 trades calibration for regime control.** At w=4, coverage
   drops from 0.66 to 0.54 — far below nominal 0.80. Use w=2 for
   balanced stress testing and w=4 only when maximum regime separation
   is needed (e.g., crash-scenario generation for downstream RL
   robustness training).

6. **All forecasts are 5-step ahead.** A 5-step horizon limits max
   drawdown duration to 4 steps per path, so generated drawdown stats
   are not comparable to the raw-real row's multi-year continuous
   drawdown. The comparison table reports per-path drawdowns only;
   scenario-level drawdowns appear in the stress demo.

---

## Reproducibility

- **Leakage tested:** `tests/test_no_leakage.py` verifies 100 random
  timestamps — every feature computed from past-only data.
- **Causal pipeline:** wavelet denoising is rolling-window (no
  look-ahead), no `.bfill()`, all regime thresholds fit on train-only.
  Documented in `docs/data_integrity.md`.
- **Seed:** `--seed 0` for all committed runs. Determinism verified by
  bit-for-bit reprotest.
- **Run folders:** every metric in this README traces to a committed
  JSON under `runs/`:
  - `retrain_d1__20260706-212108__seed0` (conditional, baselines)
  - `vanilla_eval__20260706-214104__seed0` (vanilla)
  - `cfg_eval_w2__20260713-212242__seed0` (CFG w=2 full eval)
  - `cfg_eval_w4__20260713-214944__seed0` (CFG w=4 full eval)
  - `cfg_sweep__20260713-205210__seed0` (CFG sweep, monotonic trend)

---

*DRAFT — the owner writes the final version. Every number in this
document is sourced from a committed metrics JSON. See
`COMPARISON_TABLE.md` for the full canonical table and
`docs/KNOWN_ISSUES.md` for open issues.*
