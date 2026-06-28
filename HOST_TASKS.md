# HOST_TASKS.md — GPU-Required Evaluation Tasks

Tasks that require a trained checkpoint and model sampling. Must run on the
owner's host GPU after D-① retraining completes. Never run here (CPU-only).

---

## Prerequisites
- D-① retrain on 2000–2024 data completed: `python run.py --config configs/default.yaml --seed 0 --run-name retrain_d1`
- Checkpoints saved to `runs/retrain_d1/checkpoints/`
- Evaluation modules (Phase 2) implemented and unit-tested: ✅

---

## Task 1 — Inverse Transform Reconstruction Error (confirm on real data)
```bash
python -m pytest tests/test_evaluation_inverse.py::test_reconstruction_error_report_on_real_data -v -s
```
Record the PCA variance retained and the close-level RMSE from the output.
The CPU smoke-run already reports these for the existing in-memory pipeline;
confirm on the retrained checkpoint's scaler/PCA.

---

## Task 2 — Forecast Metrics on Generated Samples
```python
# Pseudocode — integrate into run.py --eval or run standalone:
from src.evaluation.forecast_metrics import *
from src.evaluation.inverse_transform import target_pca_to_log_returns

# Load retrained checkpoint, generate samples on test split
# samples shape: (n_samples, n_time, n_target_pca_features)
# targets shape: (n_time, n_target_pca_features) — real test data

# Convert to canonical space (denoised-close log returns)
pred_returns = target_pca_to_log_returns(samples_batch, pca, scaler)
true_returns = target_pca_to_log_returns(targets_batch, pca, scaler)

# Compute metrics
metrics = {
    'crps': crps_ensemble(pred_returns, true_returns),
    'mae': mae(pred_returns, true_returns),
    'rmse': rmse(pred_returns, true_returns),
    'ql_05': quantile_loss(pred_returns, true_returns, 0.05),
    'ql_10': quantile_loss(pred_returns, true_returns, 0.10),
    'ql_50': quantile_loss(pred_returns, true_returns, 0.50),
    'ql_90': quantile_loss(pred_returns, true_returns, 0.90),
    'ql_95': quantile_loss(pred_returns, true_returns, 0.95),
    'coverage_50': coverage(pred_returns, true_returns, 0.5),
    'coverage_80': coverage(pred_returns, true_returns, 0.8),
    'coverage_90': coverage(pred_returns, true_returns, 0.9),
    'coverage_95': coverage(pred_returns, true_returns, 0.95),
    'energy_score': energy_score(pred_returns, true_returns),
    'nll': negative_log_likelihood(pred_returns, true_returns),
}
pit = pit_values(pred_returns, true_returns)
ks_stat, ks_pval = pit_ks_test(pit)
```
Save to `runs/<run_id>/metrics/forecast_metrics.json`.

---

## Task 3 — Stylized Facts Comparison
```python
from src.evaluation.stylized_facts import all_stylized_facts

# Real test returns (RAW, un-denoised)
real_returns = ...  # raw log returns from test-target close

# Generated returns (in canonical space)
gen_returns = target_pca_to_log_returns(samples, pca, scaler)

facts_real = all_stylized_facts(real_returns)
facts_gen = all_stylized_facts(gen_returns)
```
Save as `runs/<run_id>/metrics/stylized_facts.json`.
The RAW-returns column must use un-denoised log returns so the effect of
denoising on kurtosis/tails is visible and honest.

---

## Task 4 — Regime Validation (HEADLINE)
```python
from src.evaluation.regime_validation import regime_validation_report

# Requires: generated samples conditioned on each regime label.
# For each regime dimension (vol, market, macro), generate N samples per label.
# samples_by_regime = {
#     'vol_regime': {'high_vol': samples_hv, 'normal_vol': samples_nv},
#     'market_regime': {'bear': samples_bear, 'bull': samples_bull, 'sideways': samples_sw},
#     'macro_regime': {'expansion': samples_exp, 'high_inflation': samples_hi,
#                      'recession': samples_rec, 'stagflation': samples_stag},
# }

results = regime_validation_report(samples_by_regime,
                                    ['vol_regime', 'market_regime', 'macro_regime'])
```
Save as `runs/<run_id>/metrics/regime_validation.json`.

**IMPORTANT — stagflation caveat (KNOWN_ISSUES #9):**
stagflation has only 42 rows (0.67%). The validation report MUST label it
"insufficient support — validation underpowered" rather than report a clean
pass/fail. The code already detects small samples (<50) and emits a warning;
confirm this warning appears in the output.

---

## Task 5 — EVALUATION_REPORT.md
After all metrics are computed, assemble `runs/<run_id>/EVALUATION_REPORT.md`:
1. Forecast metrics table
2. Stylized facts comparison table (real vs generated vs baselines, once Phase 3 is done)
3. Regime validation table with Cohen's d
4. Honest assessment of what improved vs what didn't
5. stagflation clearly marked underpowered

---

## Task 6 — Baseline Battery (Phase 3 integration)
Once baselines are implemented (Phase 3), add their columns to the stylized
facts table and the forecast metrics comparison. The canonical comparison
space is denoised-close log returns for ALL methods (model + baselines).