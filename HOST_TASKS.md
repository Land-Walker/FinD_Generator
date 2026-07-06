# HOST_TASKS.md — GPU-Required Evaluation Tasks

Tasks that require a trained checkpoint and model sampling. Must run on the
owner's host GPU after D-① retraining completes. Never run here (CPU-only).

Phase 3 plumbing is COMPLETE on CPU. This file describes the exact commands
to execute on the host GPU machine.

---

## Prerequisites
- GPU host with CUDA available
- D-① retrain completed: `python run.py --config configs/default.yaml --seed 0 --run-name retrain_d1`
- Checkpoints saved to `runs/retrain_d1/checkpoints/`
- `--model vanilla` flag exists (added Phase 3 STEP 1)
- `src/baselines/run_baselines.py` exists (added Phase 3 STEP 2)

---

## Exact Host Command Sequence

### 1. Re-run conditional --eval on the D-① checkpoint
```bash
python run.py --config configs/default.yaml --seed 0 \
  --run-name retrain_d1 --eval \
  --eval-checkpoint runs/retrain_d1_*/checkpoints/model_best.pt \
  --num-samples 200 --max-test-steps 0
```
This regenerates forecast_metrics.json, stylized_facts.json, regime_validation.json,
and EVALUATION_REPORT.md for the conditional model.

### 2. Train vanilla TimeGrad from scratch (same budget as conditional)
```bash
python run.py --config configs/default.yaml --seed 0 \
  --run-name vanilla_retrain --model vanilla \
  --epochs 100 --num-samples 200
```
Trains VanillaTimeGradTrainingNetwork on the same clean causal pipeline,
same seed 0, same data. Must match the conditional training budget in epochs.

### 3. Eval vanilla + generate samples
```bash
python run.py --config configs/default.yaml --seed 0 \
  --run-name vanilla_eval --model vanilla --eval \
  --eval-checkpoint runs/vanilla_retrain_*/checkpoints/model_best.pt \
  --num-samples 200 --max-test-steps 0
```
Produces forecast_metrics.json, stylized_facts.json, EVALUATION_REPORT.md for vanilla.

### 4. Run CPU baselines on the D-① test split
```bash
python -m src.baselines.run_baselines --run-id retrain_d1 \
  --data-config configs/default.yaml --seed 0 --num-samples 200
```
Produces runs/retrain_d1/metrics/baseline_hist_boot.json,
baseline_block_boot.json, baseline_garch_t.json, and COMPARISON_TABLE.md.

### 5. Assemble final COMPARISON_TABLE.md (if not auto-generated)
```bash
python -m src.evaluation.run_eval --run-id retrain_d1
```
Reads all available metrics JSONs and writes COMPARISON_TABLE.md with rows:
conditional, vanilla, hist_boot, block_boot, garch_t, real (test).

### 6. Manual spot-checks
- Verify vanilla CRPS/coverage are plausible (within 2x of conditional).
- If vanilla significantly outperforms conditional on a metric, report honestly.
- Confirm stagflation warning appears in regime_validation.json.
- Verify baseline CRPS and kurtosis are in the same order of magnitude as the models.
- The comparison table's "real (test)" row should populate from stylized_facts.json.

---

## Notes
- `--num-samples 200` gives statistical stability; use the same N for all methods.
- All metrics are in denoised-close log returns (canonical space).
- Vanilla has no regime validation (unconditional model) — the comparison table
  omits regime columns for vanilla and baselines.
- Baseline numbers in the smoking run (num-samples=10) are PLUMBING-TEST ONLY.
  Real numbers require num-samples=200 on host.
