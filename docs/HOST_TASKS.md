# HOST_TASKS.md — GPU-Required Evaluation Tasks

Tasks that require a trained checkpoint and model sampling. Must run on the
owner's host GPU. Never run here (CPU-only).

Phase 3 plumbing is COMPLETE on CPU. This file describes the exact commands
to execute on the host GPU machine.

---

## Prerequisites
- GPU host with CUDA available
- D-① retrain ALREADY COMPLETE (epoch 92 of 100, best at epoch 92, seed 0):
  `runs/retrain_d1__20260628-182222__seed0/checkpoints/model_best.pt`
  ── VERIFY THIS PATH EXISTS ON HOST BEFORE RUNNING ──
- `--model vanilla` flag exists (added Phase 3 STEP 1)
- `src/baselines/run_baselines.py` exists (added Phase 3 STEP 2)

---

## Exact Host Command Sequence

### 1. Re-eval conditional on the existing D-① checkpoint (drawdown fix)
```bash
python run.py --config configs/default.yaml --seed 0 \
  --run-name retrain_d1 --eval \
  --eval-checkpoint runs/retrain_d1__20260628-182222__seed0/checkpoints/model_best.pt \
  --num-samples 200 --max-test-steps 0
```
Regenerates forecast_metrics.json, stylized_facts.json, regime_validation.json,
and EVALUATION_REPORT.md (with per-path drawdown fix). NO TRAINING — checkpoint is reused.

### 2. Train vanilla TimeGrad from scratch (the ONLY training step)
```bash
python run.py --config configs/default.yaml --seed 0 \
  --run-name vanilla_retrain --model vanilla \
  --epochs 100
```
Vanilla uses the same config as conditional (context_length=64, prediction_length=5,
diff_steps=100, beta_schedule=linear, residual_layers=6, residual_channels=32,
lr=1e-3, batch_size=64). Only difference: `--model vanilla` — no conditioning.
100 epochs match the D-① training budget.

### 3. Eval vanilla (load best checkpoint, generate samples + metrics)
```bash
python run.py --config configs/default.yaml --seed 0 \
  --run-name vanilla_eval --model vanilla --eval \
  --eval-checkpoint runs/vanilla_retrain_*/checkpoints/model_best.pt \
  --num-samples 200 --max-test-steps 0
```
Produces forecast_metrics.json, stylized_facts.json, EVALUATION_REPORT.md for vanilla.

### 4. Run CPU baselines at full N (200 ensemble members)
```bash
python -m src.baselines.run_baselines \
  --run-id retrain_d1__20260628-182222__seed0 \
  --data-config configs/default.yaml --seed 0 --num-samples 200
```
Produces runs/retrain_d1__20260628-182222__seed0/metrics/baseline_hist_boot.json,
baseline_block_boot.json, baseline_garch_t.json, and auto-generates
COMPARISON_TABLE.md in that run folder.

### 5. Assemble final COMPARISON_TABLE.md (if not auto-generated)
```bash
python -m src.evaluation.run_eval --run-id retrain_d1__20260628-182222__seed0
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
- `--num-samples 200` for all eval/inference (not training). 10 is for plumbing only;
  200 gives stable coverage and CRPS estimates.
- Regime-conditional sampling (inside `--eval`) is the **single slowest operation**
  — it generates samples for each regime label × each test window.
- All metrics are in denoised-close log returns (canonical space).
- Vanilla has no regime validation (unconditional model) — the comparison table
  omits regime columns for vanilla and baselines.
- Baseline numbers in the smoking run (num-samples=10) are PLUMBING-TEST ONLY.
  Real numbers require num-samples=200 on host.
