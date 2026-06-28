# PHASE 3 PLAN — Baseline Battery Execution Plan

## Status

| Baseline | CPU/GPU | Status | Qty |
|----------|---------|--------|-----|
| historical_bootstrap | **CPU** | ✅ IMPLEMENTED + TESTED | `src/baselines/historical_bootstrap.py` |
| block_bootstrap | **CPU** | ✅ IMPLEMENTED + TESTED | `src/baselines/block_bootstrap.py` |
| garch_t | **CPU** | ✅ IMPLEMENTED + TESTED | `src/baselines/garch_baseline.py` |
| vanilla_timegrad | **GPU (host)** | 🔧 CODE EXISTS, NEEDS TRAINING PATH | see below |

---

## CPU Baselines (DONE — next session just runs them)

All three CPU baselines are under `src/baselines/` and produce samples in the
**canonical space** (denoised-close log returns). Output shape:
`(n_samples, n_paths, horizon)`.

Unit tests: `tests/test_baseline_historical.py` (5 tests),
`tests/test_baseline_block.py` (6 tests, incl. ACF preservation),
`tests/test_baseline_garch.py` (7 tests, incl. unconditional-variance check).

---

## vanilla_timegrad — Investigation Results

### What EXISTS

| Component | File | Line |
|-----------|------|------|
| `VanillaTimeGradPredictionNetwork` | `src/predictor/prediction_network.py` | 338 |
| `VanillaTimeGradTrainingNetwork` | `src/training/training_network.py` | 364 |
| Checkpoint | `checkpoints/original_timegrad_best.pt` | — |

Both classes are **fully implemented** and reconstructable. They use the
unconditional `TimeGradBase` model (no cross-attention, no conditioning).

### What is MISSING (needs new code)

`run.py` currently hardcodes the **conditional** path only:
- `_build_networks()` creates `ConditionalTimeGradTrainingNetwork` (line 134)
  and `ConditionalTimeGradPredictionNetwork` (line 149)
- `train_and_validate()` takes `ConditionalTimeGradTrainingNetwork` (line 169)
- `run_inference()` takes `ConditionalTimeGradPredictionNetwork` (line 260)

**There is NO `--model {conditional,vanilla}` flag or vanilla training path in `run.py`.**

### Decision: Option B — Train vanilla from scratch on clean data

**Chosen.** Rationale: the existing `checkpoints/original_timegrad_best.pt` was
trained on the leaky pre-Phase-1 pipeline. For an honest apples-to-apples
comparison, vanilla must train on the **same clean causal pipeline** as the
conditional model, same seed, comparable epochs. The vanilla model is actually
lighter (no cross-attention, no conditioning overhead) — it should train faster
than the conditional model on the same budget.

---

## Exact Host Commands (next session execution checklist)

### Prerequisites
- GPU host with CUDA available
- `checkpoints/conditional_timegrad_best.pt` from D-① retrain
- `checkpoints/original_timegrad_best.pt` (existing, or from Option B retrain)

### Step 1 — Run CPU baselines (can be done on CPU or GPU host)
```bash
python -m src.baselines.run_baselines --run-id retrain_d1 \
  --data-config configs/default.yaml --seed 0
```
(This module is NOT yet implemented — the `run_baselines.py` glue needs to be
written. It loads train data, runs each CPU baseline, saves samples to
`runs/retrain_d1/samples/baseline_*.pt`, then runs forecast_metrics +
stylized_facts on each.)

### Step 2 — Train + generate vanilla TimeGrad samples (GPU host)
```bash
# Train vanilla TimeGrad from scratch on clean causal pipeline:
python run.py --config configs/default.yaml --seed 0 \
  --run-name vanilla_retrain --model vanilla \
  --epochs 200

# Generate samples from best checkpoint:
python run.py --config configs/default.yaml --seed 0 \
  --run-name vanilla_infer --model vanilla \
  --checkpoint runs/vanilla_retrain_*/checkpoints/model_best.pt \
  --inference-only --num-samples 256
```

### Step 3 — Run full evaluation on all baselines
```bash
python run.py --config configs/default.yaml --seed 0 --run-name retrain_d1 --eval
```
(Needs to be extended to call baselines + generate the comparison table.)

### Step 4 — Generate COMPARISON_TABLE.md
```bash
python -m src.evaluation.run_eval --run-id retrain_d1 --baselines all
```

---

## Canonical Evaluation Space

ALL methods (conditional TimeGrad, vanilla TimeGrad, historical bootstrap,
block bootstrap, GARCH-t) are compared in the **denoised-close log returns**
space:

- Model samples: `PCA features → target_pca_to_log_returns() → log returns`
- Baseline samples: generate log returns directly (already in canonical space)
- Evaluation functions (`forecast_metrics`, `stylized_facts`) operate on
  log returns for all methods

This ensures apples-to-apples comparison.

---

## Work Remaining for Next Session

1. **Decide vanilla_timegrad approach** (Option A vs B above) — owner decision
2. **Implement `src/baselines/run_baselines.py`** — glue that runs all 4
   baselines + evaluation (can use the 3 CPU baselines already implemented)
3. **Add `--model vanilla` support to `run.py`** (if Option B)
4. **Wire evaluation**: baselines produce canonical returns → same
   `forecast_metrics` + `stylized_facts` functions → COMPARISON_TABLE.md
5. **Include all baselines in EVALUATION_REPORT.md** per MASTER_SPEC Phase 3