# Self-check — phase1: causal data hygiene (full phase)

Date: 2026-06-12

## D1 pytest (full suite, run in two invocations due to sandbox 45 s/call cap; all tests executed)
```
tests/test_seed.py + test_run_folder.py + test_causal_wavelet.py + test_no_leakage.py
  -> 18 passed in 12.82s
tests/test_reproducibility.py
  -> 1 passed in 32.58s
TOTAL: 19 passed, 0 failed
```

## D2 smoke (exact spec command, num-samples 2)
`python run.py --config configs/default.yaml --seed 0 --run-name smoke --epochs 1 --max-train-steps 2 --max-val-steps 2 --num-samples 2`
exit 0, ~16 s end-to-end on CPU (< 5 min). Post-fix datasets: Train 4999 / Val 1018 / Test 1018 windows.

## D3 determinism (pasted diff, W2.4)
Two identical invocations, seed 0:
```
A: {"epoch": 1, "train_loss": 0.655252993106842, "val_loss": 0.5086070150136948, "lr": 0.001, ...}
B: {"epoch": 1, "train_loss": 0.655252993106842, "val_loss": 0.5086070150136948, "lr": 0.001, ...}
```
train_loss/val_loss bit-for-bit identical (exact float equality).

## D4 pyflakes
`python -m pyflakes` on all files authored/modified this phase
(src/preprocessor/data_loader.py, src/config.py, scripts/coverage_probe.py,
tests/test_no_leakage.py, tests/test_causal_wavelet.py): CLEAN.
Pre-existing: src/__init__.py re-export false positives (untouched; KNOWN_ISSUES #7).

## D5 honesty audit (files authored/modified this phase)
- [x] No `pass` placeholder, no TODO/FIXME/XXX
- [x] No commented-out real logic (legacy commented-out TimeGradDataset class
      was REMOVED from data_loader.py during the rewrite, not kept)
- [x] No bare except / silent swallowing (probe re-raises on bad checkpoint
      load; _ffill_checked and merge raise on NaN; asserts on thresholds)
- [x] No random.seed outside src/utils/seed.py (probe uses set_global_seed)
- [x] No hardcoded output paths outside runs/
- [x] Numerical functions tested: wavelet kernel + series (6 tests), all 58
      feature values (test_no_leakage), threshold fit (dedicated test),
      source-level bfill guard (dedicated test)
- Note: test_no_leakage reuses `_denoise_window` for the wavelet recompute —
  fed exclusively truncated data, so any future-dependence in the pipeline
  value would still fail the equality. All other recomputes are independent
  reimplementations.

## D6 result-plausibility (coverage probe table)
| quantity | report | independent recompute | agree |
|---|---|---|---|
| coverage_80 (overall) | 0.1165365 | 0.1165365 | ✓ |
| coverage_80 at horizon step 1 | 0.078125 | 0.078125 | ✓ |
| roll_vol train median (thresholds) | 0.00867969920913796 | 0.00867969920913796 | ✓ |

## Phase 1 acceptance checks
- pytest tests/test_no_leakage.py -v → 3 passed (100 timestamps × 58 features; 0 violations) ✓
- 80% coverage reported honestly before (0.1165 probe / 0.0938 README historical)
  and after (NOT MEASURABLE with existing checkpoints — dim incompatibility,
  KNOWN_ISSUES #6; full statement in docs/data_integrity.md) ✓ (honest)
- docs/data_integrity.md exists and is filled in ✓

VERDICT: Phase 1 complete. MANDATORY STOP (W3.3) — waiting for owner review.
