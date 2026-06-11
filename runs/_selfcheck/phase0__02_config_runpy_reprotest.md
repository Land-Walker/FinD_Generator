# Self-check — phase0 / units 02+03: config + run.py rewiring + reproducibility test

Date: 2026-06-11

## D1 pytest (full suite)
`python3 -m pytest -x --tb=short -q`
```
..........                                                               [100%]
10 passed in 28.36s
```

## D2 smoke test (exact acceptance command)
`python3 run.py --config configs/default.yaml --seed 0 --run-name smoke --epochs 1 --max-train-steps 2 --max-val-steps 2 --num-samples 1`
- exit code 0, wallclock 13.8 s on CPU (< 5 min requirement)
- tail of output:
```
✅ Datasets built. Train: 5044, Val: 1027, Test: 1028 samples.
Epoch 1: train_loss=0.7328, val_loss=0.5520
Saved checkpoint to .../runs/smoke__20260611-082835__seed0/checkpoints/model_last.pt
Generated samples shape: torch.Size([1, 64, 5, 1])
✅ Saved forecasts to .../runs/smoke__20260611-082835__seed0/samples/forecasts.pt
```
- Run-folder artifact audit (acceptance: "all required artifacts"):
```
checkpoints/model_last.pt
config.yaml
git_diff.patch
git_sha.txt
metrics/train_log.jsonl
samples/forecasts.pt
plots/  logs/   (created, empty at this phase)
```

## D3 determinism (actual programmatic diff, W2.4 — not asserted, shown)
Two runs, same command, seed 0, num-samples 2. Raw train_log.jsonl lines:
```
A: {"epoch": 1, "train_loss": 0.732781708240509, "val_loss": 0.5519608557224274, "lr": 0.001, "wallclock": 0.22255492210388184, "gpu_mem": 0}
B: {"epoch": 1, "train_loss": 0.732781708240509, "val_loss": 0.5519608557224274, "lr": 0.001, "wallclock": 0.16739916801452637, "gpu_mem": 0}
```
Programmatic comparison output:
```
epoch: A=1 B=1 diff=False
train_loss: A=0.732781708240509 B=0.732781708240509 diff=0.0
val_loss: A=0.5519608557224274 B=0.5519608557224274 diff=0.0
lr: A=0.001 B=0.001 diff=0.0
D3 DETERMINISM: bit-for-bit identical (exact float equality)
```
Additionally `torch.equal(forecasts_A, forecasts_B) == True` (shape (2,64,5,1)):
sampling is bit-for-bit reproducible too, not just the loss path.
(wallclock/gpu_mem fields are excluded by design — timing is not part of determinism.)

## D4 pyflakes
`python3 -m pyflakes src/ tests/ run.py` — files authored/modified this phase
(run.py, configs/, src/utils/*, src/preprocessor/__init__.py, tests/*) are CLEAN.
Remaining findings are pre-existing in files NOT touched this phase:
- src/__init__.py: re-export false positives (ScenarioFeatureGenerator/ScenarioSpec)
- src/preprocessor/data_loader.py: unused imports (adfuller, plot_acf, plot_pacf) — Phase 1 file, will be cleaned there

## D5 honesty audit (files authored/modified this unit)
run.py, configs/default.yaml, requirements.txt, src/preprocessor/__init__.py, tests/test_reproducibility.py
- [x] No `pass` placeholder
- [x] No TODO / FIXME / XXX (grep verified)
- [x] No commented-out real logic
- [x] No bare except / swallowed exceptions (resolve_config raises; run_inference re-raises naturally)
- [x] No random.seed() outside src/utils/seed.py
- [x] No outputs outside the run folder (checkpoint + forecasts now under runs/<id>/)
- [x] test_reproducibility.py exercises the full CLI path; resolve_config covered indirectly by both subprocess tests and the D2 command
- Note: tests/test_reproducibility.py caps steps at 3 train / 2 val. Justification: the
  property under test is trajectory determinism; the spec's own D2/D3 protocol uses
  caps of 2/2. Recorded in PROGRESS.md.

## D6 result-plausibility
N/A — no evaluation tables yet (first applies in Phase 2).

VERDICT: Phase 0 acceptance checks ALL PASS (pytest, smoke command, artifact audit, no pass/TODO).
