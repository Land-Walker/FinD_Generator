# Phase 4.4 CFG Plumbing — Self-Check

## D1 — Pytest
```
77 passed in 49.59s
```

## D2 — Cond Smoke (cfg_dropout=0, cfg_scale=1.0)
```json
{"epoch": 1, "train_loss": 0.6979, "val_loss": 0.5498}
```
Generated samples shape: [1, 64, 5, 1] — runs end-to-end.

## D2b — CFG Smoke (cfg_dropout=0.1, cfg_scale=2.0)
```json
{"epoch": 1, "train_loss": 0.5753, "val_loss": 0.3987}
```
Generated samples shape: [1, 64, 5, 1] — runs end-to-end with CFG active.
Lower loss with dropout=0.1 expected (some samples unconditioned → easier task).

## D4 — Pyflakes
Ran: `python -m pyflakes src/ tests/` — see below.

## D5 — Honesty Audit
- [x] No `pass` placeholders
- [x] No TODO / FIXME / XXX
- [x] No commented-out real logic
- [x] No bare `except:` or `except Exception: pass`
- [x] No `random.seed()` calls outside seed utility
- [x] No hardcoded paths outside run folder
- [x] New functions have tests or are smoke-verified via run.py

## Files changed
1. `src/training/training_network.py` — cfg_dropout param, dropout in forward
2. `src/predictor/prediction_network.py` — cfg_scale param, set_cfg_scale
3. `src/models/conditional_timegrad/conditioned_epsilon_theta.py` — cfg_scale, set_cfg_scale, _forward_cond refactor, CFG combination
4. `src/models/conditional_timegrad/conditional_model.py` — cfg_scale, original dims, set_cfg_scale
5. `src/evaluation/cfg_sweep.py` — NEW: sweep plumbing + report
6. `run.py` — --cfg-dropout, --cfg-scale, --cfg-sweep flags
7. `configs/default.yaml` — cfg_dropout, cfg_scale defaults
8. `docs/HOST_TASKS.md` — CFG retrain + sweep host commands
