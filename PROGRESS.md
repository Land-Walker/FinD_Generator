# PROGRESS

## 2026-05-02 — Initialization and execution plan

### Repository understanding
- The repository implements a conditional TimeGrad pipeline for financial scenario generation with key entrypoints in `run.py`, preprocessing in `src/preprocessor/`, core diffusion components under `src/models/timegrad_core/`, conditioning logic under `src/models/conditional_timegrad/`, and training/prediction wrappers in `src/training/` and `src/predictor/`.
- Existing documentation in `docs/architecture.md` and `docs/Roadmap.md` defines current architecture and known gaps that align with the requested Phase 0→4 roadmap.
- Existing model checkpoints (`checkpoints/original_timegrad_best.pt` and `checkpoints/conditional_timegrad_best.pt`) will be preserved and used for baseline/evaluation phases per instructions.

### Planned execution order
1. **Phase 0 — Reproducibility Foundation**
   - Add deterministic global seeding utility.
   - Add run-folder artifact management.
   - Introduce `omegaconf` config loading and CLI override compatibility.
   - Route outputs into run directories and persist train logs.
   - Pin requirements and add dependencies.
   - Add reproducibility test and pass acceptance checks.
2. **Phase 1 — Causal Data Hygiene**
   - Remove look-ahead leakage in wavelet denoising and missing-value handling.
   - Move regime-threshold fitting to train-only post-split.
   - Add leakage test and data integrity documentation.
3. **Phase 2 — Research-Grade Evaluation Suite**
   - Implement forecast metrics, stylized facts, and regime validation modules.
   - Add unit tests with synthetic checks.
   - Wire evaluation mode and generate consolidated report artifacts.
4. **Phase 3 — Baseline Battery**
   - Implement and run historical/bootstrap/GARCH/vanilla-TimeGrad baselines.
   - Save baseline samples and produce canonical comparison table.
5. **Phase 4 — Targeted Model Improvements**
   - DDIM sampler, cosine schedule, calibration fixes, CFG (+ optional 4.5 if budget allows).
   - Run before/after ablations for each sub-phase and document honest outcomes.

### Process constraints I will enforce
- No placeholders, fabricated metrics, or skipped baselines/tests.
- Determinism and reproducibility checks after each commit-sized change.
- Strict phase ordering with acceptance gates before advancing.
- Transparent reporting of negative or neutral outcomes.
- `BLOCKED.md` creation and stop behavior if ambiguity requires owner decision.
