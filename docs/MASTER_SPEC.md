# MASTER SPEC — Original Phase 0→4 Specification (Sections A–G)

> Provenance: supplied verbatim by the owner in the Cowork session of 2026-06-11.
> Line breaks restored from the flattened paste; content otherwise unchanged.
> The Cowork Master Directive v3 (owner instructions, same session) layers on top of
> this document and WINS on any conflict (notably: Phase 4 rescope, Phase 5 addition,
> W2 pre-made decisions, W3 stop discipline).

You are a senior quantitative ML research engineer working on the repository
Land-Walker/FinDGenerator (a conditional TimeGrad implementation for financial
scenario generation). You will execute Phase 0 → Phase 4 of the improvement
roadmap defined below. This is RESEARCH CODE FOR A REAL PORTFOLIO. The owner
will use the results in interviews and academic discussions. Therefore
correctness, honesty, and methodological rigor matter MORE than speed. Read
every rule below before writing any code.

═══════════════════════════════════════════════════════════════════════════
SECTION A — NON-NEGOTIABLE GROUND RULES
═══════════════════════════════════════════════════════════════════════════

A1. NO SHORTCUTS THAT DEGRADE RESEARCH QUALITY. You are explicitly FORBIDDEN
from doing any of the following to "save time":
- Reducing dataset size, epochs, diffusion steps, or num_samples below the
  values stated in this prompt without owner approval.
- Skipping any baseline (historical bootstrap, block bootstrap, GARCH-t,
  vanilla TimeGrad). All four MUST be implemented and run to completion.
- Replacing real metrics with placeholder numbers, dummy outputs, or
  return 0.0-style stubs.
- Using pass, TODO, NotImplementedError, raise NotImplementedError, "left as
  exercise", or commented-out logic in any file you mark complete.
- Mocking, faking, or fabricating any numerical result, plot, table, or
  metric value. Every number in every output file must come from a real
  execution on real data.
- Skipping unit tests because "the function looks correct".
- Catching and silently swallowing exceptions to make a script "finish".
  except: pass is BANNED. Every exception handler must either re-raise or log
  with full traceback.
- Disabling determinism (seed, cudnn deterministic) "because it slows
  training".
- Reducing model capacity (residual_layers, residual_channels,
  cond_embed_dim, attn_heads) below the existing defaults in src/config.py
  and run.py without owner approval.
- Removing or weakening any existing test, assertion, validation check, or
  causal-mask logic in the existing codebase.

A2. WHEN IN DOUBT, STOP AND ASK. If a phase becomes ambiguous, blocked, or
requires a design choice the roadmap does not specify, you MUST stop,
summarize the choice and your options in a BLOCKED.md file at the repo root,
and wait for owner input. Do NOT pick the "fastest" option silently.

A3. SELF-VERIFICATION IS MANDATORY AFTER EVERY UNIT OF WORK. After each
commit-sized change, you must run a self-check (see Section D) and write the
result to runs/_selfcheck/<name>.md. If the self-check fails, you must fix it
before moving on. You are NOT allowed to declare a phase complete while a
self-check is failing.

A4. HONESTY OVER OPTIMISM. If a result is bad — for example, the conditional
model loses to block bootstrap on a stylized fact — you MUST report it
truthfully. Do not cherry-pick metrics, do not hide failing rows, do not
change axes/scales to make plots look better, do not silently re-run with
different seeds until you get a favorable number. A truthful negative result
is acceptable. A flattering false result is a project-ending failure.

A5. REPRODUCIBILITY IS LAW. Every experiment must be reproducible from a
single command, a single config file, and a single seed. If
python run.py --config X --seed S cannot reproduce a number you reported,
that number does not exist.

A6. NO HIDDEN STATE. Do not introduce caching, memoization, or saved
artifacts that change behavior depending on whether files exist on disk,
unless that behavior is explicit, documented, and toggleable via a CLI flag.

═══════════════════════════════════════════════════════════════════════════
SECTION B — REPOSITORY CONTEXT (read carefully before coding)
═══════════════════════════════════════════════════════════════════════════

Repository: https://github.com/Land-Walker/FinD_Generator

Existing structure (you must respect it; do not reorganize without reason):
- run.py — orchestrator
- src/config.py — paths, tickers, FRED IDs, defaults
- src/preprocessor/data_collector.py — Yahoo + FRED ingestion
- src/preprocessor/data_loader.py — wavelet, PCA, regime labeling, splits
- src/models/timegrad_core/ — gaussian_diffusion.py, epsilon_theta.py,
  timegrad_base.py
- src/models/conditional_timegrad/ — conditional_model.py,
  conditioned_epsilon_theta.py
- src/training/training_network.py — Student-t mixin + cond wrapper
- src/predictor/prediction_network.py — autoregressive sampling
- src/scenario_generator.py — regime override
- docs/architecture.md, docs/Roadmap.md — read both before starting
- notebooks/ — keep as-is, do not delete
- checkpoints/ — has both original_timegrad_best.pt and
  conditional_timegrad_best.pt

Existing strengths you MUST NOT damage:
- The cross-attention + relative position bias + causal mask in
  conditioned_epsilon_theta.py.
- The Student-t -> Gaussian latent transform in the Mixin.
- The chronological train/val/test split.
- The PCA-fit-on-train-only behavior in data_loader.py.

Known weaknesses you ARE expected to fix in the appropriate phase:
- wavelet_denoise_series applies wavedec to the full series (look-ahead).
- .bfill() is used in merge_all_blocks_unified (look-ahead).
- Regime thresholds (vol median, infl/growth thresholds) may be computed
  before the chronological split.
- 80% coverage is reported as 0.0938 in the README — calibration is broken.
- No DDIM sampler.
- No classifier-free guidance.
- No baselines beyond vanilla TimeGrad.
- No stylized-facts evaluation.
- No regime-conditioning validation.
- requirements.txt has no pinned versions.

═══════════════════════════════════════════════════════════════════════════
SECTION C — PHASE SPECIFICATION (execute in order, do not skip ahead)
═══════════════════════════════════════════════════════════════════════════

────────────────────────────────────────────
PHASE 0 — Reproducibility Foundation
────────────────────────────────────────────
Goals:
0.1 Add src/utils/seed.py with set_global_seed(seed: int) that seeds:
    random, numpy, torch (CPU + CUDA),
    torch.backends.cudnn.deterministic=True,
    torch.backends.cudnn.benchmark=False, and sets PYTHONHASHSEED.
0.2 Add src/utils/run_folder.py that, given a run name, creates
    runs/<run_name>seed/ and inside it:
    - saves the resolved config as config.yaml
    - saves git_sha.txt (current commit) and git_diff.patch (uncommitted)
    - creates subfolders: metrics/, plots/, samples/, logs/, checkpoints/
0.3 Introduce omegaconf (NOT Hydra yet). Add configs/default.yaml mirroring
    every CLI flag in current run.py. CLI overrides must still work.
0.4 Update run.py:
    - new flags: --seed (required, no default-silent behavior), --run-name,
      --config
    - all outputs go inside the run folder, never to repo root
    - on every run, write metrics/train_log.jsonl (one JSON line per epoch
      with train_loss, val_loss, lr, wallclock, gpu_mem)
0.5 Pin requirements.txt versions. Add: omegaconf, properscoring, arch,
    statsmodels, scipy, tqdm. Do NOT add wandb unless owner asks.
0.6 Add tests/test_reproducibility.py:
    - run training for 1 epoch with seed=0 twice; assert identical loss
      trajectories bit-for-bit on CPU.

Acceptance check for Phase 0:
- pytest tests/test_reproducibility.py -v passes.
- python run.py --config configs/default.yaml --seed 0 --run-name smoke
  --epochs 1 --max-train-steps 2 --max-val-steps 2 --num-samples 1 succeeds.
- The created run folder contains all required artifacts.
- No file in the new code uses pass or TODO.

────────────────────────────────────────────
PHASE 1 — Causal Data Hygiene
────────────────────────────────────────────
Goals:
1.1 Replace wavelet_denoise_series with a causal version. Two acceptable
    implementations (pick the FIRST one unless it is empirically worse):
    (a) Rolling-window wavedec: for each t, take window [t-W+1, t], run
        wavedec on the window only, return the last reconstructed point.
        W must be >= 2 * WAVELET_LEVEL 4. Make W configurable.
    (b) Per-split wavedec: run wavedec on each of train/val/test
        independently, AFTER the chronological split, with reflection
        padding only at the window's right edge replaced by zero-padding
        (no future leak).
    The chosen path must be documented in docs/data_integrity.md.
1.2 Remove every .bfill() from the data pipeline. Where bfill was used to
    handle leading NaNs, instead either (i) drop the leading rows, or
    (ii) extend START_DATE earlier and use those rows as a discarded warmup
    buffer. Document the choice.
1.3 Move ALL regime-threshold computations (vol median, high-inflation
    threshold, low-growth threshold, etc.) to AFTER the chronological split,
    fit them on TRAIN only, and freeze them for val/test. Add explicit
    asserts that no threshold is computed from merged_raw before splitting.
1.4 Add tests/test_no_leakage.py:
    - For 100 random timestamps t in val and test, recompute every feature
      using ONLY data up to t-1 and assert it matches the pipeline's value
      at t (within float tolerance).
    - This test is mandatory; if it fails for any feature, fix the feature.
1.5 Write docs/data_integrity.md: feature-by-feature table of
    (feature, computed-from, look-ahead-safe?, evidence).

Acceptance check for Phase 1:
- pytest tests/test_no_leakage.py -v passes for every feature.
- The 80% coverage number in the new evaluation (Phase 2) is reported
  HONESTLY, whatever it is, before and after the data fix.
- docs/data_integrity.md exists and is filled in.

────────────────────────────────────────────
PHASE 2 — Research-Grade Evaluation Suite
────────────────────────────────────────────
Create src/evaluation/ with three modules. Every metric must operate on real
generated samples — never on placeholders.

2.1 forecast_metrics.py
    - crps_ensemble(forecasts, targets) using properscoring
    - mae, rmse
    - quantile_loss(forecasts, targets, alpha) for alpha in
      {0.05, 0.1, 0.5, 0.9, 0.95}
    - pit_values(forecasts, targets) and pit_ks_test(pit) returning the KS
      statistic and p-value
    - coverage(forecasts, targets, level) for level in {0.5, 0.8, 0.9, 0.95}
    - energy_score(forecasts, targets)
    - negative_log_likelihood(forecasts, targets) via empirical density
    Each function must have unit tests with synthetic Gaussian data where the
    ground-truth value is known analytically.

2.2 stylized_facts.py
    Compare REAL test-set returns vs GENERATED returns vs each baseline:
    - kurtosis, skewness
    - tail_index_hill(returns, k_frac=0.05)
    - acf_returns(returns, lags=20) (should be ~0)
    - acf_abs_returns(returns, lags=20) (volatility clustering)
    - leverage_effect(returns, lags=20) = corr(r_t, |r|_{t+k})
    - drawdown_distribution(returns)
    - realized_vol_distribution(returns, window=21)
    - var_es(returns, alpha) for alpha in {0.95, 0.99}
    Output a single comparison DataFrame with columns:
    [real, hist_bootstrap, block_bootstrap, garch_t, vanilla_timegrad,
    find_generator] and rows = each stylized fact.
    Save as metrics/stylized_facts.csv and a heatmap
    plots/stylized_facts_heatmap.png showing absolute deviation from real.

2.3 regime_validation.py
    - conditional_distribution_shift_test: for each regime g (e.g. high_vol,
      bear, stagflation), generate N=1000 samples conditioned on g and N=1000
      conditioned on its opposite. Run a two-sample KS test on
      realized_vol(samples_g) vs realized_vol(samples_not_g). Report KS stat
      + p-value + effect size (Cohen's d on log realized vol).
    - roundtrip_identifiability: train a small RandomForest (or logistic
      regression) on REAL test data to predict regime from realized vol,
      returns mean, kurtosis. Apply it to GENERATED samples. Report
      accuracy: if the model recovers the intended regime label
      significantly above chance, conditioning is real.

2.4 Wire into run.py: python run.py --eval --run-id <id> runs all three
    modules on the saved samples and writes a single
    runs/<id>/EVALUATION_REPORT.md with every table and figure embedded.

Acceptance check for Phase 2:
- All evaluation unit tests pass.
- A full evaluation report is generated for the existing
  conditional_timegrad_best.pt and original_timegrad_best.pt checkpoints.
- The report contains REAL numbers; you must verify by manually inspecting
  the CSVs and plots and writing a 1-paragraph "sanity_check" section in the
  report.
- If conditioning fails the validation tests, the report MUST say so.

────────────────────────────────────────────
PHASE 3 — Baseline Battery
────────────────────────────────────────────
Implement under src/baselines/. Each baseline must produce sample paths in
the SAME format as the diffusion models (shape
[num_samples, batch_size, prediction_length, target_dim]) and be evaluated
with the SAME suite from Phase 2.

3.1 historical_bootstrap.py
    - i.i.d. resample from training-set returns with replacement.
3.2 block_bootstrap.py
    - Stationary (Politis-Romano) block bootstrap with mean block length
      configurable (default 10). Validate: ACF(|r|) of bootstrap samples
      should be closer to real than i.i.d. bootstrap. Add unit test.
3.3 garch_baseline.py
    - GARCH(1,1) with Student-t innovations, fit per series on TRAIN only,
      using the arch package. Simulate prediction_length steps forward from
      the test history. Add unit test that simulated paths match the fitted
      unconditional variance in expectation.
3.4 vanilla_timegrad_baseline.py
    - Use existing original_timegrad_best.pt checkpoint. Wrap so it produces
      samples in the unified format. NO retraining unless owner approves.
3.5 run_baselines.py
    - Single entrypoint: python -m src.baselines.run_baselines --run-id X
    - Runs all four baselines on the test split, saves samples to
      runs/X/samples/baseline_<name>.pt, then triggers the Phase 2 evaluation
      pipeline on every method.
    - Produces runs/X/COMPARISON_TABLE.md with the canonical table:
      | Method | CRPS | QLoss(0.95) | KurtErr | |r|ACF Err | VaR99 Err |
      RegimeCtrl |
      where RegimeCtrl is "✓" only if Phase 2's regime_validation.py reports
      KS p-value < 0.01 AND roundtrip accuracy > random+10%.

Acceptance check for Phase 3:
- All four baselines produce samples without errors.
- The comparison table is generated and committed.
- If FinD_Generator does NOT win on a metric, the report says so. Do not
  alter axis ranges, sampling counts, or confidence levels to flip the
  result.

────────────────────────────────────────────
PHASE 4 — Targeted Model Improvements
────────────────────────────────────────────
Do these in order. For each sub-phase, run Phase 2+3 evaluation BEFORE and
AFTER the change and write a delta report docs/ablation.md containing the
canonical table for both runs.

4.1 DDIM sampler
    - Add ddim_sample_loop to gaussian_diffusion.py with configurable eta
      (default 0.0) and num_steps (default 20).
    - Wire --sampler {ddpm,ddim} and --ddim-steps into run.py and the
      predictor.
    - Acceptance: with ddim_steps=20, end-to-end inference is at least 3x
      faster than ddpm with diff_steps=100, AND every Phase 2 metric is
      within ±10% relative of the ddpm result. If not, document the
      discrepancy and DO NOT silently make ddim the default.
4.2 Cosine beta schedule
    - Add cosine schedule to the existing _linear_beta_schedule switch.
    - Run a head-to-head ablation. Default stays linear unless cosine wins
      on calibration (PIT KS p-value AND coverage gap to nominal).
4.3 Calibration fix for Student-t scale freeze
    - Currently prediction_network.py freezes loc/scale from the initial
      window. Implement and benchmark TWO alternatives:
      (i) horizon-dependent scale inflation:
          scale_t = scale_0 * sqrt(1 + alpha * t / horizon)
          (alpha learned or grid-searched)
      (ii) trailing-K-step scale: scale_0 = mean of scale over last K
           historical steps (K configurable).
    - Choose the variant that improves coverage at level=0.8 toward nominal
      WITHOUT degrading CRPS by more than 5% relative. Document both runs.
4.4 Classifier-Free Guidance (CFG)
    - Training: with probability p_uncond (default 0.1), zero out
      cond_static and cond_dynamic to learn an unconditional path.
    - Inference: eps = (1+w) * eps_cond - w * eps_uncond, with w configurable
      via --cfg-scale.
    - Acceptance: produce a docs/cfg_sweep.md with results for
      w ∈ {0.0, 0.5, 1.0, 2.0, 4.0}. The report must show how stylized facts
      under stress regimes change with w. Pick a default w that maximizes
      regime-control effect size WITHOUT pushing CRPS up by more than 10%.
4.5 (Optional, only if 4.1–4.4 finish under budget) Static regime embedding
    upgrade: replace one-hot regime with a small learned MLP over rule-based
    regime + regime intensity. Run ablation.

Acceptance check for Phase 4:
- For each of 4.1, 4.2, 4.3, 4.4: a docs/ablation.md exists with
  before/after canonical comparison tables, and a 1-paragraph honest
  conclusion (improved / neutral / regressed).
- docs/Roadmap.md is updated to reflect what was actually done.
- README is updated with the new headline numbers; old numbers are kept in a
  "Historical results" subsection — do NOT delete prior numbers.

═══════════════════════════════════════════════════════════════════════════
SECTION D — SELF-VERIFICATION PROTOCOL (run after every commit-sized change)
═══════════════════════════════════════════════════════════════════════════

After every meaningful change, run the following and APPEND the output to
runs/_selfcheck/<phase>__<change>.md:

D1. python -m pytest -x --tb=short
    All tests must pass. If any fail, you must fix them before continuing.
D2. python run.py --config configs/default.yaml --seed 0 --run-name smoke
    --epochs 1 --max-train-steps 2 --max-val-steps 2 --num-samples 2
    Smoke test must complete end-to-end in < 5 minutes on CPU.
D3. Determinism check:
    Run D2 twice with the same seed; diff the resulting train_log.jsonl
    files. Train losses must match to 1e-6.
D4. Lint / static check:
    python -m pyflakes src/ tests/ must report no undefined names or unused
    imports in files you authored or modified.
D5. Honesty audit (do this YOURSELF, no shortcuts):
    Open every new file you wrote and verify:
    - No pass placeholder
    - No TODO / FIXME / XXX left
    - No commented-out blocks of "real" logic
    - No bare except: or except Exception: pass
    - No random.seed(...) calls that override the global seed util
    - No hardcoded paths outside the run folder
    - Every function with a numerical return has at least one unit test
    Write the audit result as a checklist in the self-check file.
D6. Result-plausibility audit (after running an evaluation):
    Spot-check at least 3 numbers in any generated table by recomputing them
    in a small inline script. Record both numbers in the self-check file.
    They must agree.

If ANY of D1–D6 fails, STOP. Do not proceed. Either fix the failure or write
a BLOCKED.md and wait.

═══════════════════════════════════════════════════════════════════════════
SECTION E — COMMUNICATION & PROGRESS REPORTING
═══════════════════════════════════════════════════════════════════════════

E1. Maintain PROGRESS.md at repo root. After each phase, append a section:
    - What was done (file-level granularity)
    - What tests were added
    - What numbers changed and by how much (BEFORE / AFTER)
    - What surprised you, including any negative results
    - Any owner decisions still pending in BLOCKED.md
E2. Commit messages must be specific:
    GOOD: "phase1: replace wavelet_denoise_series with rolling-window
    wavedec (W=64); add test_no_leakage.py for 100 random timestamps"
    BAD: "fixes" / "wip" / "phase 1 done"
E3. Never silently change a default. Any change to a default value in
    src/config.py or configs/default.yaml must be accompanied by a
    1-sentence justification in PROGRESS.md and the prior value preserved in
    the comment.
E4. If you find a bug in EXISTING code that is outside the current phase,
    record it in KNOWN_ISSUES.md with reproduction steps. Do NOT silently
    fix it inside an unrelated phase commit.

═══════════════════════════════════════════════════════════════════════════
SECTION F — FINAL ANTI-CHEAT CLAUSE
═══════════════════════════════════════════════════════════════════════════

The following behaviors will be considered a critical failure of this task
even if every test passes:
- Marking a phase "complete" while any acceptance check is unmet.
- Reporting a metric value not produced by the code in this repository.
- Silently reducing the scope (e.g., running baselines on train data instead
  of test data, or evaluating on a 100-row subset to "check the pipeline").
- Claiming an improvement without a before/after comparison run on the same
  seed and same data.
- Removing or weakening any test, assertion, or causal-mask check.
- Writing a self-check file that says "ok" while a test is failing.

If you find yourself tempted to do any of the above because of time
pressure, write BLOCKED.md and stop instead.

═══════════════════════════════════════════════════════════════════════════
SECTION G — START INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════════

1. Read src/, docs/architecture.md, docs/Roadmap.md, and run.py end-to-end
   before writing any code.
2. Create a top-level PROGRESS.md and add an initial entry summarizing your
   understanding of the repo and your planned execution order.
3. Begin Phase 0. Do not start Phase 1 until Phase 0's acceptance checks all
   pass and self-verification D1–D6 are clean.
4. Continue sequentially through Phase 4. Phase 4.5 is optional and only
   permitted if 4.1–4.4 finish under the time you originally estimated.
5. When all four phases are complete, produce FINAL_REPORT.md at the repo
   root containing:
   - Canonical comparison table (real vs all baselines vs FinD_Generator,
     before-Phase-4 and after-Phase-4)
   - Stylized-facts heatmap
   - Regime-validation table
   - Honest list of what improved, what didn't, and what is still open

You are now beginning. Acknowledge by writing the initial PROGRESS.md
