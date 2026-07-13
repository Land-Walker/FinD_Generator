# AUDIT_PHASE5.md — Independent Audit of Phase 5 Deliverables

**Auditor**: Independent, adversarial review. No diplomatic softening.
**Scope**: Phase 5 code additions, README draft, COMPARISON_TABLE.md, stress demo.
**Method**: Read all committed JSONs, run all tests, trace every number, inspect code.

---

## VERDICT TABLE

| Check | Verdict | Key Evidence |
|-------|---------|-------------|
| A1  | PASS | 87/87 tests pass (pytest -x --tb=short) |
| A2  | PASS | 10/10 portfolio_stress unit tests pass |
| B1  | PASS | Every README number traced to committed JSON; no drift beyond rounding |
| B2  | PASS | 3 independent recomputations match JSONs |
| B3  | **FAIL** | Git-tracked per-folder COMPARISON_TABLE.md still has TBD rows (vanilla=all-TBD in 2 of 3, baselines=all-TBD in 2 of 3). The unified table was regenerated only into a gitignored runs/ path — not committed anywhere visible. |
| C1  | PASS | All 6 required limitations stated plainly in README lines 128–162 |
| C2  | PASS | No spin detected; losses stated honestly; CFG trade documented |
| D1  | PASS | Per-path drawdown confirmed (3D→2D reshape, no mega-path 1D flatten) |
| D2  | PASS | VaR/ES on terminal log-return sum, consistent with canonical space |
| D3  | PASS | Stress demo runs end-to-end (synthetic data), produces fan chart + VaR table |
| D4  | PASS | README labels stress demo as "CPU, test scale"; no numbers presented |
| E1  | PASS | No silenced exceptions, stubs, or fabricated numbers in new code |
| E2  | PASS | Commits per-unit, specific messages, pushed, git status clean |
| F1  | **FAIL** | Unified COMPARISON_TABLE.md exists only in gitignored runs/. No copy at repo root or docs/. cfg_sweep.md committed but only in runs/ subfolder. Stress fan chart / VaR table not committed. |
| F2  | PASS | README contains all key numbers inline (comparison table, regime validation, CFG dial); no clicking required |
| F3  | PASS | Sweep vs full-eval clearly labeled: lines 26–28, 62–64, column header "bear d (sweep)" |

**Total**: 14 PASS, 2 FAIL, 0 CANNOT-VERIFY

---

## DETAILED FINDINGS

### A1 — Full test suite

```
87 passed in 86.16s
```

All tests pass. No flaky tests, no skips. Includes new `test_portfolio_stress.py` (10 tests).

### A2 — Stress demo unit tests

```
tests/test_portfolio_stress.py::TestExamplePortfolio::test_single_asset_portfolio_returns_match_input PASSED
tests/test_portfolio_stress.py::TestExamplePortfolio::test_paths_shape PASSED
tests/test_portfolio_stress.py::TestExamplePortfolio::test_3d_returns_path_shape PASSED
tests/test_portfolio_stress.py::TestExamplePortfolio::test_var_es_known_synthetic PASSED
tests/test_portfolio_stress.py::TestExamplePortfolio::test_drawdown_all_positive PASSED
tests/test_portfolio_stress.py::TestExamplePortfolio::test_drawdown_negative PASSED
tests/test_portfolio_stress.py::TestComputeStressComparison::test_all_scenarios PASSED
tests/test_portfolio_stress.py::TestVarES::test_var_95_known PASSED
tests/test_portfolio_stress.py::TestVarES::test_var_99_known PASSED
tests/test_portfolio_stress.py::TestVarES::test_edge_case_all_same PASSED
```

10/10. Synthetic-based, covers edge cases.

### B1 — Number integrity: full cross-check

Every number in README.md traced to a committed metrics JSON. Rounding at display precision is correct (4 decimal places → 4 sig figs in table). Full mapping:

| README value | Source JSON | JSON value | Match? |
|---|---|---|---|
| conditional CRPS 0.0043 | retrain_d1/forecast_metrics.json | 0.004331 | ✓ |
| conditional cov 0.630 | retrain_d1/forecast_metrics.json | 0.6296 | ✓ |
| conditional kurt 3.36 | retrain_d1/stylized_facts.json (generated) | 3.3576 | ✓ |
| conditional bear d 0.70 | retrain_d1/regime_validation.json | 0.697 | ✓ |
| recession p 0.164, "No" | retrain_d1/regime_validation.json | 0.1641 | ✓ |
| CFG w=2 CRPS 0.0039 | cfg_eval_w2/forecast_metrics.json | 0.003888 | ✓ |
| CFG w=2 cov 0.658 | cfg_eval_w2/forecast_metrics.json | 0.6580 | ✓ |
| CFG w=2 kurt 4.58 | cfg_eval_w2/stylized_facts.json (generated) | 4.5756 | ✓ |
| CFG w=2 bear d 0.87 | cfg_eval_w2/regime_validation.json | 0.872 | ✓ |
| CFG w=4 CRPS 0.0042 | cfg_eval_w4/forecast_metrics.json | 0.004187 | ✓ |
| CFG w=4 cov 0.538 | cfg_eval_w4/forecast_metrics.json | 0.5376 | ✓ |
| CFG w=4 kurt 8.20 | cfg_eval_w4/stylized_facts.json (generated) | 8.2035 | ✓ |
| CFG w=4 bear d 1.21 | cfg_eval_w4/regime_validation.json | 1.206 | ✓ |
| CFG w=4 recession d 0.148 | cfg_eval_w4/regime_validation.json | 0.148 | ✓ |
| CFG w=4 recession p 0.0045 | cfg_eval_w4/regime_validation.json | 0.0045 | ✓ |
| vanilla CRPS 0.0035 | vanilla_eval/forecast_metrics.json | 0.003462 | ✓ |
| vanilla cov 0.706 | vanilla_eval/forecast_metrics.json | 0.7062 | ✓ |
| vanilla kurt 14.83 | vanilla_eval/stylized_facts.json (generated) | 14.8279 | ✓ |
| hist_boot CRPS 0.0034 | retrain_d1/baseline_hist_boot.json | 0.003376 | ✓ |
| block_boot CRPS 0.0034 | retrain_d1/baseline_block_boot.json | 0.003382 | ✓ |
| garch_t CRPS 0.0039 | retrain_d1/baseline_garch_t.json | 0.003863 | ✓ |
| garch_t kurt 327.09 | retrain_d1/baseline_garch_t.json | 327.0903 | ✓ |
| real raw kurt 2.17 | retrain_d1/stylized_facts.json (real_raw) | 2.1744 | ✓ |
| real denoised kurt 3.79 | retrain_d1/stylized_facts.json (real_denoised) | 3.7880 | ✓ |

CFG sweep table (lines 30-36) — all 15 values verified against `cfg_sweep.json`:
- w=0.0: bear -0.031, rec +0.033, p=0.5006 → ✓
- w=0.5: bear +0.162, rec -0.029, p=0.7228 → ✓
- w=1.0: bear +0.521, rec -0.058, p=0.0224 → ✓
- w=2.0: bear +0.947, rec +0.020, p=0.1812 → ✓
- w=4.0: bear +1.083, rec +0.075, p=0.0224 → ✓

Regime validation table (lines 69-79) — all 9 rows verified against retrain_d1/regime_validation.json. Rounded values (e.g., 0.70 for 0.697) are within acceptable display precision.

**Result: No fabricated numbers. All values trace to committed JSONs.**

### B2 — Independent recomputation

Three numbers recomputed by reading raw JSONs directly (not parsing the table):

1. Conditional CRPS: 0.004331 (vs README 0.0043)
2. Vanilla coverage_0.8: 0.7062 (vs README 0.706)
3. Real raw kurtosis: 2.1744 (vs README 2.17)

All match within rounding. The real_denoised kurtosis (3.7880) referenced in Limitations §4 matches JSON.

### B3 — TBD rows in committed comparison tables

**FAIL**. The git-tracked per-folder COMPARISON_TABLE.md files still contain TBD rows:

- `runs/cfg_eval_w2__20260713-212242__seed0/COMPARISON_TABLE.md`: vanilla=TBD(×10), hist_boot=TBD, block_boot=TBD, garch_t=TBD, real CRPS=TBD.
- `runs/cfg_eval_w4__20260713-214944__seed0/COMPARISON_TABLE.md`: same TBD pattern.
- `runs/retrain_d1__20260706-212108__seed0/COMPARISON_TABLE.md`: This local copy WAS regenerated by the multi-folder assembler (all rows filled). BUT it is gitignored — the committed version is the OLD per-folder table.

The `scripts/assemble_comparison.py` script was written and committed, and it correctly produces the unified table. The output file was written to the gitignored `runs/` directory. The unified table was NOT committed to the repo root or `docs/`.

**No committed file anywhere in the repo contains the complete 8-row canonical comparison table that the README references as "COMPARISON_TABLE.md".**

### C1 — Honesty limitations

All 6 required limitations present in README lines 128–162:

1. **"Bootstrap and vanilla beat this model on unconditional metrics"** — lines 130-135.
   Quotes specific CRPS and coverage numbers.

2. **"Recession is underpowered (108 rows, 1.7% of data)"** — lines 137-139.
   States it's the only non-significant regime and only reaches marginal significance at w=4.

3. **"Stagflation is severely underpowered (42 rows, 0.67% of data)"** — lines 141-143.
   Notes it is significant despite low support, but "the tiny training support means the regime embedding is noisy and unreliable."

4. **"The model learns denoised targets"** — lines 145-150.
   States kurtosis differs from raw returns; provides both references (real_denoised=3.79, real_raw=2.17) and notes CFG w=4 overshoots even the denoised reference.

5. **"CFG w=4 trades calibration for regime control"** — lines 152-156.
   Coverage drops "from 0.66 to 0.54 — far below nominal 0.80."

6. **"5-step horizon limits max drawdown duration"** — lines 158-162.
   Notes generated drawdown stats are not comparable to raw-real row's multi-year drawdown.

All 5 core points from the audit spec (C1 bullets) are present. No limitation buried in footnotes.

### C2 — Spin check

No detected spin:
- "Bootstrap and vanilla beat" (line 130) uses the verb "beat."
- "CFG w=4 overshoots even the denoised reference" (lines 149-150) — admits failure.
- "coverage drops from 0.66 to 0.54 — far below nominal 0.80" (line 153) — explicit about underperformance.
- The regime-control trade explanation (lines 134-135) is measured: "The conditional model exists to provide targeted regime control, not to win on unconditional sharpness."
- The limitation about 5-step horizon is disclosed, not buried.

### D1 — Drawdown correctness

**PASS**. The mega-path bug does NOT recur. Evidence:

In `portfolio_stress.py` `compute_risk_metrics`:
```python
if scenario_returns.ndim == 3:
    dd_input = scenario_returns.reshape(-1, scenario_returns.shape[-1])
```
This reshapes 3D `(n_scenarios, n_windows, n_steps)` into 2D `(n_scenarios * n_windows, n_steps)`, treating each (scenario, window) as an independent 5-step price path. The mega-path concatenation bug (flattening to 1D → 350k+ continuous steps → -100% drawdown) is absent.

Confirmed by independent test: per-path maxDD = -0.045 (5-step), mega-path would be -0.342 (352-step). The difference is 7.5×.

### D2 — VaR/ES canonical space

**PASS**. `compute_risk_metrics` computes `terminal_rets = np.sum(scenario_returns, axis=1)` (cumulative log return over the horizon), then calls `var_es(terminal_rets, alpha)`. This is consistent with the entire evaluation pipeline (all metrics in denoised-close log-return space).

### D3 — Stress demo end-to-end

**PASS**. The stress demo runs on CPU with synthetic data (n=10 scenarios, 3 windows, 4 steps):
- Produces `stress_var_table.md` with 3 rows (unconditional, stress, historical)
- Produces `stress_fan_chart.png` (147KB, dual-panel matplotlib figure)
- The read checkpoint path uses `cfg_smoke_cond__20260708-124243__seed0/checkpoints/model_best.pt`, which exists

### D4 — Stress demo labeling

**PASS**. README §83 says "### Quick usage (CPU, test scale)". No specific risk numbers are presented in the stress demo section — only usage commands. Full-GPU runs delegated to `HOST_TASKS.md` §5b-5c.

### E1 — Anti-cheat scan

- No `except: pass` or `except Exception:` without re-raise in new files.
- `NotImplementedError`: 0 occurrences.
- `TODO`/`FIXME`/`XXX`: 0 occurrences in new files. (Pre-existing "TODO — Phase 3" in `_write_evaluation_report()` not from Phase 5.)
- Hardcoded metric values in source code: 0 occurrences (one false-positive regex match on `.3f` format specifier).

### E2 — Git hygiene

```
af68dff phase5: stress demo, multi-folder comparison table, README draft
```

Descriptive commit. Pushed. Only unstaged change is the gitignored `runs/retrain_d1_.../COMPARISON_TABLE.md`.

### F1 — Deliverable location (FAIL)

**FAIL — two sub-issues:**

1. **Unified COMPARISON_TABLE.md**: The unified table (8-row canonical comparison) was generated by `scripts/assemble_comparison.py` into `runs/retrain_d1__20260706-212108__seed0/COMPARISON_TABLE.md`, which is in the gitignored `runs/` directory. It was NOT committed to repo root or `docs/`. The older per-folder tables ARE git-tracked in individual run folders, but they contain TBD rows for vanilla and baselines (see B3).

2. **Stress demo artifacts**: `stress_fan_chart.png` and `stress_var_table.md` are not committed anywhere. They exist only in local `runs/stress_demo_test_phase5*/` directories (not git-tracked).

The README is effectively the only visible place with the canonical table — it contains the numbers inline. A reader who clones the repo and opens README.md sees the table. But `COMPARISON_TABLE.md` as a standalone committed file does not exist at a visible path.

### F2 — README self-containment

**PASS**. The README contains:
- Comparison table (8 rows, lines 46-55) inline
- Regime validation table (9 rows, lines 69-79) inline
- CFG dial trend (5 rows, lines 30-36) inline
- Reproducibility section listing all source run folders (lines 175-181)

A GitHub reader sees all key numbers without clicking into `runs/`.

### F3 — Sweep vs full-eval labeling

**PASS**. Three distinct mechanisms prevent confusion:

1. **Column header labeling**: The CFG dial table column says "bear d (sweep)" and "recession d (sweep)" — explicitly tagged (line 30).
2. **Parenthetical caveat**: "(32-window plumbing sweep — d-values here differ from the full-eval runs below because window counts differ; the sweep is used ONLY for the monotonic-trend story, not for headline effect sizes)" — lines 26-28.
3. **Source attribution**: "The CFG bear-d headline values (0.87, 1.21) come from the full-eval runs cfg_eval_w2 / cfg_eval_w4; the sweep table above uses plumbing-scale runs with fewer windows — values are NOT interchangeable." — lines 62-64.

No instance found where sweep numbers are presented as full-eval or vice versa without distinction.

---

## MOST SERIOUS PROBLEM

**F1 — COMPARISON_TABLE.md is not committed to a visible path.** The unified 8-row canonical comparison table exists only in the gitignored `runs/` directory. The speaker's own instructions say this is WRONG: "the comparison table is the headline deliverable a recruiter/interviewer must see on GitHub." While the README contains the table inline, `COMPARISON_TABLE.md` as a standalone committed markdown file at repo root or `docs/` does not exist. The `scripts/assemble_comparison.py` script is committed and correct — but its output was never committed to a visible location.

**Fix**: Run `scripts/assemble_comparison.py` with `--out-dir` pointing to `docs/` or the repo root, and commit the resulting `COMPARISON_TABLE.md`.
