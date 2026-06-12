# Data Integrity — Causal Feature Audit (Phase 1.5)

Verified by `tests/test_no_leakage.py`: for **100 random timestamps t in the
val and test splits** (seeded RNG, 50 each), every feature below is
**independently recomputed from raw inputs truncated at t** and compared to
the pipeline's value at t (rtol 1e-7 / atol 1e-10). Any dependence on data
after t would break the equality. Test interpretation note: a feature value
*at* t may legitimately use data dated ≤ t (e.g. a return at t uses the
close at t and t-1); "no look-ahead" means no data dated **after** t.

## Phase 1 design choices
- **1.1 Causal wavelet — option (a), rolling window.** For each t,
  `wavedec` runs on the backward window `[t-W+1, t]` only (W =
  `WAVELET_WINDOW` = 64 ≥ 2³·4 = 32, configurable); the reconstructed last
  point is kept. First W−1 outputs are NaN (warmup). Option (b) was not
  needed: (a) is causal by construction and adds only ~2 s to the full
  pipeline build.
- **1.2 bfill removal — choice (i), drop leading rows.** After merging, all
  rows before the first date on which every column has a (forward-filled)
  value are dropped (warmup buffer: wavelet warmup + first quarterly
  publication). Any NaN surviving the drop raises (`_ffill_checked`); the
  legacy `fillna(0)` constant-fills were removed.
- **1.3 Train-only thresholds.** The only data-dependent regime threshold —
  the roll_vol median — is fitted on the TRAIN slice exclusively
  (`_label_regimes`), frozen for val/test, stored in
  `dm.regime_thresholds`, and guarded by asserts (fit boundary must equal
  the split boundary; threshold value re-derived from the train slice).
  Fixed constants r_thresh=0.02, v_thresh=20, high_infl=0.03,
  low_growth=0.0 are not data-dependent (W2.6). Train median 0.008680 vs
  leaky full-sample median 0.007944 — the legacy leak shifted labels.

## Alignment rules (shared by several features)
- Grid: business days spanning the target's trading range; leading rows
  dropped per 1.2. Values enter the grid by forward-fill from dates ≤ t.
- Monthly/quarterly data: observation for month m is mapped to the
  calendar month-end of m, then forward-filled onto the grid
  (`reindex(grid, method="ffill")` — weekend month-ends carried, not
  dropped; see KNOWN_ISSUES #8).
- Quarterly raw is first resampled to month-start frequency with ffill.

## Feature table

| # | Feature | Computed from | Look-ahead safe? | Evidence |
|---|---------|--------------|------------------|----------|
| 1–4 | `open/high/low/close_den` | causal wavelet (W=64) over target OHLC ≤ t | YES | test_no_leakage (per-window kernel recompute); test_causal_wavelet (future-mutation invariance) |
| 5 | `target_volume_raw` | target volume, last value ≤ t | YES | test_no_leakage |
| 6–9 | `open/high/low/close_ret` | log return of market OHLC at last trading day ≤ t | YES | test_no_leakage |
| 10 | `market_volume_raw` | log1p market volume, last value ≤ t | YES | test_no_leakage |
| 11 | `vix_daily` | raw VIX, last value ≤ t | YES | test_no_leakage |
| 12 | `yield_curve_daily` | raw DGS10, last value ≤ t | YES | test_no_leakage |
| 13 | `cpi_mom` | CPI pct-change at last published month-end ≤ t | YES | test_no_leakage |
| 14 | `unemployment_detrend` | unemployment − rolling-12 mean, months ≤ t | YES | test_no_leakage |
| 15 | `interest_rate_diff` | FEDFUNDS first difference, months ≤ t | YES | test_no_leakage |
| 16 | `trade_balance_seasdiff` | trade balance − 12-month lag, months ≤ t | YES | test_no_leakage |
| 17 | `gdp_yoy` | log diff of MS-resampled GDP, months ≤ t (semantics caveat: KNOWN_ISSUES #4) | YES | test_no_leakage |
| 18 | `gov_fiscal_balance_to_gdp` | fiscal balance / GDP, months ≤ t | YES | test_no_leakage |
| 19–21 | `gov_debt`, `tax_receipts`, `gov_spending` | passthrough, last published value ≤ t | YES | test_no_leakage |
| 22–33 | calendar (`day_of_week, month, quarter, year, is_month_end, is_quarter_end, month_sin/cos, dow_sin/cos, quarter_sin/cos`) | pure functions of the date t (B-frequency convention for *_end) | YES (deterministic) | test_no_leakage |
| 34 | `market_close` | market close, last trading day ≤ t | YES | test_no_leakage |
| 35–37 | `market_regime_{bear,bull,sideways}` | roll_return/roll_vol over last 30 grid rows ≤ t + **train-only** median | YES | test_no_leakage (labels recomputed with frozen threshold); test_threshold_is_train_only |
| 38–39 | `vol_regime_{high_vol,normal_vol}` | VIX at t vs fixed 20 | YES | test_no_leakage |
| 40–44 | `macro_regime_{expansion,high_inflation,normal,recession,stagflation}` | cpi_mom/gdp_yoy ≤ t vs fixed constants | YES | test_no_leakage (note: high_inflation/stagflation have zero support 1992–2019, KNOWN_ISSUES #5) |
| 45 | `target_pca_1` | train-fitted scaler+PCA applied to features 1–4 at t | YES (transform fitted on train < t) | test_no_leakage |
| 46–48 | `market_pca_1..3` | train-fitted scaler+PCA on features 6–9 at t | YES | test_no_leakage |
| 49–50 | `daily_vix_daily_scaled`, `daily_yield_curve_daily_scaled` | train-fitted scaler on 11–12 at t | YES | test_no_leakage |
| 51 | `volume_scaled` | train-fitted scaler on 10 at t | YES | test_no_leakage |
| 52–55 | `monthly_pca_1..4` | train-fitted scaler+PCA on 13–16 at t | YES | test_no_leakage |
| 56–58 | `quarterly_pca_1..3` | train-fitted scaler+PCA on 17–21 at t | YES | test_no_leakage |

(58 audited values; 44 merged-frame features + 14 transformed model inputs.
The model consumes: target_pca_1 (target), 46–51 + 22–33 (dynamic daily),
52–58 (dynamic monthly), 35–44 (static regimes).)

## What changed vs the legacy pipeline (look-ahead removed)
1. Wavelet denoising saw the FULL series (every `_den` value depended on the
   future) → rolling causal window.
2. `bfill` in target/market grid alignment and the final merge `bfill()`
   copied future values into leading rows → leading-row drop.
3. roll_vol median for market regimes was computed over the full sample
   (train labels depended on val/test volatility) → train-only, frozen.
4. (Alignment, not look-ahead): weekend month-end macro observations were
   dropped; duplicate `volume_raw` columns interleaved; quarterly block was
   empty. See KNOWN_ISSUES 1/2/8.

## 80% coverage before/after (Phase 1 acceptance, honest numbers)
- **BEFORE the fix** (existing `conditional_timegrad_best.pt` on the legacy
  test frame, 64 windows × 16 samples, seed 0): **0.1165** vs nominal 0.80
  (README's historical protocol reported 0.0938). Artifact:
  `runs/coverage_probe_before/coverage_report.json`.
- **AFTER the fix**: NOT MEASURABLE with the existing checkpoints — the
  repaired pipeline produces cond_dynamic=25 / cond_static=10 features vs
  the checkpoint's 22/6 (KNOWN_ISSUES #6). An after-fix coverage number
  requires retraining, which needs owner approval (W0.6). The Phase 2
  evaluation will report coverage for whatever model/data pairing the owner
  approves; the calibration defect itself (scale freeze) is Phase 4.3's
  target.
