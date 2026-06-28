# KNOWN ISSUES (E4 log)

Issues discovered outside (or beyond the minimum scope of) the current phase.
Fixed items name the fixing commit; open items await owner decision.

## 1. Quarterly macro block was silently empty — FIXED (da650f6, Phase 1)
- Repro (pre-fix): `align_and_handle_missing_values` suffixed quarterly columns
  (`gdp` → `gdp_quarterly`), then `process_quarterly_macro_raw` looked for
  `'gdp' in df.columns` → returned a 0-column frame. Consequences: no
  `gdp_qoq`/`gov_fiscal_balance_to_gdp`/`quarterly_pca_*` features, and
  `label_macro_regimes` saw no gdp series → every row labeled `normal`.
- Evidence: legacy `data/processed/*_processed.csv` contain no quarterly
  features and only `macro_regime_normal`.

## 2. Duplicate `volume_raw` columns interleaved — FIXED (cf2dc39, Phase 1)
- Repro (pre-fix): both target and market blocks emitted a column named
  `volume_raw`; `df['volume_raw'].to_numpy().reshape(-1, 1)` row-major
  flattened BOTH columns, the scaler was fit on the interleaved vector, and
  `vol_scaled[:len(df)]` kept the first half — so `volume_scaled[i]` held the
  volume of day `i//2` (target volume for even i, market volume for odd i).
  Past-only (no look-ahead), but incoherent. Columns renamed
  `target_volume_raw` / `market_volume_raw`; scaler now uses market volume.

## 3. Raw data span contradicts config — FIXED (D-②, D-③)
- `data/raw/*.parquet` have been re-downloaded for **2000-01 → 2024-12**,
  matching `src/config.py` START_DATE/END_DATE. All six parquets verified.
  Existing checkpoints invalidated (KNOW_ISSUES #6 remains).

## 4. `gdp_yoy` is mislabeled and mostly zero — FIXED (D-③, pending push)
- `process_quarterly_macro_raw` computed `log_growth` (1-step log diff) on a
  quarterly series that was already forward-filled to monthly frequency, so
  the value was 0 for ~2 of every 3 months and a QoQ (not YoY) growth at
  quarter boundaries. Renamed `gdp_yoy` → `gdp_qoq` to reflect actual semantics.
  The QoQ-negative recession logic is correct and unchanged.

## 5. `high_infl=0.03` never triggers on monthly CPI — FIXED (D-③, pending push)
- `cpi_mom` (month-over-month) never exceeds 3% in 1992–2019, so
  `high_inflation` and `stagflation` regimes had ZERO support in the old data.
- Fixed by switching the macro-regime inflation test from `cpi_mom` to
  `cpi_yoy` (trailing-12-month CPI). On the extended 2000–2024 data, this
  now fires during the 2021–22 inflation spike:
  high_inflation=30.06%, stagflation=0.67% of 6305 merged rows.
  Constants remain frozen (high_infl=0.03, low_growth=0.0 per W2.6).

## 6. Existing checkpoints are incompatible with the fixed pipeline — OPEN
- `checkpoints/*_best.pt` were trained on the legacy feature layout
  (cond_dynamic 22, cond_static 6, horizon 24, target_dim 1) matching
  `data/processed/*_processed.csv`. The Phase 1 pipeline produces
  cond_dynamic 25, cond_static 10. The checkpoints can only be evaluated on
  the preserved legacy frames; comparing them against the fixed pipeline
  requires retraining (forbidden without owner approval, W0.6).

## 7. Pre-existing lint noise in untouched files — OPEN (cosmetic)
- `src/__init__.py` re-export false positives (pyflakes).

## 8. Weekend month-end observations were silently dropped — FIXED (Phase 1)
- Repro (pre-fix): monthly/quarterly values are mapped to calendar month-ends
  before reindexing onto the business-day grid; `.reindex(daily_index).ffill()`
  discards any value whose month-end is a Saturday/Sunday (not in the grid),
  so those months kept showing the PREVIOUS month's data (e.g. Feb-2015 rows
  carried December-2014 macro values). Discovered by tests/test_no_leakage.py.
- Fix: `.reindex(daily_index, method="ffill")` — still past-only, no data loss.
