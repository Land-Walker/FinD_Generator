# KNOWN ISSUES (E4 log)

Issues discovered outside (or beyond the minimum scope of) the current phase.
Fixed items name the fixing commit; open items await owner decision.

## 1. Quarterly macro block was silently empty — FIXED (da650f6, Phase 1)
- Repro (pre-fix): `align_and_handle_missing_values` suffixed quarterly columns
  (`gdp` → `gdp_quarterly`), then `process_quarterly_macro_raw` looked for
  `'gdp' in df.columns` → returned a 0-column frame. Consequences: no
  `gdp_yoy`/`gov_fiscal_balance_to_gdp`/`quarterly_pca_*` features, and
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

## 3. Raw data span contradicts config — OPEN (owner decision)
- `data/raw/*.parquet` cover **1992-01 → 2019-12**, while `src/config.py`
  declares `START_DATE=2000-01-01`, `END_DATE=2023-12-31` (and the README/
  Roadmap reference 2000–2023 crash events). All current results, and the
  existing checkpoints, are fitted on 1992–2019. Re-downloading would change
  the dataset under every existing number. Decision needed: keep 1992–2019
  (document everywhere) vs re-download 2000–2023 (invalidates checkpoints).

## 4. `gdp_yoy` is mislabeled and mostly zero — OPEN
- `process_quarterly_macro_raw` computes `log_growth` (1-step log diff) on a
  quarterly series that was already forward-filled to monthly frequency, so
  the value is 0 for ~2 of every 3 months and a QoQ (not YoY) growth at
  quarter boundaries. With `low_growth=0.0`, "recession" labels fire only in
  boundary months of shrinking quarters (233 of 7239 rows). Renaming/
  redesigning would change the modeling problem; deferred to owner.

## 5. `high_infl=0.03` never triggers on monthly CPI — OPEN
- `cpi_mom` (month-over-month) never exceeds 3% in 1992–2019, so
  `high_inflation` and `stagflation` regimes have ZERO support in the data
  (constants are frozen by W2.6). Phase 2's macro-regime conditioning
  validation will necessarily report this; scenario overrides can still force
  these one-hots at inference, but the model never saw them in training.

## 6. Existing checkpoints are incompatible with the fixed pipeline — OPEN
- `checkpoints/*_best.pt` were trained on the legacy feature layout
  (cond_dynamic 22, cond_static 6, horizon 24, target_dim 1) matching
  `data/processed/*_processed.csv`. The Phase 1 pipeline produces
  cond_dynamic 25, cond_static 10. The checkpoints can only be evaluated on
  the preserved legacy frames; comparing them against the fixed pipeline
  requires retraining (forbidden without owner approval, W0.6).

## 7. Pre-existing lint noise in untouched files — OPEN (cosmetic)
- `src/__init__.py` re-export false positives (pyflakes).
