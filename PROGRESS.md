# PROGRESS

> Unpushed commits (W1.2): `68f79cc` (W0 bootstrap) — push failed: no GitHub credentials available in this environment. Owner must push manually or provide credentials.

## 2026-06-11 — W0 Environment bootstrap (Cowork master directive v3)

### W0.1 / W0.2 Repo state
- Working folder contains the FinD_Generator repo: `run.py`, `src/`, `docs/Roadmap.md`, `docs/architecture.md` all present.
- `src/utils/seed.py` and `configs/default.yaml` do NOT exist → this is the **pre-Phase-0 original**. Proceeding with Phase 0 per W0.2.
- Note: remote branch `codex/initialize-progress.md-and-outline-plan` contains an earlier (unmerged) PROGRESS.md plan from 2026-05-02 with no code changes. It is superseded by this file.

### W0.3 Git
- Git repo on branch `main`, remote `origin = https://github.com/Land-Walker/FinD_Generator.git`, HEAD `820d66a`.
- Remote is reachable (read). Push authentication not yet verified — will be tested on the first push; if push fails, unpushed SHAs will be recorded at the top of this file per W1.2.
- Sandbox housekeeping: working tree had pure CRLF/LF noise (Windows folder mounted into Linux); fixed with `core.autocrlf=true` + index rebuild. A stale `.git/index.lock` was removed after enabling file deletion in the sandbox. Working tree is clean at `820d66a`.

### W0.4 Environment capability
- Python 3.10.12, pip 25.3, network egress to PyPI works.
- `download.pytorch.org` and `conda.anaconda.org` are **blocked** (HTTP 403) by the sandbox network policy, so the CPU-only torch build was unavailable. Installed standard PyPI `torch 2.12.0+cu130` plus the required NVIDIA runtime libraries manually (~2.2 GB); `import torch` and tensor ops verified.
- **CUDA is NOT available** (`torch.cuda.is_available() == False`). Hardware: 2 CPU cores, ~4 GB RAM.
  - ⚠ Consequence (per W0.4): Phase 4.4 (CFG retraining) and any full retraining will be CPU-bound on 2 cores. Feasibility must be estimated before starting those phases; if a required training run would exceed ~6 h on CPU, BLOCKED.md will be written instead of silently running it.
- All other requirements installed and importable: pandas 2.3.3, numpy 2.2.6, scikit-learn 1.7.2, PyWavelets, statsmodels, scipy, yfinance, pandas-datareader, fastparquet, pyarrow, matplotlib, seaborn, joblib.
- Per W2.1/W2.3: `omegaconf 2.3.0` and `pyflakes` installed (hard dependency / self-check tool respectively).

### W0.6 Checkpoint inventory
- `checkpoints/original_timegrad_best.pt` — present ✓
- `checkpoints/conditional_timegrad_best.pt` — present ✓
- `data/processed/timegrad_checkpoint.pt` — present (legacy location, not referenced by the directive).
- Both checkpoints required by Phase 2 acceptance and Phase 3.4 exist → no BLOCKED entry needed.

### W0.7 Data inventory
- All five raw parquet files present under `data/raw/`: `target.parquet`, `market.parquet`, `daily_macro.parquet`, `monthly_macro.parquet`, `quarterly_macro.parquet` ✓
- `data/processed/{train,val,test}_processed.csv` also present (legacy outputs).

### Open item before Phase 0 starts
- The master directive layers on top of an "original Phase 0→4 specification (Sections A–G)" (Section C 0.1–0.6 / 1.1–1.5 step definitions, self-checks D1–D6, commit style E2, BLOCKED rules A2, anti-cheat F). That document is **not in the repo and was not provided in this session**; only a high-level outline of it survives in the codex branch PROGRESS.md. Owner has been asked to either supply the original spec or authorize execution from the directive + outline alone. This is recorded here rather than in BLOCKED.md because work can proceed the moment the owner answers.

### Execution plan
1. **Phase 0 — Reproducibility Foundation**: seeding utility (`src/utils/seed.py`), omegaconf config (`configs/default.yaml`, hard dependency, fail loudly), run-folder artifact management under `runs/`, train-log persistence (`metrics/train_log.jsonl`), pinned requirements, determinism test with pasted same-seed diff evidence (W2.4), pyflakes in self-check (W2.3), known bug fixes per W2.5.
2. **Phase 1 — Causal Data Hygiene**: remove look-ahead leakage (wavelet, missing-value handling: zero backward-fill per W2.2, `.ffill()` only per W2.7), train-only fit of the roll_vol median regime threshold (W2.6), `test_no_leakage.py` covering EVERY engineered feature with the list enumerated in `docs/data_integrity.md`, honest before/after 80%-coverage numbers.
3. **★ MANDATORY STOP (W3.3)** — handoff in PROGRESS.md; wait for explicit owner approval.
4. After approval: Phase 2 (regime_validation headline, effect sizes), Phase 3 (all four baselines, canonical table), Phase 4 rescoped (4.3 calibration MANDATORY, 4.4 CFG MANDATORY, 4.1 DDIM optional, 4.2/4.5 CUT), Phase 5 stress-testing showcase (`src/stress_demo/`, README rewrite marked DRAFT), FINAL_REPORT.md + 10-line owner action list.

### Positioning (governs all reports)
FinD_Generator is a regime-conditional stress scenario generator. Headline result = controllable regime conditioning (`regime_validation`), not beating classical baselines on unconditional stylized facts. Baseline wins on uncon