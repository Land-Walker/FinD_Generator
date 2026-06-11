# PROGRESS

> Unpushed commits (W1.2): **every commit on `main` after `820d66a`** is local-only (no GitHub credentials in the sandbox; owner pushes manually at review gates — owner-confirmed policy). Run `git log --oneline 820d66a..main` for the exact list. As of the last update: `68f79cc`, `a76f0bf`, `8e03abb`, `59d0551`, plus any later Phase 0+ commits.

## 2026-06-11 — W0 Environment bootstrap (Cowork master directive v3)

### Prior attempt (historical record)
A previous sandbox session implemented Phase 0 (per its own PROGRESS.md: `src/utils/seed.py`, `src/utils/run_folder.py`, `configs/default.yaml`, `src/utils/config_io.py`, `tests/test_reproducibility.py`, modified `run.py`, pinned requirements) but that work was **never pushed and is lost** — it exists on no branch of this remote. Its claimed results (pytest pass, smoke pass, determinism diff 0.0) are unauditable and are **NOT carried forward**. Two of its design choices are explicitly overridden by the master directive: the omegaconf fallback (`config_io.py`) violates W2.1 (omegaconf is a hard dependency, fail loudly), and its `ffill()/bfill()` replacement reintroduced backward-fill, which W2.2 forbids. Phase 0 here starts from scratch. The remote branches `origin/codex/initialize-progress.md-and-outline-plan` and `origin/Clean-Refactor-1` are left untouched as historical record; all work happens on `main`.

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
1. **Phase 0 — Reproducibility Foundation**: seeding utility (`src/utils/seed.py`), omegaconf config (`configs/default.yaml`, hard dependency, fail loudly), run-folder artifact management under `runs/`, train-log persistence (`metrics/train_log.jsonl`), pinned requirements, determinism test with pasted same-seed diff evidence 
## 2026-06-11 — Phase 0 COMPLETE (Reproducibility Foundation)

### What was done (file-level)
- `src/utils/seed.py` (new): `set_global_seed` — python random, numpy, torch CPU+CUDA, `cudnn.deterministic=True`, `cudnn.benchmark=False`, exports PYTHONHASHSEED. Rejects bool/negative/non-int seeds.
- `src/utils/run_folder.py` (new): `create_run_folder` → `runs/<run_name>__<timestamp>__seed<seed>/` with `config.yaml` (resolved, via OmegaConf), `git_sha.txt`, `git_diff.patch`, and subfolders `metrics/ plots/ samples/ logs/ checkpoints/`. Git metadata is captured through a throwaway index built from HEAD so a corrupt/stale `.git/index` cannot poison the diff; git failures are written into the artifact files verbatim, never swallowed.
  - Naming decision: the spec's pattern was garbled in transmission ("runs/<run_name>seed/"); chose `<run_name>__<timestamp>__seed<seed>` + collision counter so same-name/same-seed runs (the D3 protocol) never overwrite.
- `configs/default.yaml` (new): mirrors every run.py CLI flag; values byte-identical to the prior argparse defaults (E3: zero default changes; prior values noted in comments).
- `run.py` (rewritten): omegaconf config loading (HARD dependency per W2.1 — no fallback; missing file/keys/unknown keys raise), CLI-overrides-config merge, `--seed` required, `--run-name`, `--config`; all outputs now inside the run folder (checkpoint → `checkpoints/model_last.pt`, forecasts → `samples/forecasts.pt`; previously they went to `data/processed/`); per-epoch JSONL `metrics/train_log.jsonl` with train_loss/val_loss/lr/wallclock/gpu_mem. `DataCollector` import made lazy (W2.5) and its re-export removed from `src/preprocessor/__init__.py`.
- `requirements.txt`: pinned to verified environment versions; added omegaconf, properscoring, arch, statsmodels, scipy, tqdm (+pytest, pyflakes as dev tools). No wandb.
- `.gitignore`: `runs/*` ignored except `runs/_selfcheck/`.

### Tests added
- `tests/test_seed.py` (6 tests), `tests/test_run_folder.py` (3 tests), `tests/test_reproducibility.py` (1 CLI-level test: two subprocess runs, seed 0, exact float equality of loss trajectory). Full suite: **10 passed**.

### Numbers (BEFORE/AFTER)
- No metric semantics changed in this phase. Smoke reference point (seed 0, 2 train steps): train_loss 0.732781708240509, val_loss 0.5519608557224274 — bit-for-bit identical across repeated runs; generated forecasts tensor also `torch.equal` across runs.
- W2.5 status: of the four known bugs, two were already fixed on main (keyword DataModule ctor, keyword sample_autoregressive); eager DataCollector import fixed this phase; wavelet read-only crash did not reproduce in the smoke path (will re-check in Phase 1 when wavelet code is rewritten).

### Surprises / negative findings
- None affecting results. Toolchain notes: the shared-folder mount intermittently truncates files written via the desktop file API and corrupts `.git/index`; mitigated by writing code from the shell side, a sandbox-local git index, and verifying file sizes. CUDA absent (CPU-only, 2 cores) — already flagged at W0.4 for Phase 4.4 feasibility.
- `tests/test_reproducibility.py` caps steps (3 train/2 val) — same pattern as the spec's own D2/D3 protocol; the determinism property is unaffected. Justification recorded per D5.

### Owner decisions pending
- None. No BLOCKED.md. Phase 0 acceptance checks all pass (see runs/_selfcheck/phase0__02_config_runpy_reprotest.md).
