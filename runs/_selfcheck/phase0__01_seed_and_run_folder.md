# Self-check — phase0 / unit 01: seed.py + run_folder.py

Date: 2026-06-11

## D1 pytest
`python3 -m pytest tests/test_seed.py tests/test_run_folder.py -v --tb=short`

```
tests/test_seed.py::test_same_seed_reproduces_all_rngs PASSED
tests/test_seed.py::test_different_seeds_differ PASSED
tests/test_seed.py::test_cudnn_flags_set PASSED
tests/test_seed.py::test_pythonhashseed_exported PASSED
tests/test_seed.py::test_rejects_bad_seeds PASSED
tests/test_seed.py::test_returns_seed PASSED
tests/test_run_folder.py::test_creates_all_required_artifacts PASSED
tests/test_run_folder.py::test_no_overwrite_on_name_collision PASSED
tests/test_run_folder.py::test_rejects_path_separators PASSED
9 passed in 6.40s
```
Note: full-suite `pytest -x` is equivalent at this point — these are the only tests in the repo.

## D2 / D3 smoke + determinism
PENDING — run.py is not yet wired to configs/seed/run folders (that is unit 02 of this phase).
Not claimed as passing. Phase 0 cannot close until D2/D3 pass in unit 02+.

## D4 pyflakes
`python3 -m pyflakes src/utils/ tests/test_seed.py tests/test_run_folder.py` → no output (clean).

## D5 honesty audit (files authored this unit)
Files: src/utils/__init__.py, src/utils/seed.py, src/utils/run_folder.py,
tests/__init__.py, tests/test_seed.py, tests/test_run_folder.py

- [x] No `pass` placeholder
- [x] No TODO / FIXME / XXX
- [x] No commented-out blocks of real logic
- [x] No bare `except:` / `except Exception: pass` — run_folder catches only
      (FileNotFoundError, TimeoutExpired) around the git subprocess and
      returns the full traceback as data, which callers persist verbatim
      into git_sha.txt / git_diff.patch; nothing is swallowed.
- [x] No `random.seed(...)` outside the seed utility itself
- [x] No hardcoded paths outside the run folder (repo root is derived from
      `__file__`; runs_root parameterized for tests)
- [x] Every function with a numerical/structural return is unit-tested
      (set_global_seed → test_seed.py; create_run_folder → test_run_folder.py;
      _run_git/_capture_git_metadata exercised via
      test_creates_all_required_artifacts which asserts on a real SHA)

## D6 result-plausibility
N/A — no evaluation tables produced in this unit.

VERDICT: unit 01 clean; Phase 0 still open pending units 02 (config + run.py) and 03 (reproducibility test).
