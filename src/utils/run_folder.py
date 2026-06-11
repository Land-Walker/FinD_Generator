"""Run-folder artifact management (MASTER_SPEC Phase 0.2).

``create_run_folder`` creates ``runs/<run_name>__<timestamp>__seed<seed>/``
and inside it:

- ``config.yaml``       — the fully resolved run configuration
- ``git_sha.txt``       — current commit SHA
- ``git_diff.patch``    — uncommitted changes relative to HEAD
- subfolders            — ``metrics/``, ``plots/``, ``samples/``, ``logs/``,
                          ``checkpoints/``

Design note (documented in PROGRESS.md): the spec's folder pattern was
garbled in transmission ("runs/<run_name>seed/"); we use
``<run_name>__<timestamp>__seed<seed>`` so that repeated runs with the same
name and seed (e.g. the D3 determinism check) never overwrite each other.

Git metadata is captured with a temporary, freshly built index file so that a
corrupt or stale ``.git/index`` (a known hazard of the mounted filesystem
this repo is edited on) can never poison the diff. If git fails entirely the
error output is written into the artifact files verbatim — never silently
swallowed.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

from omegaconf import OmegaConf

SUBFOLDERS = ("metrics", "plots", "samples", "logs", "checkpoints")

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_git(args: list[str], cwd: Path, env: Optional[dict] = None) -> tuple[int, str, str]:
    """Run a git command, returning (returncode, stdout, stderr).

    A missing git binary is reported as returncode 127 with the full
    traceback in stderr rather than raising, so callers can persist the
    failure into the run artifacts.
    """
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 127, "", traceback.format_exc()


def _capture_git_metadata(run_dir: Path, repo_root: Path) -> None:
    """Write git_sha.txt and git_diff.patch into ``run_dir``."""
    rc, sha, err = _run_git(["rev-parse", "HEAD"], cwd=repo_root)
    if rc == 0:
        (run_dir / "git_sha.txt").write_text(sha.strip() + "\n", encoding="utf-8")
    else:
        (run_dir / "git_sha.txt").write_text(f"GIT ERROR (rev-parse, rc={rc}):\n{err}\n", encoding="utf-8")
        print(f"[run_folder] WARNING: could not capture git SHA (rc={rc}): {err.strip()}")

    # Build a throwaway index from HEAD so the diff reflects worktree-vs-HEAD
    # regardless of the state of the repository's real index.
    tmp_index = tempfile.NamedTemporaryFile(prefix="git_index_", delete=False)
    tmp_index.close()
    env = dict(os.environ)
    env["GIT_INDEX_FILE"] = tmp_index.name
    try:
        rc_rt, _, err_rt = _run_git(["read-tree", "HEAD"], cwd=repo_root, env=env)
        if rc_rt != 0:
            (run_dir / "git_diff.patch").write_text(
                f"GIT ERROR (read-tree, rc={rc_rt}):\n{err_rt}\n", encoding="utf-8"
            )
            print(f"[run_folder] WARNING: could not build temp git index (rc={rc_rt}): {err_rt.strip()}")
            return
        rc_diff, diff, err_diff = _run_git(["diff", "HEAD"], cwd=repo_root, env=env)
        if rc_diff == 0:
            (run_dir / "git_diff.patch").write_text(diff, encoding="utf-8")
        else:
            (run_dir / "git_diff.patch").write_text(
                f"GIT ERROR (diff, rc={rc_diff}):\n{err_diff}\n", encoding="utf-8"
            )
            print(f"[run_folder] WARNING: could not capture git diff (rc={rc_diff}): {err_diff.strip()}")
    finally:
        os.unlink(tmp_index.name)


def create_run_folder(
    run_name: str,
    seed: int,
    config: Mapping[str, Any],
    runs_root: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> Path:
    """Create a uniquely named run folder with all required artifacts.

    Args:
        run_name: Human-readable run name (used as the folder prefix).
        seed: The global seed for this run (embedded in the folder name).
        config: Fully resolved configuration; saved as ``config.yaml``.
        runs_root: Parent directory for run folders (default ``<repo>/runs``).
        repo_root: Repository root for git metadata (default: autodetected).

    Returns:
        Path of the created run folder.
    """
    if not run_name or any(ch in run_name for ch in "/\\"):
        raise ValueError(f"run_name must be a non-empty string without path separators, got {run_name!r}")

    repo_root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    runs_root = Path(runs_root) if runs_root is not None else repo_root / "runs"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{run_name}__{timestamp}__seed{seed}"
    run_dir = runs_root / base
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = runs_root / f"{base}__{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)

    for sub in SUBFOLDERS:
        (run_dir / sub).mkdir(exist_ok=True)

    conf = OmegaConf.create(dict(config))
    OmegaConf.save(conf, run_dir / "config.yaml")

    _capture_git_metadata(run_dir, repo_root)

    return run_dir
