"""Unit tests for src/utils/run_folder.py (Phase 0.2)."""
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.utils.run_folder import SUBFOLDERS, create_run_folder

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_creates_all_required_artifacts(tmp_path):
    cfg = {"epochs": 1, "lr": 0.001, "beta_schedule": "linear"}
    run_dir = create_run_folder("unittest", seed=0, config=cfg, runs_root=tmp_path)

    assert run_dir.exists() and run_dir.parent == tmp_path
    assert run_dir.name.startswith("unittest__")
    assert run_dir.name.endswith("__seed0")

    for sub in SUBFOLDERS:
        assert (run_dir / sub).is_dir(), f"missing subfolder {sub}"

    loaded = OmegaConf.load(run_dir / "config.yaml")
    assert loaded.epochs == 1
    assert loaded.lr == 0.001
    assert loaded.beta_schedule == "linear"

    sha = (run_dir / "git_sha.txt").read_text().strip()
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha), (
        f"git_sha.txt does not contain a commit SHA: {sha!r}"
    )
    assert (run_dir / "git_diff.patch").exists()


def test_no_overwrite_on_name_collision(tmp_path):
    cfg = {"a": 1}
    d1 = create_run_folder("collide", seed=3, config=cfg, runs_root=tmp_path)
    d2 = create_run_folder("collide", seed=3, config=cfg, runs_root=tmp_path)
    assert d1 != d2
    assert d1.exists() and d2.exists()


def test_rejects_path_separators():
    with pytest.raises(ValueError):
        create_run_folder("evil/name", seed=0, config={})
    with pytest.raises(ValueError):
        create_run_folder("", seed=0, config={})
