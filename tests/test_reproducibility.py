"""Phase 0.6 (MASTER_SPEC): reproducibility acceptance test.

Trains for 1 epoch with seed=0 twice via the real CLI entrypoint and asserts
the loss trajectories are bit-for-bit identical on CPU (exact float
equality after JSON round-trip — no tolerance).

Step caps (--max-train-steps 3 --max-val-steps 2) follow the same pattern as
the spec's own D2/D3 smoke protocol: the property under test is determinism
of the trajectory, which capping does not weaken. This is documented in
PROGRESS.md.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_training(tag: str) -> list[dict]:
    cmd = [
        sys.executable,
        "run.py",
        "--config", "configs/default.yaml",
        "--seed", "0",
        "--run-name", f"reprotest-{tag}",
        "--device", "cpu",
        "--epochs", "1",
        "--max-train-steps", "3",
        "--max-val-steps", "2",
        "--num-samples", "1",
    ]
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, (
        f"run.py failed (rc={proc.returncode})\n"
        f"STDOUT (tail):\n{proc.stdout[-2000:]}\n"
        f"STDERR (tail):\n{proc.stderr[-2000:]}"
    )

    match = re.search(r"^Run folder: (.+)$", proc.stdout, flags=re.MULTILINE)
    assert match, f"could not find run folder in stdout:\n{proc.stdout[-2000:]}"
    run_dir = Path(match.group(1).strip())

    log_path = run_dir / "metrics" / "train_log.jsonl"
    assert log_path.exists(), f"missing train log: {log_path}"
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(records) == 1, f"expected 1 epoch record, got {len(records)}"
    return records


def test_same_seed_identical_loss_trajectory():
    rec_a = _run_training("a")
    rec_b = _run_training("b")

    for ra, rb in zip(rec_a, rec_b):
        # Exact equality — bit-for-bit on CPU, no tolerance.
        assert ra["epoch"] == rb["epoch"]
        assert ra["train_loss"] == rb["train_loss"], (
            f"train_loss mismatch: {ra['train_loss']!r} != {rb['train_loss']!r}"
        )
        assert ra["val_loss"] == rb["val_loss"], (
            f"val_loss mismatch: {ra['val_loss']!r} != {rb['val_loss']!r}"
        )
        assert ra["lr"] == rb["lr"]
