"""Phase 1 coverage probe (BEFORE/AFTER the causal data fix).

Measures empirical central 80% coverage of the existing
``checkpoints/conditional_timegrad_best.pt`` on held-out test windows.

The checkpoint predates Phase 0/1 and is only dimensionally compatible with
the legacy processed frames in ``data/processed/*_processed.csv``
(context 64, horizon 24, target_dim 1, cond_dynamic 22, cond_static 6, no
quarterly features, single macro_regime column). The probe therefore reads
those frames directly. Protocol (fixed for before/after comparability):

- 64 evenly spaced test windows, 16 samples each, seed 0
- coverage = mean over (window, horizon step) of the indicator
  q10 <= y_true <= q90, quantiles taken across the 16 samples
- resumable: ``--windows i0 i1`` samples a chunk; ``--aggregate`` reports

This is a PROBE with a fixed small budget, not the Phase 2 evaluation; its
only purpose is an apples-to-apples 80%-coverage number before vs after the
Phase 1 data fix, as required by MASTER_SPEC Phase 1 acceptance and W3.2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.predictor.prediction_network import ConditionalTimeGradPredictionNetwork  # noqa: E402
from src.utils.seed import set_global_seed  # noqa: E402

CONTEXT = 64
HORIZON = 24
N_WINDOWS = 64
N_SAMPLES = 16
SEED = 0

TARGET_COLS = ["target_pca_1"]
DAILY_COLS = [
    "market_pca_1", "market_pca_2", "market_pca_3", "volume_scaled",
    "daily_vix_daily_scaled", "daily_yield_curve_daily_scaled",
    "day_of_week", "month", "quarter", "year", "is_month_end",
    "is_quarter_end", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "quarter_sin", "quarter_cos",
]
MONTHLY_COLS = ["monthly_pca_1", "monthly_pca_2", "monthly_pca_3", "monthly_pca_4"]
REGIME_COLS = [
    "market_regime_bear", "market_regime_bull", "market_regime_sideways",
    "vol_regime_high_vol", "vol_regime_normal_vol", "macro_regime_normal",
]


def load_frame() -> pd.DataFrame:
    df = pd.read_csv(REPO_ROOT / "data/processed/test_processed.csv")
    needed = TARGET_COLS + DAILY_COLS + MONTHLY_COLS + REGIME_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"legacy test frame is missing columns: {missing}")
    return df


def window_starts(n_rows: int) -> list[int]:
    max_start = n_rows - CONTEXT - HORIZON
    if max_start < 1:
        raise ValueError(f"not enough rows ({n_rows}) for context+horizon")
    return [int(round(s)) for s in np.linspace(0, max_start, N_WINDOWS)]


def build_predictor() -> ConditionalTimeGradPredictionNetwork:
    predictor = ConditionalTimeGradPredictionNetwork(
        target_dim=1,
        context_length=CONTEXT,
        prediction_length=HORIZON,
        cond_dynamic_dim=len(DAILY_COLS) + len(MONTHLY_COLS),
        cond_static_dim=len(REGIME_COLS),
        diff_steps=100,
        beta_end=0.1,
        beta_schedule="linear",
        residual_layers=6,
        residual_channels=32,
        cond_embed_dim=64,
        cond_attn_heads=4,
        cond_attn_dropout=0.1,
    )
    state = torch.load(REPO_ROOT / "checkpoints/conditional_timegrad_best.pt", map_location="cpu")
    missing, unexpected = predictor.load_state_dict(state, strict=False)
    # Loud report: anything not loaded must be visible, never silent.
    print(f"[probe] load_state_dict: missing={list(missing)} unexpected={list(unexpected)}")
    if any(k.startswith("model.") or k.startswith("history_encoder") for k in missing):
        raise RuntimeError(f"checkpoint failed to populate model weights: missing={missing}")
    predictor.eval()
    return predictor


def sample_chunk(i0: int, i1: int, out_dir: Path) -> None:
    set_global_seed(SEED + i0)  # per-chunk reseed keeps chunks independent and reproducible
    df = load_frame()
    starts = window_starts(len(df))[i0:i1]
    predictor = build_predictor()

    x = df[TARGET_COLS].to_numpy(dtype=np.float32)
    dyn = df[DAILY_COLS + MONTHLY_COLS].to_numpy(dtype=np.float32)
    reg = df[REGIME_COLS].to_numpy(dtype=np.float32)

    x_hist = torch.tensor(np.stack([x[s : s + CONTEXT] for s in starts]))
    y_true = torch.tensor(np.stack([x[s + CONTEXT : s + CONTEXT + HORIZON] for s in starts]))
    cond_dyn = torch.tensor(np.stack([dyn[s : s + CONTEXT] for s in starts]))
    cond_stat = torch.tensor(np.stack([reg[s + CONTEXT - 1] for s in starts]))

    with torch.no_grad():
        samples = predictor.sample_autoregressive(
            x_hist=x_hist,
            cond_dynamic=cond_dyn,
            cond_static=cond_stat,
            num_samples=N_SAMPLES,
        )  # [S, B, H, 1]

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"starts": starts, "samples": samples, "y_true": y_true},
        out_dir / f"chunk_{i0:03d}_{i1:03d}.pt",
    )
    print(f"[probe] saved chunk {i0}:{i1} samples shape {tuple(samples.shape)}")


def aggregate(out_dir: Path, label: str) -> None:
    chunks = sorted(out_dir.glob("chunk_*.pt"))
    if not chunks:
        raise FileNotFoundError(f"no chunks in {out_dir}")
    samples_list, y_list, starts_all = [], [], []
    for c in chunks:
        d = torch.load(c)
        samples_list.append(d["samples"])
        y_list.append(d["y_true"])
        starts_all.extend(d["starts"])
    samples = torch.cat(samples_list, dim=1)  # [S, B, H, 1]
    y_true = torch.cat(y_list, dim=0)  # [B, H, 1]
    assert len(set(starts_all)) == len(starts_all), "duplicate windows aggregated"

    q10 = torch.quantile(samples, 0.10, dim=0)
    q90 = torch.quantile(samples, 0.90, dim=0)
    inside = ((y_true >= q10) & (y_true <= q90)).float()
    coverage80 = float(inside.mean())
    cov_by_step = inside.mean(dim=(0, 2)).tolist()

    report = {
        "label": label,
        "protocol": {
            "n_windows": len(starts_all),
            "n_samples": int(samples.shape[0]),
            "context": CONTEXT,
            "horizon": HORIZON,
            "seed_base": SEED,
            "interval": "central 80% (q10..q90, empirical)",
            "checkpoint": "checkpoints/conditional_timegrad_best.pt",
            "data": "data/processed/test_processed.csv (legacy frame)",
        },
        "coverage_80": coverage80,
        "coverage_80_by_horizon_step": cov_by_step,
    }
    out_path = out_dir / "coverage_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("label", "coverage_80")}, indent=2))
    print(f"[probe] full report: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, help="probe output dir under runs/")
    ap.add_argument("--windows", nargs=2, type=int, metavar=("I0", "I1"), help="chunk of window indices to sample")
    ap.add_argument("--aggregate", action="store_true", help="aggregate chunks into the coverage report")
    ap.add_argument("--label", default="probe", help="report label (e.g. before_fix/after_fix)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.windows:
        sample_chunk(args.windows[0], args.windows[1], out_dir)
    elif args.aggregate:
        aggregate(out_dir, args.label)
    else:
        raise SystemExit("specify --windows I0 I1 or --aggregate")


if __name__ == "__main__":
    main()
