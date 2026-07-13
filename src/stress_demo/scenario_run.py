"""scenario_run.py — Regime-conditional scenario path generator.

Given a regime spec (e.g. macro_regime=high_inflation) and a cfg-scale,
loads a trained checkpoint and generates N scenario paths.

GPU-optional: works on CPU with small N for testing.
Full-GPU run command goes to HOST_TASKS.md.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from src.predictor.prediction_network import ConditionalTimeGradPredictionNetwork
from src.preprocessor.data_loader import TimeGradDataModule
from src.evaluation.inverse_transform import target_pca_to_log_returns
from src.utils.seed import set_global_seed


REGIME_DIM_MAP = {
    "vol_regime": "vol_regime_",
    "market_regime": "market_regime_",
    "macro_regime": "macro_regime_",
}


def _load_data_and_pipeline(data_path: Path = None) -> Tuple[TimeGradDataModule, ConditionalTimeGradPredictionNetwork]:
    if data_path is None:
        data_path = Path("data/raw")
    data_dict = {}
    for name in ["target", "market", "daily_macro", "monthly_macro", "quarterly_macro"]:
        data_dict[name] = pd.read_parquet(data_path / f"{name}.parquet")

    dm = TimeGradDataModule(data_dict, seq_len=64, forecast_horizon=5, batch_size=16, device="cpu")
    dm.preprocess_and_split()
    dm.build_datasets()

    fc = dm.get_feature_columns_by_type()
    td = len(fc["target"])
    cdd = len(fc["daily"]) + len(fc["monthly"])
    csd = len(fc["regime"])

    predictor = ConditionalTimeGradPredictionNetwork(
        target_dim=td, context_length=64, prediction_length=5,
        cond_dynamic_dim=cdd, cond_static_dim=csd,
        diff_steps=100, beta_end=0.1, beta_schedule="linear",
        residual_layers=6, residual_channels=32,
        cond_embed_dim=64, cond_attn_heads=4, cond_attn_dropout=0.1,
        cfg_scale=2.0,
    )
    return dm, predictor


def build_regime_onehot(
    regime_cols: List[str],
    overrides: Dict[str, str],
    n_batch: int,
) -> np.ndarray:
    cond_static = np.zeros((n_batch, len(regime_cols)), dtype=np.float32)
    for dim_key, label in overrides.items():
        prefix = REGIME_DIM_MAP.get(dim_key)
        if prefix is None:
            continue
        col_name = f"{prefix}{label}"
        if col_name in regime_cols:
            idx = regime_cols.index(col_name)
            cond_static[:, idx] = 1.0
    return cond_static


def generate_scenario_paths(
    checkpoint_path: Path,
    regime_spec: Dict[str, str],
    cfg_scale: float,
    num_scenarios: int,
    num_windows: int = 0,
    device_str: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dm, predictor = _load_data_and_pipeline()
    device = torch.device(device_str)

    predictor.set_cfg_scale(cfg_scale)
    state = torch.load(checkpoint_path, map_location=device)
    predictor.load_state_dict(state, strict=False)
    predictor.to(device)
    predictor.eval()

    fc = dm.get_feature_columns_by_type()
    regime_cols = fc["regime"]
    pca = dm.pcas.get("target_pca")
    scaler = dm.scalers.get("target_scaler")

    loader = dm.test_dataloader()
    all_samples = []
    all_targets = []

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if num_windows and step >= num_windows:
                break

            x_hist = batch["x_hist"].to(device)
            x_future = batch["x_future"].to(device)
            cond_dynamic = batch["cond_dynamic"].to(device)
            n_b = x_hist.size(0)

            override_static = build_regime_onehot(regime_cols, regime_spec, n_b)
            cond_static = torch.tensor(override_static, device=device, dtype=torch.float32)

            samples = predictor.sample_autoregressive(
                x_hist=x_hist, cond_dynamic=cond_dynamic, cond_static=cond_static,
                num_samples=num_scenarios, sampling_strategy="full_horizon",
            )
            all_samples.append(samples.cpu().numpy())
            all_targets.append(x_future.cpu().numpy())

    samples_np = np.concatenate([s.transpose(1, 0, 2, 3) for s in all_samples], axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    # samples_np: (n_windows, n_scenarios, horizon, target_dim)

    scenario_returns = target_pca_to_log_returns(samples_np.transpose(1, 0, 2, 3), pca, scaler)
    # scenario_returns: (n_scenarios, n_windows, horizon-1)

    target_returns = target_pca_to_log_returns(targets_np, pca, scaler)
    # target_returns: (n_windows, horizon-1)

    return scenario_returns, target_returns, samples_np, targets_np


def _get_raw_test_returns(dm: TimeGradDataModule) -> np.ndarray:
    target_raw = dm._ensure_index(dm.data["target"])
    test_df = dm.test_df
    close_raw = target_raw["close"].reindex(test_df.index).ffill()
    if close_raw.isna().any():
        close_raw = close_raw.ffill().bfill()
    log_rets = np.log(close_raw.values[1:] / close_raw.values[:-1])
    return log_rets


def run_scenario(
    checkpoint_path: Path,
    regime_spec: Dict[str, str],
    cfg_scale: float,
    run_dir: Path,
    num_scenarios: int = 100,
    num_windows: int = 0,
    device_str: str = "cpu",
) -> Dict[str, Any]:
    t0 = time.time()

    print(f"Generating scenarios: regime={regime_spec}, cfg_scale={cfg_scale}")
    scenario_rets, target_rets, samples_np, targets_np = generate_scenario_paths(
        checkpoint_path=checkpoint_path,
        regime_spec=regime_spec,
        cfg_scale=cfg_scale,
        num_scenarios=num_scenarios,
        num_windows=num_windows,
        device_str=device_str,
    )

    # Also generate unconditional (w=0) for comparison
    print("Generating unconditional (w=0) for comparison ...")
    uncond_rets, _, _, _ = generate_scenario_paths(
        checkpoint_path=checkpoint_path,
        regime_spec=regime_spec,
        cfg_scale=0.0,
        num_scenarios=num_scenarios,
        num_windows=num_windows,
        device_str=device_str,
    )

    # Historical reference
    dm, _ = _load_data_and_pipeline()
    hist_rets = _get_raw_test_returns(dm)

    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(samples_dir / "scenario_returns.npz",
                        scenario=scenario_rets, unconditional=uncond_rets,
                        targets=target_rets, historical=hist_rets)
    print(f"Saved scenario returns to {samples_dir}")

    elapsed = time.time() - t0
    print(f"Scenario generation complete in {elapsed:.1f}s")

    return {
        "scenario_returns": scenario_rets,
        "unconditional_returns": uncond_rets,
        "target_returns": target_rets,
        "historical_returns": hist_rets,
        "regime_spec": regime_spec,
        "cfg_scale": cfg_scale,
        "num_scenarios": num_scenarios,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Regime-conditional scenario generation")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    p.add_argument("--regime", type=str, required=True, help="Regime spec as JSON, e.g. '{\"macro_regime\": \"high_inflation\"}'")
    p.add_argument("--cfg-scale", type=float, default=2.0, help="CFG guidance scale")
    p.add_argument("--run-dir", type=Path, required=True, help="Output run directory")
    p.add_argument("--num-scenarios", type=int, default=100, help="Number of scenarios per window")
    p.add_argument("--num-windows", type=int, default=0, help="Max test windows (0=all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_global_seed(args.seed)
    regime_spec = json.loads(args.regime)
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    run_scenario(
        checkpoint_path=args.checkpoint,
        regime_spec=regime_spec,
        cfg_scale=args.cfg_scale,
        run_dir=run_dir,
        num_scenarios=args.num_scenarios,
        num_windows=args.num_windows,
        device_str=args.device,
    )
