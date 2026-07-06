"""
run_baselines.py — Phase 3 baseline battery runner.

Given a fitted TimeGradDataModule and run-id, this module:
1. Runs each CPU baseline (historical bootstrap, block bootstrap, GARCH-t)
   on the test split, producing samples in canonical space (denoised-close
   log returns).
2. Computes forecast_metrics and stylized_facts on each baseline via the
   SAME code path the model uses.
3. Saves per-baseline JSON to runs/<run_id>/metrics/baseline_<name>.json.

Usage:
    python -m src.baselines.run_baselines --run-id <id> \
      --data-config configs/default.yaml --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.evaluation import forecast_metrics as fm
from src.evaluation import stylized_facts as sf
from src.evaluation.inverse_transform import target_pca_to_log_returns
from src.preprocessor.data_loader import TimeGradDataModule
from src.utils.seed import set_global_seed


def _load_data_dict(config_path: str) -> Dict[str, pd.DataFrame]:
    """Load parquet data from data/raw, matching run.py's logic."""
    candidates = [
        Path("data") / "raw",
    ]
    repo_root = Path(__file__).resolve().parents[2]
    candidates.insert(0, repo_root / "data" / "raw")

    for base in candidates:
        datasets = {
            "target": base / "target.parquet",
            "market": base / "market.parquet",
            "daily_macro": base / "daily_macro.parquet",
            "monthly_macro": base / "monthly_macro.parquet",
            "quarterly_macro": base / "quarterly_macro.parquet",
        }
        if all(path.exists() for path in datasets.values()):
            return {name: pd.read_parquet(path) for name, path in datasets.items()}

    raise FileNotFoundError(f"No raw data found in any of: {[str(b) for b in candidates]}")


def _get_train_test_log_returns(dm: TimeGradDataModule):
    """Extract train and test log returns from denoised close."""
    train_close = dm.train_df["close_den"].values
    train_logret = np.log(train_close[1:] / train_close[:-1])

    test_close = dm.test_df["close_den"].values
    test_logret = np.log(test_close[1:] / test_close[:-1])

    return train_logret, test_logret


def _get_target_returns(dm: TimeGradDataModule) -> np.ndarray:
    """Iterate test dataloader and convert x_future → canonical log returns."""
    scaler = dm.scalers.get("target_scaler")
    pca = dm.pcas.get("target_pca")
    if scaler is None or pca is None:
        raise RuntimeError("Scaler/PCA not fitted. Run preprocess_and_split + build_datasets first.")

    targets_all = []
    for batch in dm.test_dataloader():
        x_future = batch["x_future"].cpu().numpy()
        rets = target_pca_to_log_returns(x_future, pca, scaler)
        targets_all.append(rets)
    return np.concatenate(targets_all, axis=0)


def _get_garch_history(dm: TimeGradDataModule) -> np.ndarray:
    """For each test window, extract context_length of log returns preceding the horizon.

    Returns shape (n_paths, context_length-1).
    """
    test_close = dm.test_df["close_den"].values
    horizon = dm.forecast_horizon
    context_len = dm.seq_len
    n_paths = len(dm.test_set)

    history = np.empty((n_paths, context_len - 1), dtype=np.float64)
    for i in range(n_paths):
        win_close = test_close[i : i + context_len]
        history[i] = np.log(win_close[1:] / win_close[:-1])
    return history


def _flatten_samples(samples: np.ndarray) -> np.ndarray:
    """Flatten baseline samples: (n_samples, n_paths, horizon) → (n_samples, n_total)."""
    return samples.reshape(samples.shape[0], -1)


def _compute_forecast_metrics_baseline(
    samples: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, Any]:
    """Compute Phase 2 forecast metrics for baseline samples.

    samples : (n_samples, n_paths, horizon_or_horizon-1) in log-return space
    targets : (n_paths, horizon_or_horizon-1) in log-return space
    """
    pred_flat = _flatten_samples(samples)
    true_flat = targets.ravel()

    result = {
        "crps": fm.crps_ensemble(pred_flat, true_flat),
        "mae": fm.mae(pred_flat, true_flat),
        "rmse": fm.rmse(pred_flat, true_flat),
    }
    for alpha in [0.05, 0.10, 0.50, 0.90, 0.95]:
        result[f"quantile_loss_{alpha}"] = fm.quantile_loss(pred_flat, true_flat, alpha)
    for level in [0.5, 0.8, 0.9, 0.95]:
        result[f"coverage_{level}"] = fm.coverage(pred_flat, true_flat, level)
    pit = fm.pit_values(pred_flat, true_flat)
    ks_stat, ks_pval = fm.pit_ks_test(pit)
    result["pit_ks_stat"] = ks_stat
    result["pit_ks_pvalue"] = ks_pval
    result["energy_score"] = fm.energy_score(pred_flat, true_flat)
    result["nll"] = fm.negative_log_likelihood(pred_flat, true_flat)
    return result


def _compute_stylized_facts_baseline(
    samples: np.ndarray,
) -> Dict[str, float]:
    """Compute stylized facts for baseline samples.

    samples : (n_samples, n_paths, horizon_or_horizon-1) in log-return space
    """
    gen_ret_1d = samples.ravel()
    gen_facts = sf.all_stylized_facts(gen_ret_1d)

    gen_ret_paths = samples.reshape(-1, samples.shape[-1])
    gen_dd = sf.drawdown_distribution(gen_ret_paths)
    for k, v in gen_dd.items():
        gen_facts[f"drawdown_{k}"] = v
    return gen_facts


def run_all_baselines(
    dm: TimeGradDataModule,
    run_dir: Path,
    seed: int,
    num_samples: int = 200,
    block_length: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """Run all 3 CPU baselines and save results.

    Returns dict of {baseline_name: {forecast: ..., stylized_facts: ...}}.
    """
    train_logret, _ = _get_train_test_log_returns(dm)
    targets = _get_target_returns(dm)
    n_paths = len(dm.test_set)
    horizon = dm.forecast_horizon

    print(f"Baselines: train returns={train_logret.shape}, targets={targets.shape}, "
          f"n_paths={n_paths}, horizon={horizon}")

    rng = np.random.default_rng(seed)
    results: Dict[str, Dict[str, Any]] = {}

    # ── Historical Bootstrap ──
    from src.baselines.historical_bootstrap import generate_samples as hist_gen

    print("Running historical bootstrap ...")
    hist_samples = hist_gen(train_logret, num_samples, n_paths, horizon, rng)
    hist_samples_ret = hist_samples[..., :-1] if hist_samples.shape[-1] == horizon else hist_samples

    hist_fm = _compute_forecast_metrics_baseline(hist_samples_ret, targets)
    hist_sf = _compute_stylized_facts_baseline(hist_samples_ret)
    results["hist_boot"] = {"forecast": hist_fm, "stylized_facts": hist_sf}
    print(f"  hist_boot CRPS={hist_fm['crps']:.6f}, kurtosis={hist_sf['kurtosis']:.4f}")

    # ── Block Bootstrap ──
    from src.baselines.block_bootstrap import generate_samples as block_gen

    print("Running block bootstrap ...")
    block_samples = block_gen(train_logret, num_samples, n_paths, horizon, block_length, rng)
    block_samples_ret = block_samples[..., :-1] if block_samples.shape[-1] == horizon else block_samples

    block_fm = _compute_forecast_metrics_baseline(block_samples_ret, targets)
    block_sf = _compute_stylized_facts_baseline(block_samples_ret)
    results["block_boot"] = {"forecast": block_fm, "stylized_facts": block_sf}
    print(f"  block_boot CRPS={block_fm['crps']:.6f}, kurtosis={block_sf['kurtosis']:.4f}")

    # ── GARCH-t ──
    from src.baselines.garch_baseline import GARCHBaseline

    print("Fitting GARCH(1,1)-t ...")
    garch = GARCHBaseline(dist="t", seed=seed).fit(train_logret)
    history = _get_garch_history(dm)
    print(f"  GARCH history shape: {history.shape}")

    garch_samples = garch.generate_samples(num_samples, n_paths, horizon, history)
    garch_samples_ret = garch_samples[..., :-1] if garch_samples.shape[-1] == horizon else garch_samples

    garch_fm = _compute_forecast_metrics_baseline(garch_samples_ret, targets)
    garch_sf = _compute_stylized_facts_baseline(garch_samples_ret)
    results["garch_t"] = {"forecast": garch_fm, "stylized_facts": garch_sf}
    print(f"  garch_t CRPS={garch_fm['crps']:.6f}, kurtosis={garch_sf['kurtosis']:.4f}")

    # ── Save ──
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for name, res in results.items():
        path = metrics_dir / f"baseline_{name}.json"
        serializable = {}
        for key in ["forecast", "stylized_facts"]:
            serializable[key] = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                                 for k, v in res[key].items()}
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"  Saved {path}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 3 baseline battery")
    parser.add_argument("--run-id", required=True, help="Run directory under runs/")
    parser.add_argument("--data-config", default="configs/default.yaml", help="Path to YAML config for data pipeline")
    parser.add_argument("--seed", type=int, required=True, help="Global seed for reproducibility")
    parser.add_argument("--num-samples", type=int, default=200, help="Ensemble members per baseline")
    parser.add_argument("--block-length", type=int, default=10, help="Expected block length for block bootstrap")
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)

    run_dir = Path("runs") / args.run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    data_dict = _load_data_dict(args.data_config)
    config = OmegaConf.load(args.data_config)
    dm = TimeGradDataModule(
        data_dict=data_dict,
        seq_len=config.context_length,
        forecast_horizon=config.prediction_length,
        batch_size=config.batch_size,
        device="cpu",
    )
    dm.preprocess_and_split()
    dm.build_datasets()

    results = run_all_baselines(
        dm, run_dir, args.seed, num_samples=args.num_samples,
        block_length=args.block_length,
    )
    print(f"Baseline battery complete. {len(results)} baselines saved to {run_dir}/metrics/")

    from src.evaluation.run_eval import write_comparison_table
    write_comparison_table(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
