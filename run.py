"""Convenience script to train and run conditional TimeGrad forecasts.

The script wires together the existing data pipeline, the training
wrapper, and the autoregressive predictor so you can:

1) Load prepared data from ``data/raw`` (or optionally download fresh
   data with the collector).
2) Train the conditioning-aware TimeGrad model for a small number of
   epochs.
3) Generate autoregressive forecasts on the held-out test split while
   respecting causal masking, relative positional bias, FiLM modulation,
   and history-derived conditioning used in the model stack.

Phase 0 (MASTER_SPEC 0.3/0.4): configuration is loaded from a YAML file via
omegaconf (hard dependency — a missing omegaconf or config file is a loud
failure), CLI flags override config values, ``--seed`` is required, and all
outputs are written inside ``runs/<run_name>__<timestamp>__seed<seed>/``
(never to the repo root). One JSON line per epoch is appended to
``metrics/train_log.jsonl``.

Example (fast smoke test on CPU):
    python run.py --config configs/default.yaml --seed 0 --run-name smoke \
        --epochs 1 --max-train-steps 2 --max-val-steps 2 --num-samples 1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from omegaconf import OmegaConf

from src import config
from src.preprocessor.data_loader import TimeGradDataModule
from src.predictor.prediction_network import ConditionalTimeGradPredictionNetwork, VanillaTimeGradPredictionNetwork
from src.training.training_network import ConditionalTimeGradTrainingNetwork, VanillaTimeGradTrainingNetwork
from src.utils.run_folder import create_run_folder
from src.utils.seed import set_global_seed

# Every key a config file must define (mirrors the CLI; Phase 0.3).
CONFIG_KEYS = (
    "device",
    "epochs",
    "model",
    "batch_size",
    "context_length",
    "prediction_length",
    "lr",
    "diff_steps",
    "beta_end",
    "beta_schedule",
    "residual_layers",
    "residual_channels",
    "cond_embed_dim",
    "cond_attn_heads",
    "cond_attn_dropout",
    "num_samples",
    "max_train_steps",
    "max_val_steps",
    "max_test_steps",
    "download",
    "run_name",
    "seed",
    "cfg_dropout",
    "cfg_scale",
)


def _load_local_data() -> Dict[str, pd.DataFrame]:
    """Load pre-downloaded parquet data from ``data/raw``.

    Returns a dict matching the collector output keys.
    """

    candidates = [
        Path(config.RAW_DATA_DIR),
        Path(__file__).resolve().parent / "data" / "raw",
        Path("data") / "raw",
    ]

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

    raise FileNotFoundError(
        "Missing raw parquet files in all known locations. "
        "Either download them with --download or place them under data/raw/."
    )


def _load_data(use_local: bool) -> Dict[str, pd.DataFrame]:
    if use_local:
        return _load_local_data()

    # Imported lazily (W2.5): the collector pulls yfinance/pandas_datareader,
    # which local-data runs must not depend on at import time.
    from src.preprocessor.data_collector import DataCollector

    collector = DataCollector()
    return collector.collect_all_data()


def _prepare_datamodule(args: argparse.Namespace, device: torch.device) -> TimeGradDataModule:
    data = _load_data(use_local=not args.download)
    dm = TimeGradDataModule(
        data_dict=data,
        seq_len=args.context_length,
        forecast_horizon=args.prediction_length,
        batch_size=args.batch_size,
        device=str(device),
    )
    dm.preprocess_and_split()
    dm.build_datasets()
    return dm


def _build_networks(
    dm: TimeGradDataModule, args: argparse.Namespace, device: torch.device
):
    feature_cols = dm.get_feature_columns_by_type()
    target_dim = len(feature_cols["target"])

    if args.model == "vanilla":
        train_net = VanillaTimeGradTrainingNetwork(
            target_dim=target_dim,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            diff_steps=args.diff_steps,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            residual_layers=args.residual_layers,
            residual_channels=args.residual_channels,
        ).to(device)

        predictor = VanillaTimeGradPredictionNetwork(
            target_dim=target_dim,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            diff_steps=args.diff_steps,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            residual_layers=args.residual_layers,
            residual_channels=args.residual_channels,
        ).to(device)

        return train_net, predictor

    cond_dynamic_dim = len(feature_cols["daily"]) + len(feature_cols["monthly"])
    cond_static_dim = len(feature_cols["regime"])

    train_net = ConditionalTimeGradTrainingNetwork(
        target_dim=target_dim,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        cond_dynamic_dim=cond_dynamic_dim,
        cond_static_dim=cond_static_dim,
        diff_steps=args.diff_steps,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        residual_layers=args.residual_layers,
        residual_channels=args.residual_channels,
        cond_embed_dim=args.cond_embed_dim,
        cond_attn_heads=args.cond_attn_heads,
        cond_attn_dropout=args.cond_attn_dropout,
        cfg_dropout=getattr(args, "cfg_dropout", 0.0) or 0.0,
    ).to(device)

    predictor = ConditionalTimeGradPredictionNetwork(
        target_dim=target_dim,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        cond_dynamic_dim=cond_dynamic_dim,
        cond_static_dim=cond_static_dim,
        diff_steps=args.diff_steps,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        residual_layers=args.residual_layers,
        residual_channels=args.residual_channels,
        cond_embed_dim=args.cond_embed_dim,
        cond_attn_heads=args.cond_attn_heads,
        cond_attn_dropout=args.cond_attn_dropout,
        cfg_scale=getattr(args, "cfg_scale", 1.0) or 1.0,
    ).to(device)

    return train_net, predictor


def train_and_validate(
    model,
    dm: TimeGradDataModule,
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Path,
) -> Path:
    """Train the model, log one JSON line per epoch, save the last + best checkpoint.

    Returns the *best* checkpoint path inside the run folder.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_log_path = run_dir / "metrics" / "train_log.jsonl"
    last_path = run_dir / "checkpoints" / "model_last.pt"
    best_path = run_dir / "checkpoints" / "model_best.pt"
    best_epoch_path = run_dir / "metrics" / "best_epoch.json"

    best_val = float("inf")
    best_epoch = 0

    is_vanilla = args.model == "vanilla"

    model.train()
    with open(train_log_path, "a", encoding="utf-8") as log_file:
        for epoch in range(args.epochs):
            epoch_start = time.time()
            train_loss = 0.0
            for step, batch in enumerate(dm.train_dataloader()):
                x_hist = batch["x_hist"].to(device)
                x_future = batch["x_future"].to(device)

                optimizer.zero_grad()
                if is_vanilla:
                    loss = model(x_hist, x_future)
                else:
                    cond_dynamic = batch["cond_dynamic"].to(device)
                    cond_static = batch["cond_static"].to(device)
                    loss = model(x_hist, x_future, cond_dynamic, cond_static)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                if args.max_train_steps and (step + 1) >= args.max_train_steps:
                    break

            avg_train = train_loss / max(1, (step + 1))

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for v_step, batch in enumerate(dm.val_dataloader()):
                    x_hist = batch["x_hist"].to(device)
                    x_future = batch["x_future"].to(device)
                    if is_vanilla:
                        val_loss += model(x_hist, x_future).item()
                    else:
                        cond_dynamic = batch["cond_dynamic"].to(device)
                        cond_static = batch["cond_static"].to(device)
                        val_loss += model(x_hist, x_future, cond_dynamic, cond_static).item()
                    if args.max_val_steps and (v_step + 1) >= args.max_val_steps:
                        break

            avg_val = val_loss / max(1, (v_step + 1))
            gpu_mem = (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            )
            record = {
                "epoch": epoch + 1,
                "train_loss": avg_train,
                "val_loss": avg_val,
                "lr": optimizer.param_groups[0]["lr"],
                "wallclock": time.time() - epoch_start,
                "gpu_mem": gpu_mem,
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            print(f"Epoch {epoch + 1}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}")

            # Save best checkpoint
            if avg_val < best_val:
                best_val = avg_val
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_path)
                print(f"  → new best, saved to {best_path}")

            model.train()

    # Always save last
    torch.save(model.state_dict(), last_path)
    print(f"Saved last checkpoint to {last_path}")

    with open(best_epoch_path, "w", encoding="utf-8") as f:
        json.dump({"best_epoch": best_epoch, "best_val_loss": best_val}, f)
    print(f"Best epoch: {best_epoch} (val_loss={best_val:.4f})")

    return best_path


def run_inference(
    predictor,
    dm: TimeGradDataModule,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
    run_dir: Path,
) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    predictor.load_state_dict(state, strict=False)
    predictor.eval()

    test_batch = next(iter(dm.test_dataloader()))
    x_hist = test_batch["x_hist"].to(device)

    if args.model == "vanilla":
        samples = predictor.sample_forecast(
            x_hist=x_hist,
            num_samples=args.num_samples,
        )
    else:
        cond_dynamic = test_batch["cond_dynamic"].to(device)
        cond_static = test_batch["cond_static"].to(device)
        samples = predictor.sample_autoregressive(
            x_hist=x_hist,
            cond_dynamic=cond_dynamic,
            cond_static=cond_static,
            num_samples=args.num_samples,
        )

    output_path = run_dir / "samples" / "forecasts.pt"
    torch.save(samples.cpu(), output_path)
    print(f"Generated samples shape: {samples.shape}")
    print(f"✅ Saved forecasts to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI flags.

    Every model/data flag defaults to ``None`` so that explicitly passed
    values can be distinguished from config-file values (CLI wins).
    """
    parser = argparse.ArgumentParser(description="Train and run conditional TimeGrad")
    parser.add_argument("--config", default="configs/default.yaml", help="path to YAML config")
    parser.add_argument("--seed", type=int, required=True, help="global seed (required; no silent default)")
    parser.add_argument("--run-name", default=None, help="run folder name prefix")
    parser.add_argument("--device", default=None, help="cpu or cuda; defaults to cuda if available")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--model", type=str, default=None, choices=["conditional", "vanilla"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--prediction-length", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--diff-steps", type=int, default=None)
    parser.add_argument("--beta-end", type=float, default=None)
    parser.add_argument("--beta-schedule", type=str, default=None)
    parser.add_argument("--residual-layers", type=int, default=None)
    parser.add_argument("--residual-channels", type=int, default=None)
    parser.add_argument("--cond-embed-dim", type=int, default=None)
    parser.add_argument("--cond-attn-heads", type=int, default=None)
    parser.add_argument("--cond-attn-dropout", type=float, default=None)
    parser.add_argument("--num-samples", type=int, default=None, help="forecast samples per series")
    parser.add_argument("--max-train-steps", type=int, default=None, help="optional cap for train steps per epoch")
    parser.add_argument("--max-val-steps", type=int, default=None, help="optional cap for val steps per epoch")
    parser.add_argument("--max-test-steps", type=int, default=None, help="optional cap for test windows during eval")
    parser.add_argument(
        "--download",
        action="store_const",
        const=True,
        default=None,
        help="Download fresh data instead of loading local parquet files",
    )
    parser.add_argument("--cfg-dropout", type=float, default=None, help="CFG conditioning dropout probability (0=off)")
    parser.add_argument("--cfg-scale", type=float, default=None, help="CFG guidance scale (1.0=conditional, >1=amplified)")
    parser.add_argument(
        "--eval",
        action="store_const",
        const=True,
        default=None,
        help="Run Phase 2 evaluation after training/inference",
    )
    parser.add_argument("--eval-checkpoint", type=str, default=None, help="path to checkpoint for evaluation")
    parser.add_argument(
        "--cfg-sweep",
        action="store_const",
        const=True,
        default=None,
        help="Run CFG w-sweep over regime validation (host-only)",
    )
    return parser.parse_args()


def resolve_config(cli: argparse.Namespace) -> Dict[str, Any]:
    """Merge YAML config with CLI overrides (CLI wins). Fails loudly."""
    config_path = Path(cli.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    file_cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(file_cfg, dict):
        raise TypeError(f"Config root must be a mapping, got {type(file_cfg).__name__}")

    missing = [k for k in CONFIG_KEYS if k not in file_cfg]
    if missing:
        raise KeyError(f"Config {config_path} is missing required keys: {missing}")
    unknown = [k for k in file_cfg if k not in CONFIG_KEYS]
    if unknown:
        raise KeyError(f"Config {config_path} has unknown keys: {unknown}")

    merged: Dict[str, Any] = dict(file_cfg)
    for key in CONFIG_KEYS:
        if key in ("seed", "run_name"):
            continue
        cli_value = getattr(cli, key, None)
        if cli_value is not None:
            merged[key] = cli_value

    # seed is CLI-required; run_name: CLI > config.
    merged["seed"] = cli.seed
    if cli.run_name is not None:
        merged["run_name"] = cli.run_name
    if not merged.get("run_name"):
        raise ValueError("run_name must be set via --run-name or the config file")
    merged["download"] = bool(merged["download"])
    merged["config_file"] = str(config_path)
    return merged


def main() -> None:
    cli = parse_args()
    cfg = resolve_config(cli)
    set_global_seed(cfg["seed"])

    device_str = cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Using device: {device}")

    run_dir = create_run_folder(cfg["run_name"], cfg["seed"], cfg)
    print(f"Run folder: {run_dir}")

    # Namespace view of the resolved config for the helper functions.
    args = argparse.Namespace(**{k: cfg[k] for k in CONFIG_KEYS})
    # Forward CLI-only flags that aren't in CONFIG_KEYS
    args.eval = bool(cli.eval)
    args.eval_checkpoint = cli.eval_checkpoint
    args.cfg_sweep = bool(cli.cfg_sweep)

    dm = _prepare_datamodule(args, device)
    train_net, predictor = _build_networks(dm, args, device)

    # ── cfg-sweep mode: run regime validation across cfg_scale values ──
    if getattr(args, "cfg_sweep", None) and getattr(args, "eval_checkpoint", None):
        from src.evaluation.cfg_sweep import run_cfg_sweep

        ckpt = Path(args.eval_checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found for CFG sweep: {ckpt}")
        print(f"Loading checkpoint: {ckpt}")
        print("Running CFG sweep (regime validation across w ∈ {{0.0, 0.5, 1.0, 2.0, 4.0}}) ...")
        run_cfg_sweep(predictor, dm, ckpt, device, run_dir,
                      num_samples=args.num_samples,
                      sampling_strategy="full_horizon",
                      max_regime_windows=min(args.max_test_steps or 32, 32))
        print("CFG sweep complete. See cfg_sweep.md and metrics/cfg_sweep.json")
        return

    # ── eval-only mode: skip training / inference — load checkpoint directly ──
    if getattr(args, "eval", None) and getattr(args, "eval_checkpoint", None):
        from src.evaluation.run_eval import run_full_evaluation

        ckpt = Path(args.eval_checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found for evaluation: {ckpt}")
        print(f"Loading checkpoint: {ckpt}")
        print("Running Phase 2 evaluation (eval-only, no training) ...")
        result = run_full_evaluation(predictor, dm, run_dir, ckpt, device,
                                     num_samples=args.num_samples,
                                     sampling_strategy="full_horizon",
                                     max_test_windows=args.max_test_steps,
                                     model_type=args.model)

        # Print headline numbers
        fm_res = result.get("forecast", {})
        facts_res = result.get("stylized_facts", {})
        gen_facts = facts_res.get("generated", {})
        real_facts = facts_res.get("real_raw_un_denoised", {})
        print("── Headline metrics ──")
        print(f"  CRPS:          {fm_res.get('crps', float('nan')):.6f}")
        print(f"  coverage_0.8:  {fm_res.get('coverage_0.8', float('nan')):.6f}")
        print(f"  kurtosis real: {real_facts.get('kurtosis', float('nan')):.4f}")
        print(f"  kurtosis gen:  {gen_facts.get('kurtosis', float('nan')):.4f}")
        return

    checkpoint_path = train_and_validate(train_net, dm, args, device, run_dir)
    run_inference(predictor, dm, args, device, checkpoint_path, run_dir)

    if getattr(args, "eval", None):
        from src.evaluation.run_eval import run_full_evaluation
        if getattr(args, "eval_checkpoint", None):
            ckpt = Path(args.eval_checkpoint)
        else:
            best = run_dir / "checkpoints" / "model_best.pt"
            last = run_dir / "checkpoints" / "model_last.pt"
            ckpt = best if best.exists() else last
        print(f"Loading checkpoint: {ckpt}")
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found for evaluation: {ckpt}")
        print("Running Phase 2 evaluation ...")
        run_full_evaluation(predictor, dm, run_dir, ckpt, device,
                            num_samples=args.num_samples,
                            sampling_strategy="full_horizon",
                            max_test_windows=args.max_test_steps,
                            model_type=args.model)


if __name__ == "__main__":
    main()
