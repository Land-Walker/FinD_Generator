"""cfg_sweep.py — Classifier-Free Guidance sweep over regime validation.

Runs regime_validation across cfg_scale values
w ∈ {0.0, 0.5, 1.0, 2.0, 4.0} and records how each regime's effect size
(Cohen's d) and coverage move with w.

Real run is host-only (GPU needed for full test windows). This script
provides the plumbing; invoke via run.py --cfg-sweep.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from . import regime_validation as rv
from .inverse_transform import target_pca_to_log_returns


CFG_SCALES = [0.0, 0.5, 1.0, 2.0, 4.0]


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def generate_regime_conditional_samples_cfg(
    predictor,
    dm,
    device: torch.device,
    num_samples: int,
    regime_cols: List[str],
    all_x_hist: np.ndarray,
    all_cond_dynamic: np.ndarray,
    cfg_scale: float,
    sampling_strategy: str = "full_horizon",
) -> Dict[str, Dict[str, np.ndarray]]:
    """Generate regime-conditional samples at a specific cfg_scale."""
    predictor.set_cfg_scale(cfg_scale)

    regime_dims = {
        'vol_regime': [c for c in regime_cols if c.startswith('vol_regime_')],
        'market_regime': [c for c in regime_cols if c.startswith('market_regime_')],
        'macro_regime': [c for c in regime_cols if c.startswith('macro_regime_')],
    }

    n_batch = all_x_hist.shape[0]
    samples_by_regime: Dict[str, Dict[str, np.ndarray]] = {}

    with torch.no_grad():
        for dim_key, cols in regime_dims.items():
            dim_samples: Dict[str, np.ndarray] = {}
            for col in cols:
                label = col.split('_', 1)[1]
                cond_static = np.zeros((n_batch, len(regime_cols)), dtype=np.float32)
                col_idx = regime_cols.index(col)
                cond_static[:, col_idx] = 1.0

                label_samples = []
                for b in range(0, n_batch, 16):
                    end = min(b + 16, n_batch)
                    xh = torch.tensor(all_x_hist[b:end], device=device)
                    cd = torch.tensor(all_cond_dynamic[b:end], device=device)
                    cs = torch.tensor(cond_static[b:end], device=device)

                    batch_out = predictor.sample_autoregressive(
                        x_hist=xh,
                        cond_dynamic=cd,
                        cond_static=cs,
                        num_samples=num_samples,
                        sampling_strategy=sampling_strategy,
                    )
                    label_samples.append(_to_numpy(batch_out))

                all_label = np.concatenate(
                    [s.transpose(1, 0, 2, 3) for s in label_samples], axis=0
                )
                dim_samples[label] = all_label
            samples_by_regime[dim_key] = dim_samples

    return samples_by_regime


def run_cfg_sweep(
    predictor,
    dm,
    checkpoint_path: Path,
    device: torch.device,
    run_dir: Path,
    num_samples: int = 10,
    sampling_strategy: str = "full_horizon",
    max_regime_windows: int = 32,
    cfg_scales: List[float] = None,
) -> Dict[str, Any]:
    """Run regime_validation across cfg_scale values and write cfg_sweep.md.

    Returns sweep results dict.
    """
    if cfg_scales is None:
        cfg_scales = CFG_SCALES

    state = torch.load(checkpoint_path, map_location=device)
    predictor.load_state_dict(state, strict=False)
    predictor.eval()

    feature_cols = dm.get_feature_columns_by_type()
    regime_cols = feature_cols["regime"]
    pca = dm.pcas.get("target_pca")
    scaler = dm.scalers.get("target_scaler")

    dynamic_list = []
    xhist_list = []
    loader = dm.test_dataloader()
    for batch in loader:
        dynamic_list.append(_to_numpy(batch["cond_dynamic"]))
        xhist_list.append(_to_numpy(batch["x_hist"]))
    all_cond_dynamic = np.concatenate(dynamic_list, axis=0)
    all_x_hist_arr = np.concatenate(xhist_list, axis=0)

    max_regime_windows = min(len(all_x_hist_arr), max_regime_windows)

    sweep_results: Dict[str, Any] = {}
    for w in cfg_scales:
        print(f"  CFG scale w={w:.1f} ...")
        t0 = time.time()
        predictor.set_cfg_scale(w)

        samples_by_regime = generate_regime_conditional_samples_cfg(
            predictor, dm, device, num_samples,
            regime_cols,
            all_x_hist_arr[:max_regime_windows],
            all_cond_dynamic[:max_regime_windows],
            w,
            sampling_strategy,
        )

        regime_results = {}
        for dim, labels in samples_by_regime.items():
            dim_canon = {}
            for label, s in labels.items():
                s_canon = target_pca_to_log_returns(s, pca, scaler)
                s_2d = s_canon.reshape(-1, s_canon.shape[-1]) if s_canon.ndim > 2 else s_canon
                dim_canon[label] = s_2d
            regime_results[dim] = dim_canon

        regime_val = rv.regime_validation_report(regime_results, list(regime_results.keys()))
        elapsed = time.time() - t0

        sweep_results[f"w={w:.1f}"] = {
            "regime_validation": regime_val,
            "elapsed_s": elapsed,
        }

        for dim, dim_res in regime_val.items():
            for label, res in dim_res.items():
                if 'warning' in res:
                    print(f"    {dim}/{label}: {res['warning']}")
                else:
                    print(f"    {dim}/{label}: KS p={res['ks_pvalue']:.4f}, d={res['cohens_d']:.3f}")

    # Restore default cfg_scale
    predictor.set_cfg_scale(1.0)

    # Write cfg_sweep.md
    _write_cfg_sweep_md(run_dir, sweep_results)

    # Also save raw JSON
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "cfg_sweep.json", "w") as f:
        json.dump(sweep_results, f, indent=2)

    return sweep_results


def _write_cfg_sweep_md(run_dir: Path, sweep_results: Dict[str, Any]) -> None:
    """Write docs/cfg_sweep.md with per-w regime validation results."""
    path = run_dir / "cfg_sweep.md"

    lines = [
        "# CFG Sweep — Classifier-Free Guidance",
        "",
        "Sweeps cfg_scale w ∈ {0.0, 0.5, 1.0, 2.0, 4.0} and records",
        "regime-conditioned effect sizes (Cohen's d) and coverage.",
        "",
        "## Summary Table",
        "",
        "| CFG Scale w | Regime Dimension | Label | Cohen's d | KS p-value |",
        "|-------------|-----------------|-------|-----------|------------|",
    ]

    for w_key, w_data in sweep_results.items():
        regime_val = w_data.get("regime_validation", {})
        for dim, dim_res in regime_val.items():
            for label, res in dim_res.items():
                if 'warning' in res:
                    lines.append(
                        f"| {w_key} | {dim} | {label} | — | — ⚠ {res['warning']} |"
                    )
                else:
                    d = res.get('cohens_d', float('nan'))
                    ks = res.get('ks_pvalue', float('nan'))
                    lines.append(
                        f"| {w_key} | {dim} | {label} | {d:.3f} | {ks:.4f} |"
                    )

    lines += [
        "",
        "## Interpretation",
        "",
        "- w=1.0 = pure conditional (default, current behavior)",
        "- w=0.0 = unconditional (regime conditioning zeroed)",
        "- w>1.0 = amplified conditioning (stress-test mode)",
        "- The default w should maximize regime-control effect size",
        "  without pushing CRPS up by more than 10%.",
        "",
        "**Note:** This report was generated on CPU with limited test windows.",
        "Full sweep requires GPU host — see HOST_TASKS.md.",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"CFG sweep report saved to {path}")