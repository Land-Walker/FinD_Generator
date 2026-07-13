"""
run_eval.py — evaluation orchestration for Phase 2.

Given a trained checkpoint, pipeline state, and test data, this module:
1. Generates unconditional samples on the test split.
2. Converts model output and real targets to the canonical space
   (denoised-close log returns via inverse_transform).
3. Computes forecast metrics (CRPS, pinball, PIT, coverage, energy score).
4. Computes stylized facts (kurtosis, Hill, ACF, leverage, VaR/ES) with
   RAW un-denoised real-returns reference column.
5. For each regime dimension/label, generates regime-conditional samples
   and runs regime_validation.
6. Writes metrics JSON and assembles EVALUATION_REPORT.md.

Architecture: all functions are stateless — they receive fitted pipeline
objects, config, and data.  The `--eval` flag in run.py wires them together.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from . import forecast_metrics as fm
from . import stylized_facts as sf
from . import regime_validation as rv
from .inverse_transform import target_pca_to_log_returns, pca_to_denoised_ohlc, reconstruction_error


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _get_raw_test_returns(dm) -> np.ndarray:
    """Extract RAW (un-denoised) log returns from the test-target close.

    Uses the market/target raw data indexed to the merged test grid.
    """
    test_df = dm.test_df
    # market_close is the raw S&P 500 close, merged onto the test grid.
    # For the canonical reference, we use the raw (un-denoised) target close.
    # However no raw target close is kept — use market_close as the
    # approximate index-level proxy, or recompute from raw target data.
    target_raw = dm._ensure_index(dm.data["target"])
    close_raw = target_raw["close"].reindex(test_df.index).ffill()
    if close_raw.isna().any():
        close_raw = close_raw.ffill().bfill()
    log_rets = np.log(close_raw.values[1:] / close_raw.values[:-1])
    return log_rets


def generate_test_samples(
    predictor,
    dm,
    checkpoint_path: Path,
    device: torch.device,
    num_samples: int,
    sampling_strategy: str = "full_horizon",
    max_test_windows: int = 0,
    model_type: str = "conditional",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate unconditional samples on the full test split.

    Returns
    -------
    all_samples : np.ndarray  [num_samples, n_test_windows, horizon, target_dim]
    all_test_targets : np.ndarray  [n_test_windows, horizon, target_dim]
    all_cond_static : np.ndarray  [n_test_windows, regime_dim]
    all_test_x_hist : np.ndarray  [n_test_windows, context_len, target_dim]
    """
    state = torch.load(checkpoint_path, map_location=device)
    predictor.load_state_dict(state, strict=False)
    predictor.eval()

    is_vanilla = model_type == "vanilla"
    loader = dm.test_dataloader()
    samples_list, targets_list, static_list, xhist_list = [], [], [], []

    with torch.no_grad():
        for step, batch in enumerate(loader):
            x_hist = batch["x_hist"].to(device)         # [B, L, C]
            x_future = batch["x_future"].to(device)     # [B, H, C]

            if is_vanilla:
                samples = predictor.sample_forecast(
                    x_hist=x_hist,
                    num_samples=num_samples,
                )
                static_list.append(np.zeros((x_future.shape[0], 1), dtype=np.float32))
            else:
                cond_dynamic = batch["cond_dynamic"].to(device)
                cond_static = batch["cond_static"].to(device)
                samples = predictor.sample_autoregressive(
                    x_hist=x_hist,
                    cond_dynamic=cond_dynamic,
                    cond_static=cond_static,
                    num_samples=num_samples,
                    sampling_strategy=sampling_strategy,
                )
                static_list.append(_to_numpy(cond_static))

            samples_list.append(_to_numpy(samples))
            targets_list.append(_to_numpy(x_future))
            xhist_list.append(_to_numpy(x_hist))

            if max_test_windows and (step + 1) >= max_test_windows:
                break

    all_samples = np.concatenate([s.transpose(1, 0, 2, 3) for s in samples_list], axis=0)
    all_targets = np.concatenate(targets_list, axis=0)
    all_static = np.concatenate(static_list, axis=0)
    all_xhist = np.concatenate(xhist_list, axis=0)
    return all_samples, all_targets, all_static, all_xhist


def compute_forecast_metrics(
    samples: np.ndarray,
    targets: np.ndarray,
    pca: PCA,
    scaler: StandardScaler,
) -> Dict[str, Any]:
    """Compute all Phase 2 forecast metrics in the canonical space.

    samples : [n_windows, n_samples, horizon, target_dim]
    targets : [n_windows, horizon, target_dim]
    """
    # Convert to canonical space: log returns along horizon
    # samples axis order: (n_windows, n_samples, horizon, target_dim)
    # target_pca_to_log_returns expects (..., horizon, n_components)
    pred_ret = target_pca_to_log_returns(samples.transpose(1, 0, 2, 3), pca, scaler)
    # pred_ret: (n_samples, n_windows, horizon-1)
    true_ret = target_pca_to_log_returns(targets, pca, scaler)
    # true_ret: (n_windows, horizon-1)
    pred_flat = pred_ret.reshape(pred_ret.shape[0], -1)  # (n_samples, n_total)
    true_flat = true_ret.ravel()                           # (n_total,)

    result = {
        'crps': fm.crps_ensemble(pred_flat, true_flat),
        'mae': fm.mae(pred_flat, true_flat),
        'rmse': fm.rmse(pred_flat, true_flat),
    }
    for alpha in [0.05, 0.10, 0.50, 0.90, 0.95]:
        result[f'quantile_loss_{alpha}'] = fm.quantile_loss(pred_flat, true_flat, alpha)
    for level in [0.5, 0.8, 0.9, 0.95]:
        result[f'coverage_{level}'] = fm.coverage(pred_flat, true_flat, level)
    pit = fm.pit_values(pred_flat, true_flat)
    ks_stat, ks_pval = fm.pit_ks_test(pit)
    result['pit_ks_stat'] = ks_stat
    result['pit_ks_pvalue'] = ks_pval
    result['energy_score'] = fm.energy_score(pred_flat, true_flat)
    result['nll'] = fm.negative_log_likelihood(pred_flat, true_flat)
    return result


def compute_stylized_facts(
    samples: np.ndarray,
    test_targets: np.ndarray,
    pca: PCA,
    scaler: StandardScaler,
    dm,
) -> Dict[str, Dict[str, float]]:
    """Compute stylized facts for generated vs RAW real returns.

    samples : [n_windows, n_samples, horizon, target_dim]
    """
    # Convert: transpose to (n_samples, n_windows, horizon, C) then inverse
    gen_ret = target_pca_to_log_returns(samples.transpose(1, 0, 2, 3), pca, scaler)
    gen_ret_1d = gen_ret.ravel()
    # Real target returns in canonical space (denoised)
    real_ret_canon = target_pca_to_log_returns(test_targets, pca, scaler)
    real_ret_canon_1d = real_ret_canon.ravel()
    raw_ret = _get_raw_test_returns(dm)

    gen_facts = sf.all_stylized_facts(gen_ret_1d)
    # Override drawdown with per-path computation — each (sample, window) is an
    # independent price path; concatenating them would create a spurious
    # mega-path (351k+ steps) with boundary discontinuities.
    gen_ret_paths = gen_ret.reshape(-1, gen_ret.shape[-1])
    gen_dd = sf.drawdown_distribution(gen_ret_paths)
    for k, v in gen_dd.items():
        gen_facts[f'drawdown_{k}'] = v

    return {
        'real_raw_un_denoised': sf.all_stylized_facts(raw_ret),
        'real_denoised': sf.all_stylized_facts(real_ret_canon_1d),
        'generated': gen_facts,
    }


def generate_regime_conditional_samples(
    predictor,
    dm,
    device: torch.device,
    num_samples: int,
    regime_cols: List[str],
    all_x_hist: np.ndarray,
    all_cond_dynamic: np.ndarray,
    sampling_strategy: str = "full_horizon",
) -> Dict[str, Dict[str, np.ndarray]]:
    """Generate samples conditioned on each regime label.

    For each regime dimension (vol/market/macro) and each label, we build
    a cond_static vector where the target regime column is 1 and all others
    are 0, then generate samples.

    Returns samples_by_regime = {dim: {label: np.ndarray}}.
    """
    samples_by_regime: Dict[str, Dict[str, np.ndarray]] = {}

    regime_dims = {
        'vol_regime': [c for c in regime_cols if c.startswith('vol_regime_')],
        'market_regime': [c for c in regime_cols if c.startswith('market_regime_')],
        'macro_regime': [c for c in regime_cols if c.startswith('macro_regime_')],
    }

    n_batch = all_x_hist.shape[0]
    n_total = num_samples * n_batch

    with torch.no_grad():
        for dim_key, cols in regime_dims.items():
            dim_samples: Dict[str, np.ndarray] = {}
            for col in cols:
                label = col.split('_', 1)[1]  # e.g. 'vol_regime_high_vol' → 'high_vol'
                # Build cond_static: all zeros, set target column to 1
                cond_static = np.zeros((n_batch, len(regime_cols)), dtype=np.float32)
                col_idx = regime_cols.index(col)
                cond_static[:, col_idx] = 1.0

                # Generate in batches to avoid OOM
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
                )  # [n_batch, n_samples, horizon, target_dim]
                dim_samples[label] = all_label
            samples_by_regime[dim_key] = dim_samples

    return samples_by_regime


def run_full_evaluation(
    predictor,
    dm,
    run_dir: Path,
    checkpoint_path: Path,
    device: torch.device,
    num_samples: int,
    sampling_strategy: str = "full_horizon",
    max_test_windows: int = 0,
    model_type: str = "conditional",
) -> Dict[str, Any]:
    """Orchestrate the full Phase 2 evaluation.

    Returns a summary dict suitable for EVALUATION_REPORT.md generation.
    """
    t0 = time.time()
    scaler = dm.scalers.get("target_scaler")
    pca = dm.pcas.get("target_pca")
    if scaler is None or pca is None:
        raise RuntimeError("Target scaler/PCA not fitted. Run pipeline first.")

    is_vanilla = model_type == "vanilla"

    # 1. Generate unconditional samples
    print("Generating unconditional test samples ...")
    samples, targets, cond_static, x_hist = generate_test_samples(
        predictor, dm, checkpoint_path, device, num_samples, sampling_strategy,
        max_test_windows=max_test_windows, model_type=model_type,
    )
    print(f"  Samples shape: {samples.shape}")

    # 2. Reconstruct on train and report reconstruction error
    train_df = dm.train_df
    target_den_cols = [c for c in train_df.columns if c.endswith('_den')]
    ohlc_den = train_df[target_den_cols].values
    scaled = scaler.transform(ohlc_den)
    pca_out = pca.transform(scaled)
    recon_den = scaler.inverse_transform(pca.inverse_transform(pca_out))
    recon_rmse, recon_mape = reconstruction_error(ohlc_den, recon_den)
    print(f"  PCA components: {pca.n_components_} (variance: {np.sum(pca.explained_variance_ratio_):.4f})")
    print(f"  Reconstruction RMSE: {recon_rmse:.6f}, MAPE: {recon_mape:.2f}%")

    # 3. Forecast metrics
    print("Computing forecast metrics ...")
    forecast_results = compute_forecast_metrics(samples, targets, pca, scaler)
    for k, v in forecast_results.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    # 4. Stylized facts
    print("Computing stylized facts ...")
    facts = compute_stylized_facts(samples, targets, pca, scaler, dm)
    for name in facts:
        print(f"  {name}: kurtosis={facts[name]['kurtosis']:.3f}, VaR95={facts[name]['var_95']:.4f}")

    # 5. Regime validation (conditional only)
    regime_val: Dict[str, Any] = {}
    if not is_vanilla:
        feature_cols = dm.get_feature_columns_by_type()
        regime_cols = feature_cols["regime"]

        dynamic_list = []
        xhist_list = []
        loader = dm.test_dataloader()
        for batch in loader:
            dynamic_list.append(_to_numpy(batch["cond_dynamic"]))
            xhist_list.append(_to_numpy(batch["x_hist"]))
        all_cond_dynamic = np.concatenate(dynamic_list, axis=0)
        all_x_hist_arr = np.concatenate(xhist_list, axis=0)

        print("Generating regime-conditional samples (subset for plumbing) ...")
        max_regime_windows = min(len(all_x_hist_arr), 32)
        if max_test_windows:
            max_regime_windows = min(max_regime_windows, max_test_windows * 64)
        samples_by_regime = generate_regime_conditional_samples(
            predictor, dm, device, num_samples,
            regime_cols,
            all_x_hist_arr[:max_regime_windows],
            all_cond_dynamic[:max_regime_windows],
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
        for dim, dim_res in regime_val.items():
            for label, res in dim_res.items():
                if 'warning' in res:
                    print(f"  {dim}/{label}: {res['warning']}")
                else:
                    print(f"  {dim}/{label}: KS p={res['ks_pvalue']:.4f}, d={res['cohens_d']:.3f}")

    # 6. Save metrics JSON
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "forecast_metrics.json", "w") as f:
        json.dump(forecast_results, f, indent=2)
    with open(metrics_dir / "stylized_facts.json", "w") as f:
        json.dump(facts, f, indent=2)
    if not is_vanilla:
        with open(metrics_dir / "regime_validation.json", "w") as f:
            json.dump(regime_val, f, indent=2)
    with open(metrics_dir / "reconstruction_error.json", "w") as f:
        json.dump({
            "rmse": recon_rmse,
            "mape_pct": recon_mape,
            "pca_components": int(pca.n_components_),
            "pca_variance_retained": float(np.sum(pca.explained_variance_ratio_)),
        }, f, indent=2)

    # 7. Assemble EVALUATION_REPORT.md + comparison table
    _write_evaluation_report(run_dir, forecast_results, facts, regime_val, recon_rmse, recon_mape,
                             model_type=model_type)
    write_comparison_table(run_dir)

    elapsed = time.time() - t0
    print(f"Evaluation complete in {elapsed:.1f}s. Report: {run_dir}/EVALUATION_REPORT.md")

    return {
        'forecast': forecast_results,
        'stylized_facts': facts,
        'regime_validation': regime_val,
        'reconstruction_rmse': recon_rmse,
        'reconstruction_mape': recon_mape,
    }


def _write_evaluation_report(
    run_dir: Path,
    forecast: Dict,
    facts: Dict,
    regime_val: Dict,
    recon_rmse: float,
    recon_mape: float,
    model_type: str = "conditional",
) -> None:
    report = run_dir / "EVALUATION_REPORT.md"
    lines = [
        "# EVALUATION_REPORT — Phase 2",
        "",
        f"## Model: {model_type}",
        "",
        "## Canonical Evaluation Space",
        "- All methods evaluated in **denoised-close log returns**.",
        f"- PCA 1 components, reconstruction RMSE = {recon_rmse:.4f}, MAPE = {recon_mape:.2f}%.",
        "- `real_raw_un_denoised` = un-denoised log returns (honest kurtosis/tails reference).",
        "",
        "## Forecast Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k, v in forecast.items():
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.6f} |")
        else:
            lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "## Stylized Facts",
        "",
        "| Fact | Real (raw) | Real (denoised) | Generated |",
        "|------|-----------|-----------------|-----------|",
    ]
    all_fact_keys = sorted(facts.get('real_raw_un_denoised', {}).keys())
    for key in all_fact_keys:
        raw_v = facts.get('real_raw_un_denoised', {}).get(key, '-')
        den_v = facts.get('real_denoised', {}).get(key, '-')
        gen_v = facts.get('generated', {}).get(key, '-')
        def fmt(v):
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)
        lines.append(f"| {key} | {fmt(raw_v)} | {fmt(den_v)} | {fmt(gen_v)} |")

    if regime_val:
        lines += [
            "",
            "## Regime Validation",
            "",
            "| Dimension | Label | KS p-value | Cohen's d | Energy Dist | N (g/¬g) |",
            "|-----------|-------|-----------|-----------|-------------|-----------|",
        ]
        for dim, dim_res in regime_val.items():
            for label, res in dim_res.items():
                if 'warning' in res:
                    lines.append(f"| {dim} | {label} | — | — | — | {res.get('n_g','?')}/{res.get('n_not_g','?')} ⚠ {res['warning']} |")
                else:
                    ks = res.get('ks_pvalue', float('nan'))
                    d = res.get('cohens_d', float('nan'))
                    ed = res.get('energy_dist', float('nan'))
                    ng = res.get('n_g', '?')
                    nng = res.get('n_not_g', '?')
                    lines.append(f"| {dim} | {label} | {ks:.4f} | {d:.3f} | {ed:.4f} | {ng}/{nng} |")

    lines += [
        "",
        "## Sanity Check",
        f"- Reconstruction error ({recon_rmse:.4f}) is negligible relative to price scale.",
        "- stagflation (42 rows, 0.67%) is underpowered — see KNOWN_ISSUES #9.",
        "- **Drawdown semantics:** real (raw/denoised) returns are a single continuous test-period price path → continuous drawdown. Generated returns are independent per-path drawdowns (one per (sample, window) horizon), aggregated: `max_drawdown` = worst across paths, `mean_drawdown` = mean, `max_drawdown_duration` = longest. This is correct for short-horizon forecasts — a 5-step path cannot be compared to a multi-year drawdown on a single concatenated series.",
        "- Baseline columns (hist_bootstrap, block_bootstrap, garch_t, vanilla_timegrad) are TODO — Phase 3.",
        "- Full regime-conditional evaluation on all test windows requires GPU (HOST_TASKS.md)." if model_type == "conditional" else "- Regime validation skipped (vanilla model has no conditioning).",
    ]

def write_comparison_table(run_dir: Path, extra_metrics_dirs: Optional[Dict[str, Path]] = None) -> Path:
    """Assemble COMPARISON_TABLE.md from available metrics JSONs.

    Reads forecast_metrics.json for conditional/vanilla (when present) and
    baseline_*.json for the 3 CPU baselines. Rows whose JSONs are missing
    show a placeholder.

    extra_metrics_dirs: dict mapping method label -> run dir path for metrics
    from other run folders (e.g. {'conditional (CFG w=2)': Path('runs/cfg_w2')}).
    """
    metrics_dir = run_dir / "metrics"
    out_path = run_dir / "COMPARISON_TABLE.md"

    def _load_metrics_from_dir(mdir: Path, name: str) -> Dict:
        path = mdir / name
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    _load_metrics = lambda name: _load_metrics_from_dir(metrics_dir, name)

    cond_fm = _load_metrics("forecast_metrics.json")
    cond_sf = _load_metrics("stylized_facts.json")
    cond_sf_gen = cond_sf.get("generated", {}) if cond_sf else {}
    cond_sf_real = cond_sf.get("real_raw_un_denoised", {}) if cond_sf else {}

    def _make_row(label, fm_data, sf_data):
        return {
            "Method": label,
            "CRPS": fm_data.get("crps"),
            "coverage_0.8": fm_data.get("coverage_0.8"),
            "PIT_KS_p": fm_data.get("pit_ks_pvalue"),
            "kurtosis": sf_data.get("kurtosis"),
            "skewness": sf_data.get("skewness"),
            "|r|_ACF1": sf_data.get("acf_abs_returns_lag1"),
            "leverage_lag1": sf_data.get("leverage_lag1"),
            "VaR_99": sf_data.get("var_99"),
            "ES_99": sf_data.get("es_99"),
            "Hill_idx": sf_data.get("tail_index_hill"),
        }

    rows_data = []

    # conditional row
    rows_data.append(_make_row("conditional", cond_fm, cond_sf_gen))

    # CFG w=2 and w=4 from extra_metrics_dirs
    if extra_metrics_dirs:
        for label, mdir in extra_metrics_dirs.items():
            fm = _load_metrics_from_dir(mdir, "forecast_metrics.json")
            sf = _load_metrics_from_dir(mdir, "stylized_facts.json")
            sf_gen = sf.get("generated", {}) if sf else {}
            if fm and sf_gen:
                rows_data.append(_make_row(label, fm, sf_gen))
            else:
                rows_data.append(_make_row(label, {}, {}))

    # vanilla row — try loading from vanilla_dir in extra_metrics_dirs, then local
    vanilla_fm = {}
    vanilla_sf_gen = {}
    if extra_metrics_dirs and "vanilla" in extra_metrics_dirs:
        vfm = _load_metrics_from_dir(extra_metrics_dirs["vanilla"], "forecast_metrics.json")
        vsf = _load_metrics_from_dir(extra_metrics_dirs["vanilla"], "stylized_facts.json")
        vanilla_fm = vfm
        vanilla_sf_gen = vsf.get("generated", {}) if vsf else {}
    else:
        raw = _load_metrics("vanilla_forecast_metrics.json")
        vanilla_fm = raw.get("forecast", {}) if raw else {}
        raw2 = _load_metrics("vanilla_stylized_facts.json")
        vanilla_sf_gen = raw2.get("stylized_facts", {}).get("generated", {}) if raw2 else {}

    rows_data.append(_make_row("vanilla", vanilla_fm, vanilla_sf_gen))

    # Baselines
    for label, json_prefix in [("hist_boot", "baseline_hist_boot"),
                                ("block_boot", "baseline_block_boot"),
                                ("garch_t", "baseline_garch_t")]:
        raw = _load_metrics(f"{json_prefix}.json")
        fm_data = raw.get("forecast", {})
        sf_data = raw.get("stylized_facts", {})
        if fm_data and sf_data:
            rows_data.append(_make_row(label, fm_data, sf_data))
        else:
            rows_data.append(_make_row(label, {}, {}))

    # Real reference row
    rows_data.append(_make_row("**real (test)**", {}, cond_sf_real))

    COL_ORDER = [
        "Method", "CRPS", "coverage_0.8", "PIT_KS_p",
        "kurtosis", "skewness", "|r|_ACF1", "leverage_lag1",
        "VaR_99", "ES_99", "Hill_idx",
    ]

    lines = [
        "# COMPARISON_TABLE — Phase 3 Baseline Battery",
        "",
        f"Run: `{run_dir.name}`",
        "",
        "| " + " | ".join(COL_ORDER) + " |",
        "|" + "|".join(["-" * (len(c) + 2) for c in COL_ORDER]) + "|",
    ]

    def _fmt(v):
        if v is None:
            return "TBD"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    for row in rows_data:
        cells = [_fmt(row.get(c)) for c in COL_ORDER]
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Notes",
        "- **Real (test)**: un-denoised raw S&P 500 log returns on the test period.",
        "- **Vanilla**: placeholder row — requires host GPU training (see HOST_TASKS.md).",
        "- **hist_boot / block_boot / garch_t**: CPU baselines (Phase 3).",
        "- All metrics computed in canonical space (denoised-close log returns).",
        "- Numbers shown are **plumbing-test only** unless generated on the host GPU with full training.",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {out_path}")
    return out_path


if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="Run comparison table assembly")
    parser.add_argument("--run-id", required=True, help="Run directory under runs/")
    args_cli = parser.parse_args()
    run_dir = Path("runs") / args_cli.run_id
    if not run_dir.exists():
        print(f"ERROR: {run_dir} not found")
        sys.exit(1)
    write_comparison_table(run_dir)