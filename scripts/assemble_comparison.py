#!/usr/bin/env python
"""assemble_comparison.py — Multi-folder comparison table assembly.

Reads forecast_metrics.json + stylized_facts.json + regime_validation.json
from one or more run folders and assembles a unified COMPARISON_TABLE.md.

Usage (CPU, runs from committed JSONs only):
    source .venv/bin/activate
    python scripts/assemble_comparison.py \
        --conditional runs/retrain_d1__20260706-212108__seed0 \
        --vanilla runs/vanilla_eval__20260706-214104__seed0 \
        --cfg-w2 runs/cfg_eval_w2__CPU__seed0 \
        --cfg-w4 runs/cfg_eval_w4__CPU__seed0 \
        --out-dir runs/retrain_d1__20260706-212108__seed0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _load_metrics_json(run_dir: Path, filename: str) -> Dict[str, Any]:
    path = run_dir / "metrics" / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def assemble_comparison_table(
    conditional_dir: Optional[Path] = None,
    vanilla_dir: Optional[Path] = None,
    cfg_w2_dir: Optional[Path] = None,
    cfg_w4_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    out_name: str = "COMPARISON_TABLE.md",
) -> str:
    def _load_run(rd: Optional[Path]) -> Dict[str, Any]:
        if rd is None:
            return {}
        fm = _load_metrics_json(rd, "forecast_metrics.json")
        sf_raw = _load_metrics_json(rd, "stylized_facts.json")
        sf = sf_raw.get("generated", {}) if sf_raw else {}
        rv = _load_metrics_json(rd, "regime_validation.json")
        return {"forecast": fm, "stylized_facts": sf, "regime_validation": rv}

    runs: Dict[str, Dict] = {}
    if conditional_dir:
        runs["conditional"] = _load_run(conditional_dir)

    if vanilla_dir:
        runs["vanilla"] = _load_run(vanilla_dir)

    if cfg_w2_dir:
        runs["conditional (CFG w=2)"] = _load_run(cfg_w2_dir)

    if cfg_w4_dir:
        runs["conditional (CFG w=4)"] = _load_run(cfg_w4_dir)

    # Baselines: read from conditional_dir (where they're stored)
    baseline_dir = conditional_dir

    baseline_names = ["hist_boot", "block_boot", "garch_t"]
    for bn in baseline_names:
        raw = _load_metrics_json(baseline_dir, f"baseline_{bn}.json") if baseline_dir else {}
        if raw:
            runs[bn] = {
                "forecast": raw.get("forecast", {}),
                "stylized_facts": raw.get("stylized_facts", {}),
            }

    # Real reference: from conditional_dir stylized_facts
    real_sf = {}
    if conditional_dir:
        sf = _load_metrics_json(conditional_dir, "stylized_facts.json")
        real_sf = sf.get("real_raw_un_denoised", {})

    COL_ORDER = [
        "Method", "CRPS", "coverage_0.8", "PIT_KS_p",
        "kurtosis", "skewness", "|r|_ACF1", "leverage_lag1",
        "VaR_99", "ES_99", "Hill_idx",
    ]

    def _fmt(v):
        if v is None:
            return "TBD"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = [
        "# COMPARISON_TABLE — Canonical Method Comparison",
        "",
        "Assembled from committed run folders:",
    ]
    for label, rd in [
        ("conditional", conditional_dir),
        ("vanilla", vanilla_dir),
        ("CFG w=2", cfg_w2_dir),
        ("CFG w=4", cfg_w4_dir),
    ]:
        if rd:
            lines.append(f"- {label}: `{rd.name}`")
    lines += [
        "",
        "| " + " | ".join(COL_ORDER) + " |",
        "|" + "|".join(["-" * (len(c) + 2) for c in COL_ORDER]) + "|",
    ]

    # Methods in display order
    method_order = [
        "conditional",
        "conditional (CFG w=2)",
        "conditional (CFG w=4)",
        "vanilla",
        "hist_boot",
        "block_boot",
        "garch_t",
    ]

    for method_label in method_order:
        rd = runs.get(method_label, {})
        fm_data = rd.get("forecast", {})
        sf_data = rd.get("stylized_facts", {})
        row = {
            "Method": method_label,
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
        cells = [_fmt(row.get(c)) for c in COL_ORDER]
        lines.append("| " + " | ".join(cells) + " |")

    # Real reference
    if real_sf:
        real_row = {
            "Method": "**real (test)**",
            "CRPS": None, "coverage_0.8": None, "PIT_KS_p": None,
            "kurtosis": real_sf.get("kurtosis"),
            "skewness": real_sf.get("skewness"),
            "|r|_ACF1": real_sf.get("acf_abs_returns_lag1"),
            "leverage_lag1": real_sf.get("leverage_lag1"),
            "VaR_99": real_sf.get("var_99"),
            "ES_99": real_sf.get("es_99"),
            "Hill_idx": real_sf.get("tail_index_hill"),
        }
        cells = [_fmt(real_row.get(c)) for c in COL_ORDER]
        lines.append("| " + " | ".join(cells) + " |")

    # Regime validation summary (from conditional)
    if conditional_dir:
        rv = _load_metrics_json(conditional_dir, "regime_validation.json")
        if rv:
            lines += [
                "",
                "## Regime Validation (conditional, w=1.0)",
                "",
                "| Dimension | Label | KS p-value | Cohen's d |",
                "|-----------|-------|-----------|-----------|",
            ]
            for dim, dim_res in sorted(rv.items()):
                for label, res in sorted(dim_res.items()):
                    ks = res.get('ks_pvalue', float('nan'))
                    d = res.get('cohens_d', float('nan'))
                    tag = ""
                    if ks > 0.05:
                        tag = " (not sig.)"
                    lines.append(f"| {dim} | {label} | {ks:.4f} | {d:.3f}{tag} |")

    # CFG regime validation comparison if available
    cfg_rv_rows = []
    for cfg_label, cfg_dir in [("w=2", cfg_w2_dir), ("w=4", cfg_w4_dir)]:
        if cfg_dir:
            rv = _load_metrics_json(cfg_dir, "regime_validation.json")
            if rv:
                for dim, dim_res in sorted(rv.items()):
                    for label, res in sorted(dim_res.items()):
                        d_val = res.get('cohens_d', float('nan'))
                        cfg_rv_rows.append((cfg_label, dim, label, d_val))

    if cfg_rv_rows:
        lines += [
            "",
            "## CFG Regime Effect (bear cohens_d)",
            "",
            "| CFG Scale | Cohen's d (bear) |",
            "|-----------|-----------------|",
        ]
        # Also add conditional w=1 bear
        if conditional_dir:
            rv = _load_metrics_json(conditional_dir, "regime_validation.json")
            bear_d = rv.get("market_regime", {}).get("regime_bear", {}).get("cohens_d", float('nan'))
            lines.append(f"| w=1.0 (conditional) | {bear_d:.3f} |")
        for cfg_label, dim, label, d_val in cfg_rv_rows:
            if label == "regime_bear":
                lines.append(f"| {cfg_label} | {d_val:.3f} |")

    lines += [
        "",
        "## Notes",
        "- All metrics in canonical denoised-close log-return space.",
        "- **Real (test)**: un-denoised raw S&P 500 log returns on the test period.",
        "- **Vanilla**: unconditional TimeGrad (no regime conditioning).",
        "- **hist_boot / block_boot / garch_t**: CPU baselines (Phase 3).",
        "- CFG w=2/w=4 rows are full-eval runs (cfg_eval_w2, cfg_eval_w4) —",
        "  not plumbing-scale. Regime d-values above are from those full evals.",
        "- Bear d in the cfg_sweep table (32-window plumbing) differs from the",
        "  full-eval d-values above because of window-count differences.",
        "  The sweep is only used to show the monotonic trend across w.",
    ]

    result = "\n".join(lines)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / out_name
        out_path.write_text(result, encoding="utf-8")
        print(f"Wrote {out_path}")

    return result


def parse_args():
    p = argparse.ArgumentParser(description="Multi-folder comparison table assembly")
    p.add_argument("--conditional", type=Path, default=None, help="Conditional run folder")
    p.add_argument("--vanilla", type=Path, default=None, help="Vanilla run folder")
    p.add_argument("--cfg-w2", type=Path, default=None, help="CFG w=2 run folder")
    p.add_argument("--cfg-w4", type=Path, default=None, help="CFG w=4 run folder")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for COMPARISON_TABLE.md")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    table = assemble_comparison_table(
        conditional_dir=args.conditional,
        vanilla_dir=args.vanilla,
        cfg_w2_dir=args.cfg_w2,
        cfg_w4_dir=args.cfg_w4,
        out_dir=args.out_dir,
    )
    print(table)
