"""portfolio_stress.py — Apply scenario returns to a configurable example portfolio.

Computes VaR(95/99), ES(95/99), and max-drawdown distribution under:
(i) unconditional (w=0), (ii) each stress regime at w=2 and w=4,
(iii) historical test returns as reference.

Reuses fixed per-path drawdown logic from stylized_facts.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.evaluation.stylized_facts import var_es, drawdown_distribution


class ExamplePortfolio:
    def __init__(
        self,
        weights: np.ndarray = None,
        names: List[str] = None,
        initial_value: float = 1_000_000.0,
    ):
        if weights is None:
            weights = np.array([1.0], dtype=np.float64)
        if names is None:
            names = [f"asset_{i}" for i in range(len(weights))]

        self.weights = np.asarray(weights, dtype=np.float64) / np.sum(weights)
        self.names = names
        self.initial_value = float(initial_value)

    def apply_returns(
        self,
        scenario_returns: np.ndarray,
    ) -> np.ndarray:
        """Apply scenario returns to the portfolio.

        scenario_returns: (n_scenarios, n_steps) or (n_scenarios, n_windows, n_steps)
            Log returns for the single asset (S&P 500 proxy).

        Returns: portfolio log returns, shape (n_scenarios, n_steps) or (n_scenarios, n_windows, n_steps)
            For a single-asset portfolio these are identical to the input,
            but this method generalizes to multi-asset when available.
        """
        return scenario_returns * self.weights[0]

    def portfolio_returns_to_paths(
        self,
        portfolio_log_returns: np.ndarray,
    ) -> np.ndarray:
        """Convert log returns to price paths (cumulative), shape (n_paths, n_steps)."""
        if portfolio_log_returns.ndim == 3:
            n_scen, n_win, n_steps = portfolio_log_returns.shape
            flat = portfolio_log_returns.reshape(n_scen * n_win, n_steps)
            paths = np.exp(np.cumsum(flat, axis=1))
            return paths.reshape(n_scen, n_win, n_steps)
        paths = np.exp(np.cumsum(portfolio_log_returns, axis=-1))
        return paths

    def compute_risk_metrics(
        self,
        scenario_returns: np.ndarray,
        confidence_levels: List[float] = None,
        label: str = "",
    ) -> Dict[str, Any]:
        """Compute VaR, ES, and max-drawdown from scenario log returns.

        scenario_returns: (n_scenarios, n_steps) log returns.
        Drawdown uses per-path computation via stylized_facts.drawdown_distribution.

        Returns dict of risk metrics.
        """
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]

        # Each scenario path's cumulative return (terminal value)
        if scenario_returns.ndim == 1:
            terminal_rets = scenario_returns
        elif scenario_returns.ndim == 2:
            terminal_rets = np.sum(scenario_returns, axis=1)
        elif scenario_returns.ndim == 3:
            terminal_rets = np.sum(scenario_returns, axis=-1).ravel()
        else:
            raise ValueError(f"Unexpected scenario_returns shape: {scenario_returns.shape}")

        metrics = {}
        for alpha in confidence_levels:
            var, es = var_es(terminal_rets, alpha)
            metrics[f"VaR_{int(alpha*100)}"] = var
            metrics[f"ES_{int(alpha*100)}"] = es

        # Drawdown: use per-path computation from stylized_facts
        if scenario_returns.ndim == 3:
            dd_input = scenario_returns.reshape(-1, scenario_returns.shape[-1])
        elif scenario_returns.ndim == 2:
            dd_input = scenario_returns
        else:
            dd_input = scenario_returns.reshape(1, -1)

        dd = drawdown_distribution(dd_input)
        for k, v in dd.items():
            metrics[f"drawdown_{k}"] = v

        metrics["mean_return"] = float(np.mean(terminal_rets))
        metrics["std_return"] = float(np.std(terminal_rets))
        metrics["n_paths"] = len(terminal_rets)
        metrics["label"] = label

        return metrics


def compute_stress_comparison(
    scenario_results: Dict[str, Any],
    portfolio: ExamplePortfolio = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute risk metrics for stress scenario, unconditional, and historical.

    scenario_results: dict from scenario_run.run_scenario() with keys:
        scenario_returns, unconditional_returns, historical_returns
    """
    if portfolio is None:
        portfolio = ExamplePortfolio()

    results = {}

    # Unconditional (w=0)
    uncond = scenario_results.get("unconditional_returns")
    if uncond is not None:
        results["unconditional (w=0)"] = portfolio.compute_risk_metrics(
            uncond, label="unconditional (w=0)"
        )

    # Stress scenario at the specified cfg_scale
    scenario = scenario_results.get("scenario_returns")
    cfg_scale = scenario_results.get("cfg_scale", 2.0)
    regime_spec = scenario_results.get("regime_spec", {})
    regime_label = ",".join(f"{k}={v}" for k, v in regime_spec.items())
    if scenario is not None:
        results[f"stress regime (w={cfg_scale})"] = portfolio.compute_risk_metrics(
            scenario, label=f"stress {regime_label} (w={cfg_scale})"
        )

    # Historical test returns
    hist = scenario_results.get("historical_returns")
    if hist is not None:
        results["historical (test)"] = portfolio.compute_risk_metrics(
            hist, label="historical (test)"
        )

    return results


def write_stress_var_table(
    stress_results: Dict[str, Dict[str, Any]],
    out_path: Path,
    regime_spec: Optional[Dict[str, str]] = None,
) -> str:
    lines = [
        "# Stress VaR/ES Comparison",
        "",
    ]
    if regime_spec:
        lines.append(f"Regime: {', '.join(f'{k}={v}' for k, v in regime_spec.items())}")
    lines += [
        "",
        "| Method | VaR_95 | ES_95 | VaR_99 | ES_99 | Max DD | Mean DD | Mean Return | Std Return |",
        "|--------|--------|-------|--------|-------|--------|---------|-------------|------------|",
    ]

    ROW_KEYS = ["VaR_95", "ES_95", "VaR_99", "ES_99",
                "drawdown_max_drawdown", "drawdown_mean_drawdown",
                "mean_return", "std_return"]

    for method, metrics in stress_results.items():
        cells = [method]
        for k in ROW_KEYS:
            v = metrics.get(k, "-")
            if isinstance(v, float):
                cells.append(f"{v:.5f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")

    result = "\n".join(lines)
    out_path.write_text(result, encoding="utf-8")
    return result


def generate_stress_fan_chart(
    scenario_results: Dict[str, Any],
    out_path: Path,
    title: str = "Stress Scenario Fan Chart",
    n_paths_to_plot: int = 50,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping fan chart plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    scenario = scenario_results.get("scenario_returns")
    uncond = scenario_results.get("unconditional_returns")

    for ax, data, label in [
        (axes[0], uncond, "Unconditional (w=0)"),
        (axes[1], scenario, f"Stress (cfg={scenario_results.get('cfg_scale', 2.0)})"),
    ]:
        if data is None:
            ax.set_title(f"{label}: no data")
            continue
        # Take first n_paths_to_plot paths, each path is the terminal return
        # data shape: (n_scenarios, n_windows, n_steps) or (n_scenarios, n_steps)
        if data.ndim == 3:
            display = data[:, 0, :]
        else:
            display = data

        n = min(n_paths_to_plot, display.shape[0])
        for i in range(n):
            cum = np.exp(np.cumsum(display[i]))
            ax.plot(cum, alpha=0.3, linewidth=0.5, color="steelblue")

        median_cum = np.exp(np.cumsum(np.median(display, axis=0)))
        ax.plot(median_cum, color="darkred", linewidth=1.5, label="median")
        ax.set_title(label)
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative return")
        ax.legend()

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved fan chart to {out_path}")
