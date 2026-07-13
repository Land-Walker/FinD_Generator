"""Unit tests for src/stress_demo/portfolio_stress.py."""
import numpy as np
from src.stress_demo.portfolio_stress import ExamplePortfolio, compute_stress_comparison
from src.evaluation.stylized_facts import var_es


class TestExamplePortfolio:
    def test_single_asset_portfolio_returns_match_input(self):
        portfolio = ExamplePortfolio(weights=np.array([1.0]))
        returns = np.array([[0.01, -0.02, 0.005], [0.003, -0.001, 0.002]])
        result = portfolio.apply_returns(returns)
        np.testing.assert_array_almost_equal(result, returns)

    def test_paths_shape(self):
        portfolio = ExamplePortfolio()
        returns = np.random.randn(100, 5) * 0.01
        paths = portfolio.portfolio_returns_to_paths(returns)
        assert paths.shape == (100, 5)

    def test_3d_returns_path_shape(self):
        portfolio = ExamplePortfolio()
        returns = np.random.randn(50, 10, 4) * 0.01
        paths = portfolio.portfolio_returns_to_paths(returns)
        assert paths.shape == (50, 10, 4)

    def test_var_es_known_synthetic(self):
        portfolio = ExamplePortfolio()
        np.random.seed(42)
        normal_returns = np.random.randn(10000, 1) * 0.01 - 0.001
        normal_returns = normal_returns.ravel()

        metrics = portfolio.compute_risk_metrics(normal_returns.reshape(10000, 1))

        # VaR at 95%: ~ -0.001 - 1.645*0.01 ≈ -0.01745
        assert metrics["VaR_95"] < -0.01
        assert metrics["VaR_95"] > -0.03

        # ES_95 should be more extreme than VaR_95
        assert metrics["ES_95"] < metrics["VaR_95"]

        # VaR_99 more extreme than VaR_95
        assert metrics["VaR_99"] < metrics["VaR_95"]

    def test_drawdown_all_positive(self):
        portfolio = ExamplePortfolio()
        positive_returns = np.abs(np.random.randn(100, 5) * 0.01)
        metrics = portfolio.compute_risk_metrics(positive_returns)
        assert metrics["drawdown_max_drawdown"] >= 0.0

    def test_drawdown_negative(self):
        portfolio = ExamplePortfolio()
        negative_returns = -np.abs(np.random.randn(100, 5) * 0.01) * 0.05
        metrics = portfolio.compute_risk_metrics(negative_returns)
        assert metrics["drawdown_max_drawdown"] < 0.0


class TestComputeStressComparison:
    def test_all_scenarios(self):
        np.random.seed(0)
        n_scen, n_win, n_steps = 50, 4, 4

        scenario_results = {
            "scenario_returns": np.random.randn(n_scen, n_win, n_steps) * 0.02 - 0.003,
            "unconditional_returns": np.random.randn(n_scen, n_win, n_steps) * 0.01,
            "historical_returns": np.random.randn(200,),
            "cfg_scale": 2.0,
            "regime_spec": {"market_regime": "bear"},
        }

        results = compute_stress_comparison(scenario_results)
        assert "unconditional (w=0)" in results
        assert "stress regime (w=2.0)" in results
        assert "historical (test)" in results

        for method, metrics in results.items():
            assert "VaR_95" in metrics
            assert "ES_95" in metrics
            assert "VaR_99" in metrics
            assert "ES_99" in metrics
            assert "drawdown_max_drawdown" in metrics


class TestVarES:
    def test_var_95_known(self):
        np.random.seed(0)
        data = np.random.randn(100_000)
        var, es = var_es(data, 0.95)
        assert -1.70 < var < -1.60
        assert es < var

    def test_var_99_known(self):
        np.random.seed(0)
        data = np.random.randn(100_000)
        var, es = var_es(data, 0.99)
        assert -2.40 < var < -2.25
        assert es < var

    def test_edge_case_all_same(self):
        data = np.ones(100) * 0.01
        var, es = var_es(data, 0.95)
        assert var == 0.01
        assert np.isclose(es, 0.01, atol=1e-14)
