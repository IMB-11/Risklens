"""Tests for StressTestingEngine: historical, hypothetical, Monte Carlo, reverse stress tests."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import pytest
from backend.risk.stress_testing import StressTestingEngine


class TestStressTestingEngine:
    def setup_method(self):
        self.engine = StressTestingEngine()
        # Typical daily returns for a stock
        self.returns = [0.001, -0.002, 0.003, -0.001, 0.002,
                        -0.003, 0.001, 0.004, -0.002, 0.001,
                        -0.001, 0.002, -0.004, 0.003, 0.001,
                        -0.002, 0.001, -0.001, 0.002, 0.003,
                        -0.001, 0.001, -0.003, 0.002, 0.001]

    def test_empty_returns(self):
        result = self.engine.run_full_stress_test([])
        assert result["input_summary"]["return_count"] == 1
        assert result["input_summary"]["mean_return"] == 0.0

    def test_basic_run(self):
        result = self.engine.run_full_stress_test(self.returns)
        assert "input_summary" in result
        assert "historical_results" in result
        assert "hypothetical_results" in result
        assert "monte_carlo_results" in result
        assert "reverse_stress_results" in result
        assert "aggregate_stress_risk" in result

    def test_input_summary(self):
        result = self.engine.run_full_stress_test(self.returns)
        summary = result["input_summary"]
        assert summary["return_count"] == len(self.returns)
        assert summary["worst_return"] <= 0.0
        assert summary["best_return"] >= 0.0

    # ------------------------------------------------------------------
    # Historical scenarios
    # ------------------------------------------------------------------

    def test_historical_scenarios_all(self):
        result = self.engine.run_full_stress_test(self.returns)
        hist = result["historical_results"]
        assert len(hist) == 5  # 5 built-in scenarios
        scenario_ids = {r["scenario_id"] for r in hist}
        assert "2015_crash" in scenario_ids
        assert "2020_covid" in scenario_ids

    def test_historical_scenario_output_fields(self):
        result = self.engine.run_full_stress_test(self.returns, historical_scenarios=["2015_crash"])
        assert len(result["historical_results"]) == 1
        scenario = result["historical_results"][0]
        assert scenario["scenario_id"] == "2015_crash"
        assert scenario["name"] == "2015 A-Share Crash"
        assert scenario["market_shock"] == -0.45
        assert scenario["portfolio_loss"] < 0  # Loss should be negative
        assert scenario["liquidity_cost"] >= 0
        assert scenario["duration_days"] == 45

    def test_historical_scenario_with_weights_and_sectors(self):
        weights = [0.4, 0.3, 0.3]
        sectors = ["tech", "consumer", "financials"]
        result = self.engine.run_full_stress_test(
            self.returns,
            weights=weights,
            asset_sectors=sectors,
            historical_scenarios=["2020_covid"],
        )
        scenario = result["historical_results"][0]
        assert scenario["portfolio_loss"] < 0

    def test_historical_unknown_scenario(self):
        result = self.engine.run_full_stress_test(self.returns, historical_scenarios=["nonexistent"])
        assert result["historical_results"] == []

    # ------------------------------------------------------------------
    # Hypothetical scenarios
    # ------------------------------------------------------------------

    def test_hypothetical_no_scenarios(self):
        result = self.engine.run_full_stress_test(self.returns)
        assert result["hypothetical_results"] == []

    def test_hypothetical_scenario(self):
        scenarios = [
            {
                "name": "Rate Shock",
                "market_drop": -0.15,
                "vol_increase": 2.0,
                "rate_change_bp": 100,
                "duration_days": 20,
            }
        ]
        result = self.engine.run_full_stress_test(self.returns, hypothetical_scenarios=scenarios)
        assert len(result["hypothetical_results"]) == 1
        hypo = result["hypothetical_results"][0]
        assert hypo["scenario_id"] == "hypothetical_1"
        assert hypo["name"] == "Rate Shock"
        assert hypo["parameters"]["market_drop"] == -0.15
        assert hypo["parameters"]["vol_increase"] == 2.0

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------

    def test_monte_carlo_basic(self):
        result = self.engine.run_full_stress_test(self.returns, monte_carlo_sims=5000)
        mc = result["monte_carlo_results"]
        assert mc["n_simulations"] == 5000
        assert mc["horizon_days"] == 30
        assert mc["var_95"] >= 0.0
        assert mc["var_99"] >= mc["var_95"]
        assert mc["cvar_95"] >= mc["var_95"]
        assert mc["cvar_99"] >= mc["var_99"]
        assert 0.0 <= mc["probability_of_loss"] <= 1.0

    def test_monte_carlo_loss_distribution(self):
        result = self.engine.run_full_stress_test(self.returns, monte_carlo_sims=10000)
        ld = result["monte_carlo_results"]["loss_distribution"]
        # Percentiles should be monotonically non-decreasing
        assert ld["p5"] <= ld["p25"] <= ld["p50"] <= ld["p75"] <= ld["p95"] <= ld["p99"]

    def test_monte_carlo_drawdown_stats(self):
        result = self.engine.run_full_stress_test(self.returns, monte_carlo_sims=5000)
        mc = result["monte_carlo_results"]
        assert mc["max_drawdown_worst"] <= 0.0
        assert mc["max_drawdown_avg"] <= 0.0

    # ------------------------------------------------------------------
    # Reverse stress test
    # ------------------------------------------------------------------

    def test_reverse_stress_default_targets(self):
        result = self.engine.run_full_stress_test(self.returns)
        reverse = result["reverse_stress_results"]
        assert len(reverse) == 4  # Default: [-0.10, -0.15, -0.20, -0.30]

    def test_reverse_stress_custom_targets(self):
        result = self.engine.run_full_stress_test(self.returns, reverse_targets=[-0.10])
        reverse = result["reverse_stress_results"]
        assert len(reverse) == 1
        assert reverse[0]["target_loss"] == -0.10
        assert reverse[0]["severity"] == "mild"

    def test_reverse_stress_severity_levels(self):
        result = self.engine.run_full_stress_test(self.returns, reverse_targets=[-0.05, -0.12, -0.25, -0.40])
        severities = [r["severity"] for r in result["reverse_stress_results"]]
        assert "mild" in severities
        assert "moderate" in severities
        assert "severe" in severities
        assert "extreme" in severities

    def test_reverse_stress_output_fields(self):
        result = self.engine.run_full_stress_test(self.returns, reverse_targets=[-0.20])
        r = result["reverse_stress_results"][0]
        assert "required_market_shock" in r
        assert "required_vol_multiplier" in r
        assert "estimated_trigger" in r
        assert "probability_estimate" in r
        assert 0.0 <= r["probability_estimate"] <= 1.0

    # ------------------------------------------------------------------
    # Aggregate stress risk
    # ------------------------------------------------------------------

    def test_aggregate_stress_risk(self):
        result = self.engine.run_full_stress_test(self.returns)
        agg = result["aggregate_stress_risk"]
        assert 0.0 <= agg["composite_stress_score"] <= 1.0
        assert agg["stress_level"] in ("low", "medium", "high", "critical")
        assert "component_scores" in agg

    def test_aggregate_composite_bounds(self):
        # Very stable returns should yield low stress
        stable_returns = [0.001] * 50
        result = self.engine.run_full_stress_test(stable_returns)
        agg = result["aggregate_stress_risk"]
        assert agg["composite_stress_score"] <= 1.0

    # ------------------------------------------------------------------
    # Available scenarios
    # ------------------------------------------------------------------

    def test_available_scenarios(self):
        scenarios = self.engine.get_available_scenarios()
        assert len(scenarios) == 5
        for s in scenarios:
            assert "id" in s
            assert "name" in s
            assert "description" in s
            assert "market_shock" in s
            assert "duration_days" in s
