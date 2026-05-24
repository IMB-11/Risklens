"""Tests for risk_metrics: TailRiskMetrics, LiquidityRiskMetrics, CreditRiskMetrics, ConcentrationRiskMetrics."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import pytest
from backend.risk.risk_metrics import (
    TailRiskMetrics,
    LiquidityRiskMetrics,
    CreditRiskMetrics,
    ConcentrationRiskMetrics,
    _quantile,
    _clip,
    _safe_mean,
    _safe_stdev,
)


# ============================================================
# Helper utilities
# ============================================================

class TestHelperFunctions:
    def test_clip_within_range(self):
        assert _clip(0.5) == 0.5

    def test_clip_below_lower(self):
        assert _clip(-0.1) == 0.0

    def test_clip_above_upper(self):
        assert _clip(1.5) == 1.0

    def test_clip_custom_bounds(self):
        assert _clip(5, lower=0, upper=10) == 5
        assert _clip(-1, lower=0, upper=10) == 0
        assert _clip(15, lower=0, upper=10) == 10

    def test_quantile_empty(self):
        assert _quantile([], 0.5) == 0.0

    def test_quantile_single_value(self):
        assert _quantile([0.5], 0.5) == 0.5

    def test_quantile_known_values(self):
        vals = list(range(1, 101))
        assert _quantile(vals, 0.0) == 1.0
        assert _quantile(vals, 1.0) == 100.0
        assert _quantile(vals, 0.5) == 50.5

    def test_safe_mean_empty(self):
        assert _safe_mean([]) == 0.0

    def test_safe_mean_with_values(self):
        assert _safe_mean([1.0, 2.0, 3.0]) == 2.0

    def test_safe_stdev_single(self):
        assert _safe_stdev([1.0]) == 0.0

    def test_safe_stdev_empty(self):
        assert _safe_stdev([]) == 0.0


# ============================================================
# TailRiskMetrics
# ============================================================

class TestTailRiskMetrics:
    def setup_method(self):
        self.engine = TailRiskMetrics()

    def test_empty_returns(self):
        result = self.engine.compute([])
        assert result["sample_size"] == 0
        assert result["data_quality"] == "no_data"
        assert result["expected_shortfall"] == 0.0
        assert result["cvar_1d"] == 0.0

    def test_single_return(self):
        result = self.engine.compute([0.01])
        assert result["sample_size"] == 1
        assert result["data_quality"] == "estimated"
        assert result["expected_shortfall"] >= 0.0

    def test_known_distribution(self):
        # Uniform returns: 100 values from -0.05 to +0.05
        returns = [i * 0.001 - 0.05 for i in range(100)]
        result = self.engine.compute(returns, alpha=0.95)
        assert result["sample_size"] == 100
        assert result["es_alpha"] == 0.95
        assert result["expected_shortfall"] > 0.0
        assert result["cvar_1d"] > 0.0
        assert result["loss_mean"] > 0.0
        assert result["loss_std"] > 0.0

    def test_multi_timeframe_cvar(self):
        # Generate 100 daily returns
        returns = [0.001 * (i % 10 - 5) for i in range(100)]
        result = self.engine.compute(returns)
        # 1w CVaR should be >= 1d CVaR in absolute terms for same alpha
        assert result["cvar_1w"] >= 0.0
        assert result["cvar_1m"] >= 0.0

    def test_extreme_loss_probability(self):
        # Returns with a fat tail
        returns = [0.01] * 90 + [-0.10] * 10
        result = self.engine.compute(returns)
        assert result["actual_extreme_count"] >= 0
        assert 0.0 <= result["actual_extreme_ratio"] <= 1.0
        assert 0.0 <= result["extreme_loss_probability"] <= 1.0

    def test_all_returns_positive(self):
        returns = [0.01, 0.02, 0.03, 0.01, 0.02]
        result = self.engine.compute(returns)
        # With all positive returns, losses are negative
        # VaR is clamped to >= 0, but CVaR can be negative
        assert result["expected_shortfall"] >= 0.0
        assert result["sample_size"] == 5

    def test_output_fields_complete(self):
        result = self.engine.compute([0.01, -0.02, 0.03, -0.01])
        expected_keys = {
            "expected_shortfall", "es_alpha", "cvar_1d", "cvar_1w", "cvar_1m",
            "extreme_loss_probability", "actual_extreme_ratio", "actual_extreme_count",
            "threshold_3sigma", "loss_mean", "loss_std", "sample_size", "data_quality",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_inf_returns_filtered(self):
        returns = [0.01, float('inf'), -0.02, float('-inf'), 0.03]
        result = self.engine.compute(returns)
        assert result["sample_size"] == 3  # inf values filtered out


# ============================================================
# LiquidityRiskMetrics
# ============================================================

class TestLiquidityRiskMetrics:
    def setup_method(self):
        self.engine = LiquidityRiskMetrics()

    def test_no_data(self):
        result = self.engine.compute()
        assert result["composite_liquidity_score"] >= 0.0
        assert result["bid_ask_quality"] == "default"
        assert result["lcr_quality"] == "default"

    def test_ohlc_data(self):
        # Simulate 20 days of OHLC data
        prices = [100 + i * 0.5 for i in range(20)]
        highs = [p + 1.0 for p in prices]
        lows = [p - 1.0 for p in prices]
        volumes = [1000000 + i * 10000 for i in range(20)]
        result = self.engine.compute(
            volume_data=volumes,
            price_data=prices,
            high_data=highs,
            low_data=lows,
        )
        assert result["bid_ask_quality"] == "estimated"
        assert result["bid_ask_spread_proxy"] >= 0.0
        assert result["composite_liquidity_score"] >= 0.0
        assert result["composite_liquidity_score"] <= 1.0

    def test_volume_price_only(self):
        prices = [100.0, 101.0, 99.0, 102.0, 100.5]
        volumes = [1e6, 1.1e6, 0.9e6, 1.2e6, 1.0e6]
        result = self.engine.compute(volume_data=volumes, price_data=prices)
        assert result["kyle_lambda_quality"] in ("estimated", "insufficient_data")
        assert 0.0 <= result["composite_liquidity_score"] <= 1.0

    def test_financial_data_lcr(self):
        financial = {
            "liquid_assets": 5000,
            "current_liabilities": 3000,
            "total_assets": 20000,
        }
        result = self.engine.compute(financial_data=financial)
        assert result["lcr_quality"] == "estimated"
        assert result["lcr_approximation"] == pytest.approx(5000 / 3000, rel=1e-3)

    def test_financial_data_rough_lcr(self):
        financial = {
            "total_assets": 20000,
            "current_liabilities": 5000,
        }
        result = self.engine.compute(financial_data=financial)
        assert result["lcr_quality"] == "rough_estimate"

    def test_amihud_illiquidity(self):
        prices = [100.0, 100.5, 101.0, 100.0, 102.0]
        volumes = [1e6, 1e6, 1e6, 1e6, 1e6]
        result = self.engine.compute(volume_data=volumes, price_data=prices)
        assert result["amihud_illiquidity"] >= 0.0

    def test_composite_score_bounds(self):
        result = self.engine.compute(
            volume_data=[1e6] * 20,
            price_data=[100.0 + i for i in range(20)],
        )
        assert 0.0 <= result["composite_liquidity_score"] <= 1.0


# ============================================================
# CreditRiskMetrics
# ============================================================

class TestCreditRiskMetrics:
    def setup_method(self):
        self.engine = CreditRiskMetrics()

    def test_no_data(self):
        result = self.engine.compute()
        assert result["altman_z_score"] is None
        assert result["probability_of_default"] is None
        assert result["z_score_zone"] == "unknown"

    def test_healthy_company(self):
        financial = {
            "total_assets": 100000,
            "working_capital": 20000,
            "retained_earnings": 15000,
            "ebit": 12000,
            "revenue": 80000,
            "total_liabilities": 30000,
        }
        market = {"market_cap": 150000, "equity_volatility": 0.3}
        result = self.engine.compute(financial_data=financial, market_data=market)
        assert result["altman_z_score"] is not None
        assert result["altman_z_score"] > 2.99  # Should be in "safe" zone
        assert result["z_score_zone"] == "safe"
        assert result["probability_of_default"] is not None
        assert 0.0 <= result["probability_of_default"] <= 1.0
        assert 0.0 <= result["composite_credit_score"] <= 1.0

    def test_distressed_company(self):
        financial = {
            "total_assets": 100000,
            "working_capital": -5000,
            "retained_earnings": -20000,
            "ebit": -8000,
            "revenue": 20000,
            "total_liabilities": 95000,
        }
        market = {"market_cap": 5000, "equity_volatility": 0.8}
        result = self.engine.compute(financial_data=financial, market_data=market)
        assert result["altman_z_score"] is not None
        assert result["altman_z_score"] < 1.81  # Should be in "distressed" zone
        assert result["z_score_zone"] == "distressed"
        assert result["probability_of_default"] is not None
        assert result["probability_of_default"] > 0.1  # High default probability

    def test_gray_zone(self):
        # Z-score between 1.81 and 2.99 = gray zone
        # Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
        # Need Z around 2.5 to be in gray zone
        financial = {
            "total_assets": 100000,
            "working_capital": 8000,    # X1 = 0.08
            "retained_earnings": 6000,   # X2 = 0.06
            "ebit": 4000,               # X3 = 0.04
            "revenue": 80000,            # X5 = 0.8
            "total_liabilities": 40000,
        }
        market = {"market_cap": 80000}  # X4 = 80000/40000 = 2.0
        # Z = 1.2*0.08 + 1.4*0.06 + 3.3*0.04 + 0.6*2.0 + 1.0*0.8
        # Z = 0.096 + 0.084 + 0.132 + 1.2 + 0.8 = 2.312 (gray zone)
        result = self.engine.compute(financial_data=financial, market_data=market)
        assert result["z_score_zone"] == "gray"

    def test_credit_spread_from_volatility(self):
        market = {"equity_volatility": 0.5}
        result = self.engine.compute(market_data=market)
        assert result["credit_spread_proxy"] == pytest.approx(0.5 * 0.3, rel=1e-3)
        assert result["spread_quality"] == "estimated"

    def test_credit_spread_default(self):
        result = self.engine.compute()
        assert result["credit_spread_proxy"] == 0.02
        assert result["spread_quality"] == "default"


# ============================================================
# ConcentrationRiskMetrics
# ============================================================

class TestConcentrationRiskMetrics:
    def setup_method(self):
        self.engine = ConcentrationRiskMetrics()

    def test_empty_portfolio(self):
        result = self.engine.compute()
        assert result["hhi"] == 0.0
        assert result["normalized_hhi"] == 0.0
        assert result["total_positions"] == 0

    def test_equal_weight_portfolio(self):
        weights = [0.25, 0.25, 0.25, 0.25]
        # Use higher limit so equal-weight doesn't breach
        result = self.engine.compute(portfolio_weights=weights, single_asset_limit=0.30)
        assert result["total_positions"] == 4
        # Equal weight: HHI = 4 * (0.25)^2 = 0.25
        assert result["hhi"] == pytest.approx(0.25, abs=1e-4)
        assert result["normalized_hhi"] == pytest.approx(0.0, abs=1e-3)
        assert result["single_asset_breach_count"] == 0

    def test_concentrated_portfolio(self):
        weights = [0.80, 0.10, 0.10]
        result = self.engine.compute(
            portfolio_weights=weights,
            asset_names=["A", "B", "C"],
            single_asset_limit=0.30,
        )
        assert result["largest_weight"] == pytest.approx(0.80, abs=1e-4)
        assert result["single_asset_breach_count"] == 1
        assert result["single_asset_breaches"][0]["asset"] == "A"

    def test_sector_breaches(self):
        weights = [0.35, 0.35, 0.30]
        sectors = ["tech", "tech", "finance"]
        result = self.engine.compute(
            portfolio_weights=weights,
            asset_sectors=sectors,
            sector_limit=0.50,
        )
        # Tech sector total = 0.70 > 0.50
        assert result["sector_breach_count"] == 1
        assert result["sector_breaches"][0]["sector"] == "tech"

    def test_effective_positions(self):
        weights = [0.50, 0.25, 0.25]
        result = self.engine.compute(portfolio_weights=weights)
        # HHI = 0.25 + 0.0625 + 0.0625 = 0.375
        assert result["hhi"] == pytest.approx(0.375, abs=1e-4)
        # Effective N = 1 / 0.375 = 2.666...
        assert result["effective_positions"] == pytest.approx(2.67, abs=0.1)

    def test_composite_score_bounds(self):
        weights = [0.50, 0.30, 0.20]
        result = self.engine.compute(portfolio_weights=weights)
        assert 0.0 <= result["composite_concentration_score"] <= 1.0

    def test_single_asset_portfolio(self):
        weights = [1.0]
        result = self.engine.compute(portfolio_weights=weights)
        assert result["normalized_hhi"] == 1.0
        assert result["total_positions"] == 1
