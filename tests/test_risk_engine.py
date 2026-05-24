"""Tests for RiskControlEngine: public API assess() and assess_portfolio()."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import pytest
from unittest.mock import patch, MagicMock
from backend.risk.risk_engine import RiskControlEngine, Alert, _clip, _quantile, _normalize_weights

# Mock external dependencies that make network calls
MOCK_PATCHES = {
    "backend.risk.risk_engine.alternative_data_engine": MagicMock(),
    "backend.risk.risk_engine.pretrained_headline_risk_scorer": MagicMock(),
}


def _apply_mocks():
    """Return a dict of context managers for mocking external deps."""
    patches = {}
    for target, mock_obj in MOCK_PATCHES.items():
        p = patch(target, mock_obj)
        patches[target] = p.start()
    return patches


def _stop_mocks(patches):
    for target in MOCK_PATCHES:
        patch.stopall()


# ============================================================
# Helper utility tests (no mocking needed)
# ============================================================

class TestHelperFunctions:
    def test_clip_basic(self):
        assert _clip(0.5) == 0.5
        assert _clip(-0.1) == 0.0
        assert _clip(1.5) == 1.0

    def test_quantile_symmetric(self):
        vals = [-2, -1, 0, 1, 2]
        assert _quantile(vals, 0.5) == 0.0
        assert _quantile(vals, 0.0) == -2.0
        assert _quantile(vals, 1.0) == 2.0

    def test_normalize_weights(self):
        assert _normalize_weights([]) == []
        assert _normalize_weights([1.0, 1.0, 1.0]) == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-6)
        assert _normalize_weights([-1.0, 1.0]) == pytest.approx([-0.5, 0.5], abs=1e-6)

    def test_normalize_weights_zero_sum(self):
        result = _normalize_weights([0.0, 0.0, 0.0])
        assert len(result) == 3
        assert sum(result) == pytest.approx(1.0, abs=1e-6)


# ============================================================
# Alert dataclass
# ============================================================

class TestAlert:
    def test_to_dict(self):
        alert = Alert(
            code="TEST_ALERT",
            severity="high",
            message="Test message",
            metric="test_metric",
            value=0.75,
            threshold=0.50,
            stage="L3",
            escalation_action="Freeze new risk",
        )
        d = alert.to_dict()
        assert d["code"] == "TEST_ALERT"
        assert d["severity"] == "high"
        assert d["value"] == 0.75
        assert d["threshold"] == 0.50
        assert d["stage"] == "L3"


# ============================================================
# RiskControlEngine (with mocked external deps)
# ============================================================

@pytest.fixture(autouse=True)
def mock_external_deps():
    """Auto-mock all external API calls for every test in this module."""
    with patch("backend.risk.risk_engine.alternative_data_engine") as mock_alt, \
         patch("backend.risk.risk_engine.pretrained_headline_risk_scorer") as mock_scorer:
        # Default return values for mocks
        mock_scorer.score.return_value = {
            "available": False,
            "semantic_risk": 0.5,
            "model_alias": None,
            "sample_count": 0,
        }
        mock_alt.analyze_sentiment_overheating.return_value = {
            "overheating_score": 0,
            "overheating_level": "low",
            "signals": [],
        }
        mock_alt.fetch_dragon_tiger.return_value = {
            "lhb_count": 0,
            "net_buy_amount": 0.0,
            "data_quality": "unavailable",
        }
        yield


class TestRiskControlEngine:
    def setup_method(self):
        self.engine = RiskControlEngine()

    def test_normalize_profile_valid(self):
        assert self.engine._normalize_profile("moderate") == "moderate"
        assert self.engine._normalize_profile("aggressive") == "aggressive"
        assert self.engine._normalize_profile("conservative") == "conservative"

    def test_normalize_profile_invalid(self):
        assert self.engine._normalize_profile("unknown") == "moderate"
        assert self.engine._normalize_profile("") == "moderate"
        assert self.engine._normalize_profile(None) == "moderate"

    def test_risk_label(self):
        assert self.engine._risk_label("low") == "LOW"
        assert self.engine._risk_label("medium") == "MEDIUM"
        assert self.engine._risk_label("high") == "HIGH"
        assert self.engine._risk_label("critical") == "CRITICAL"

    def test_map_risk_level(self):
        thresholds = (36.0, 56.0, 74.0)
        assert self.engine._map_risk_level(20.0, thresholds) == "low"
        assert self.engine._map_risk_level(45.0, thresholds) == "medium"
        assert self.engine._map_risk_level(65.0, thresholds) == "high"
        assert self.engine._map_risk_level(80.0, thresholds) == "critical"

    def test_assess_empty_news(self):
        result = self.engine.assess("贵州茅台", "moderate", [])
        assert result["stock_name"] == "贵州茅台"
        assert result["risk_profile"] == "moderate"
        assert 0.0 <= result["risk_score"] <= 100.0
        assert result["risk_level"] in ("low", "medium", "high", "critical")
        assert "risk_label" in result
        assert "quantile_risk" in result
        assert "factor_breakdown" in result
        assert "alerts" in result
        assert "escalation_strategy" in result
        assert "control_actions" in result

    def test_assess_with_news(self):
        news = [
            {"title": "业绩增长", "content": "公司营收超预期", "sentiment_type": "positive", "source": "reuters"},
            {"title": "行业利好", "content": "政策支持", "sentiment_type": "positive", "source": "bloomberg"},
        ]
        inference = {
            "trend_predictions": {
                "1h": {"direction": "up", "confidence": 0.7},
                "1d": {"direction": "up", "confidence": 0.6},
                "1w": {"direction": "up", "confidence": 0.5},
            },
            "summary": {"confidence": 0.8},
        }
        multi_model = {
            "current_suggestion": {"action": "buy"},
            "confidence": 0.75,
            "market_state": {"state": "bull", "confidence": 0.6},
        }
        result = self.engine.assess("贵州茅台", "moderate", news, inference, multi_model)
        assert result["stock_name"] == "贵州茅台"
        assert 0.0 <= result["risk_score"] <= 100.0
        assert "enhanced_risk_metrics" in result
        assert "tail_risk" in result["enhanced_risk_metrics"]
        assert "liquidity_risk" in result["enhanced_risk_metrics"]

    def test_assess_all_profiles(self):
        news = [{"title": "test", "content": "test"}]
        for profile in ["aggressive", "moderate", "conservative"]:
            result = self.engine.assess("贵州茅台", profile, news)
            assert result["risk_profile"] == profile
            assert 0.0 <= result["risk_score"] <= 100.0

    def test_assess_quantile_risk_fields(self):
        result = self.engine.assess("贵州茅台", "moderate", [])
        qr = result["quantile_risk"]
        assert "var_alpha" in qr
        assert "cvar_alpha" in qr
        assert "var_1d" in qr
        assert "cvar_1d" in qr
        assert "var_norm" in qr
        assert "cvar_norm" in qr
        assert "var_methods" in qr
        assert "historical" in qr["var_methods"]
        assert "parametric" in qr["var_methods"]
        assert "monte_carlo" in qr["var_methods"]

    def test_assess_factor_breakdown(self):
        result = self.engine.assess("贵州茅台", "moderate", [])
        fb = result["factor_breakdown"]
        expected_factors = {"sentiment", "trend", "volatility", "event", "uncertainty", "market_regime"}
        assert expected_factors.issubset(set(fb.keys()))
        for factor_name, factor_data in fb.items():
            assert "score" in factor_data
            assert "weight" in factor_data
            assert "contribution" in factor_data

    def test_assess_controls(self):
        result = self.engine.assess("贵州茅台", "moderate", [])
        controls = result["control_actions"]
        assert "action" in controls
        assert "position_limit" in controls
        assert "risk_budget_hint" in controls
        assert "next_steps" in controls
        assert controls["action"] in (
            "normal_monitoring", "heightened_monitoring", "risk_alert", "human_review_and_reduce"
        )

    def test_assess_escalation(self):
        result = self.engine.assess("贵州茅台", "moderate", [])
        esc = result["escalation_strategy"]
        assert esc["current_stage"] in ("L1", "L2", "L3", "L4")
        assert "next_action" in esc

    def test_build_empty_assessment(self):
        result = self.engine.build_empty_assessment("测试股票", "moderate")
        assert result["stock_name"] == "测试股票"
        assert result["risk_score"] == 50.0
        assert result["risk_level"] == "high"
        assert len(result["alerts"]) > 0

    def test_assess_portfolio_empty(self):
        result = self.engine.assess_portfolio([])
        assert result["portfolio_size"] == 0
        assert result["risk_level"] == "high"

    def test_assess_portfolio_basic(self):
        assets = [
            {"stock_name": "A", "weight": 0.5, "sector": "tech", "style": "growth",
             "returns": [0.01, -0.02, 0.03, -0.01, 0.02, -0.01, 0.01, 0.03, -0.02, 0.01] * 3},
            {"stock_name": "B", "weight": 0.3, "sector": "finance", "style": "value",
             "returns": [0.005, -0.01, 0.015, -0.005, 0.01, -0.005, 0.005, 0.015, -0.01, 0.005] * 3},
            {"stock_name": "C", "weight": 0.2, "sector": "consumer", "style": "growth",
             "returns": [0.008, -0.012, 0.02, -0.008, 0.015, -0.008, 0.008, 0.02, -0.012, 0.008] * 3},
        ]
        result = self.engine.assess_portfolio(assets, risk_profile="moderate")
        assert result["portfolio_size"] == 3
        assert 0.0 <= result["risk_score"] <= 100.0
        assert "quantile_risk" in result
        assert "exposure_breakdown" in result
        assert "enhanced_risk_metrics" in result
        assert "alerts" in result

    def test_assess_portfolio_with_leverage(self):
        assets = [
            {"stock_name": "A", "weight": 0.6, "sector": "tech", "leverage": 1.5},
            {"stock_name": "B", "weight": 0.4, "sector": "finance", "leverage": 1.0},
        ]
        result = self.engine.assess_portfolio(
            assets,
            gross_leverage=1.3,
            net_leverage=0.2,
        )
        assert result["portfolio_size"] == 2
        assert "exposure_breakdown" in result

    def test_assess_portfolio_with_covariance(self):
        assets = [
            {"stock_name": "A", "weight": 0.5, "returns": [0.01, -0.02, 0.03] * 5},
            {"stock_name": "B", "weight": 0.5, "returns": [0.005, -0.01, 0.015] * 5},
        ]
        cov = [[0.0004, 0.0001], [0.0001, 0.0002]]
        result = self.engine.assess_portfolio(assets, covariance_matrix=cov)
        assert result["portfolio_size"] == 2

    def test_profile_config_consistency(self):
        for profile in ["aggressive", "moderate", "conservative"]:
            config = RiskControlEngine.PROFILE_CONFIG[profile]
            weights = config["weights"]
            assert abs(sum(weights.values()) - 1.0) < 0.01
            assert config["var_alpha"] > 0.5
            assert config["cvar_alpha"] > 0.5
            assert config["var_cap"] > 0
            assert config["cvar_cap"] > 0
