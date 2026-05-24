"""Tests for AnomalyDetectionEngine."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from backend.risk.anomaly_detection import (
    AnomalyDetectionEngine,
    AnomalyAlert,
    KEYWORD_DICT,
    anomaly_detection_engine,
    _clip,
    _z_scores,
    _safe_mean,
    _safe_stdev,
)


# ============================================================
# Helper / utility tests
# ============================================================

class TestHelperFunctions:
    def test_clip_inside_range(self):
        assert _clip(0.5) == 0.5

    def test_clip_below_lower(self):
        assert _clip(-0.3) == 0.0

    def test_clip_above_upper(self):
        assert _clip(2.5) == 1.0

    def test_clip_custom_bounds(self):
        assert _clip(5, lower=3, upper=10) == 5
        assert _clip(1, lower=3, upper=10) == 3
        assert _clip(99, lower=3, upper=10) == 10

    def test_z_scores_uniform(self):
        vals = [5.0, 5.0, 5.0]
        zs = _z_scores(vals)
        assert all(z == 0.0 for z in zs)

    def test_z_scores_known_values(self):
        vals = [1.0, 2.0, 3.0]
        zs = _z_scores(vals)
        assert len(zs) == 3
        # Mean=2, pstdev=0.8165
        assert zs[1] == pytest.approx(0.0, abs=1e-6)
        assert zs[2] > 0
        assert zs[0] < 0

    def test_safe_mean_empty(self):
        assert _safe_mean([]) == 0.0

    def test_safe_stdev_single_element(self):
        assert _safe_stdev([42.0]) == 0.0


# ============================================================
# AnomalyAlert dataclass
# ============================================================

class TestAnomalyAlert:
    def test_to_dict_fields(self):
        alert = AnomalyAlert(
            alert_type="wash_trade",
            severity="high",
            message="测试消息",
            matched_keywords=["自买自卖"],
            confidence=0.8,
        )
        d = alert.to_dict()
        assert d["alert_type"] == "wash_trade"
        assert d["severity"] == "high"
        assert d["matched_keywords"] == ["自买自卖"]
        assert d["confidence"] == pytest.approx(0.8, abs=1e-4)

    def test_default_matched_keywords(self):
        alert = AnomalyAlert(
            alert_type="test", severity="low", message="msg"
        )
        assert alert.matched_keywords == []
        assert alert.confidence == 0.0


# ============================================================
# Keyword matching
# ============================================================

class TestKeywordMatch:
    def setup_method(self):
        self.engine = AnomalyDetectionEngine()

    def test_empty_text_returns_zero(self):
        results = self.engine._keyword_match("")
        for info in results.values():
            assert info["score"] == 0.0
            assert info["matched"] == []

    def test_single_keyword_hit(self):
        results = self.engine._keyword_match("该公司涉嫌内幕交易")
        assert "insider_trading" in results
        assert "内幕" in results["insider_trading"]["matched"]
        assert results["insider_trading"]["score"] > 0

    def test_multiple_keywords_same_type(self):
        text = "频繁撤单行为严重，多次幌骗"
        results = self.engine._keyword_match(text)
        matched = results["high_frequency"]["matched"]
        assert "频繁撤单" in matched
        assert "幌骗" in matched
        # "撤单" also matches as a substring of "频繁撤单", so 3 hits
        assert len(matched) >= 2
        assert results["high_frequency"]["score"] > 0

    def test_cross_type_detection(self):
        text = "内幕消息泄露，涉及操纵市场"
        results = self.engine._keyword_match(text)
        assert results["insider_trading"]["score"] > 0
        assert results["market_manipulation"]["score"] > 0

    def test_no_match_returns_zero(self):
        results = self.engine._keyword_match("今天天气很好")
        for info in results.values():
            assert info["score"] == 0.0
            assert info["matched"] == []


# ============================================================
# Volume-price anomaly
# ============================================================

class TestVolumePriceAnomaly:
    def setup_method(self):
        self.engine = AnomalyDetectionEngine()

    def test_no_data_returns_zero(self):
        assert self.engine._volume_price_anomaly(None, None) == 0.0
        assert self.engine._volume_price_anomaly([], []) == 0.0

    def test_insufficient_data(self):
        assert self.engine._volume_price_anomaly([1, 2], [1, 2]) == 0.0

    def test_stable_data_low_score(self):
        vol = [100.0] * 10
        price = [10.0] * 10
        assert self.engine._volume_price_anomaly(vol, price) == 0.0

    def test_spike_detection(self):
        # Last two data points: huge volume + huge price move
        vol = [100.0] * 8 + [500.0, 600.0]
        price = [10.0] * 8 + [10.0, 15.0]
        score = self.engine._volume_price_anomaly(vol, price)
        assert score > 0.1


# ============================================================
# Temporal clustering
# ============================================================

class TestTemporalClustering:
    def setup_method(self):
        self.engine = AnomalyDetectionEngine()

    def test_no_data(self):
        assert self.engine._temporal_clustering(None) == 0.0
        assert self.engine._temporal_clustering([]) == 0.0

    def test_short_data(self):
        assert self.engine._temporal_clustering([1, 2, 3]) == 0.0

    def test_clustered_data(self):
        # Alternating high/low should have low autocorrelation
        alt = [10, 100, 10, 100, 10, 100, 10, 100]
        score_alt = self.engine._temporal_clustering(alt)
        # Trending data should have high autocorrelation
        trend = [10, 20, 30, 40, 50, 60, 70, 80]
        score_trend = self.engine._temporal_clustering(trend)
        assert score_trend > score_alt


# ============================================================
# Sentiment spike
# ============================================================

class TestSentimentSpike:
    def setup_method(self):
        self.engine = AnomalyDetectionEngine()

    def test_no_data(self):
        assert self.engine._sentiment_spike(None, None) == 0.0

    def test_insufficient_data(self):
        assert self.engine._sentiment_spike([1, 2], [1, 2]) == 0.0

    def test_stable_data_low_score(self):
        vol = [100.0] * 10
        price = [10.0] * 10
        assert self.engine._sentiment_spike(vol, price) == 0.0

    def test_spike_data_high_score(self):
        vol = [100.0] * 8 + [500.0, 800.0]
        price = [10.0] * 8 + [10.0, 15.0]
        score = self.engine._sentiment_spike(vol, price)
        assert score > 0.05


# ============================================================
# Full detect() pipeline
# ============================================================

class TestDetect:
    def setup_method(self):
        self.engine = AnomalyDetectionEngine()

    def test_returns_all_keys(self):
        result = self.engine.detect("测试股票")
        expected_keys = {
            "anomaly_score", "anomaly_level",
            "trading_anomalies", "compliance_risks", "alerts",
        }
        assert set(result.keys()) == expected_keys

    def test_clean_input_low_score(self):
        result = self.engine.detect("普通股票", news_text="今日正常交易")
        assert result["anomaly_score"] < 0.15
        assert result["anomaly_level"] in ("low", "medium")

    def test_news_keyword_drives_score_up(self):
        result = self.engine.detect(
            "某股票",
            news_text="涉嫌操纵市场坐庄内幕泄露自买自卖",
        )
        assert result["anomaly_score"] > 0.2
        assert len(result["alerts"]) > 0

    def test_volume_price_spike_adds_alert(self):
        vol = [100.0] * 8 + [500.0, 600.0]
        price = [10.0] * 8 + [10.0, 15.0]
        result = self.engine.detect(
            "异常股", volume_data=vol, price_data=price
        )
        alert_types = [a["alert_type"] for a in result["alerts"]]
        assert "volume_price_anomaly" in alert_types

    def test_level_mapping(self):
        # Force a high score via many keyword hits
        result = self.engine.detect(
            "高风险股",
            news_text="频繁撤单幌骗诱多自买自卖倒仓对敲"
                       "大单买入巨量内幕泄露突击入股"
                       "操纵坐庄拉高出货监管处罚立案调查",
        )
        assert result["anomaly_score"] > 0.4
        assert result["anomaly_level"] in ("medium", "high", "critical")

    def test_trading_and_compliance_split(self):
        result = self.engine.detect(
            "综合风险股",
            news_text="频繁撤单内幕消息操纵市场",
        )
        trading_types = {a["alert_type"] for a in result["trading_anomalies"]}
        compliance_types = {a["alert_type"] for a in result["compliance_risks"]}
        # high_frequency is trading, insider_trading and market_manipulation are compliance
        assert "high_frequency" in trading_types
        assert "insider_trading" in compliance_types
        assert "market_manipulation" in compliance_types

    def test_module_singleton_is_engine(self):
        assert isinstance(anomaly_detection_engine, AnomalyDetectionEngine)


# ============================================================
# Keyword dictionary completeness
# ============================================================

class TestKeywordDict:
    def test_has_six_types(self):
        assert len(KEYWORD_DICT) == 6

    def test_all_keywords_are_strings(self):
        for atype, keywords in KEYWORD_DICT.items():
            assert isinstance(keywords, list)
            for kw in keywords:
                assert isinstance(kw, str)
                assert len(kw) > 0

    def test_expected_types_present(self):
        expected = {
            "high_frequency", "wash_trade", "large_order_anomaly",
            "insider_trading", "market_manipulation", "regulatory_risk",
        }
        assert set(KEYWORD_DICT.keys()) == expected
