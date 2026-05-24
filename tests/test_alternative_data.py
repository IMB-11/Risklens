"""Tests for AlternativeDataEngine: hotness, dragon-tiger, margin, overheating."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from backend.risk.alternative_data import AlternativeDataEngine


@pytest.fixture
def engine():
    return AlternativeDataEngine()


# ============================================================
# fetch_search_hotness
# ============================================================

class TestFetchSearchHotness:
    @patch("backend.risk.alternative_data.ak")
    def test_normal_return(self, mock_ak, engine):
        df = pd.DataFrame({
            "股票代码": ["600519", "000001"],
            "当前排名": [12, 300],
            "当前热度": [98000.0, 12000.0],
        })
        mock_ak.stock_hot_rank_em.return_value = df
        result = engine.fetch_search_hotness("600519")
        assert result["data_quality"] == "good"
        assert result["hot_rank"] == 12
        assert result["hot_value"] == 98000.0

    @patch("backend.risk.alternative_data.ak")
    def test_stock_not_found(self, mock_ak, engine):
        df = pd.DataFrame({
            "股票代码": ["999999"],
            "当前排名": [50],
            "当前热度": [1000.0],
        })
        mock_ak.stock_hot_rank_em.return_value = df
        result = engine.fetch_search_hotness("600519")
        assert result["data_quality"] == "not_found"
        assert result["hot_rank"] == 0

    @patch("backend.risk.alternative_data.ak")
    def test_api_error(self, mock_ak, engine):
        mock_ak.stock_hot_rank_em.side_effect = RuntimeError("network")
        result = engine.fetch_search_hotness("600519")
        assert result["data_quality"] == "unavailable"
        assert "error" in result

    @patch("backend.risk.alternative_data.ak")
    def test_empty_dataframe(self, mock_ak, engine):
        mock_ak.stock_hot_rank_em.return_value = pd.DataFrame()
        result = engine.fetch_search_hotness("600519")
        assert result["data_quality"] == "unavailable"


# ============================================================
# fetch_dragon_tiger
# ============================================================

class TestFetchDragonTiger:
    @patch("backend.risk.alternative_data.ak")
    def test_normal_return(self, mock_ak, engine):
        df = pd.DataFrame({
            "代码": ["600519"],
            "买入额": [150000000.0],
            "卖出额": [80000000.0],
        })
        mock_ak.stock_lhb_detail_em.return_value = df
        result = engine.fetch_dragon_tiger("600519")
        assert result["data_quality"] == "good"
        assert result["records_count"] == 1
        assert result["total_buy_amount"] == 150000000.0

    @patch("backend.risk.alternative_data.ak")
    def test_no_records(self, mock_ak, engine):
        mock_ak.stock_lhb_detail_em.return_value = pd.DataFrame()
        result = engine.fetch_dragon_tiger("600519")
        assert result["data_quality"] == "not_found"

    @patch("backend.risk.alternative_data.ak")
    def test_api_error(self, mock_ak, engine):
        mock_ak.stock_lhb_detail_em.side_effect = TimeoutError("slow")
        result = engine.fetch_dragon_tiger("600519")
        assert result["data_quality"] == "unavailable"
        assert "error" in result


# ============================================================
# fetch_margin_trading
# ============================================================

class TestFetchMarginTrading:
    @patch("backend.risk.alternative_data.ak")
    def test_normal_return(self, mock_ak, engine):
        df = pd.DataFrame({
            "标的证券代码": ["600519"],
            "融资余额": [500000000.0],
            "融券余额": [100000000.0],
        })
        mock_ak.stock_margin_detail_sse.return_value = df
        result = engine.fetch_margin_trading("600519")
        assert result["data_quality"] == "good"
        assert result["margin_buy_balance"] == 500000000.0
        assert result["margin_ratio"] == pytest.approx(0.1667, abs=0.001)

    @patch("backend.risk.alternative_data.ak")
    def test_no_data(self, mock_ak, engine):
        mock_ak.stock_margin_detail_sse.return_value = None
        result = engine.fetch_margin_trading("600519")
        assert result["data_quality"] == "unavailable"

    @patch("backend.risk.alternative_data.ak")
    def test_api_error(self, mock_ak, engine):
        mock_ak.stock_margin_detail_sse.side_effect = ValueError("bad param")
        result = engine.fetch_margin_trading("600519")
        assert result["data_quality"] == "unavailable"
        assert "error" in result


# ============================================================
# analyze_sentiment_overheating
# ============================================================

class TestAnalyzeSentimentOverheating:
    @patch("backend.risk.alternative_data.ak")
    def test_critical_overheating(self, mock_ak, engine):
        hotness_df = pd.DataFrame({
            "股票代码": ["600519"],
            "当前排名": [10],
            "当前热度": [99000.0],
        })
        margin_df = pd.DataFrame({
            "标的证券代码": ["600519"],
            "融资余额": [100000000.0],
            "融券余额": [50000000.0],
        })
        mock_ak.stock_hot_rank_em.return_value = hotness_df
        mock_ak.stock_margin_detail_sse.return_value = margin_df
        result = engine.analyze_sentiment_overheating("600519")
        assert result["overheating_level"] == "critical"
        assert result["overheating_score"] >= 0.8
        assert len(result["signals"]) > 0

    @patch("backend.risk.alternative_data.ak")
    def test_low_overheating(self, mock_ak, engine):
        hotness_df = pd.DataFrame({
            "股票代码": ["600519"],
            "当前排名": [400],
            "当前热度": [1000.0],
        })
        margin_df = pd.DataFrame({
            "标的证券代码": ["600519"],
            "融资余额": [1000000.0],
            "融券余额": [10000.0],
        })
        mock_ak.stock_hot_rank_em.return_value = hotness_df
        mock_ak.stock_margin_detail_sse.return_value = margin_df
        result = engine.analyze_sentiment_overheating("600519")
        assert result["overheating_level"] == "low"
        assert result["overheating_score"] < 0.2

    @patch("backend.risk.alternative_data.ak")
    def test_both_apis_fail(self, mock_ak, engine):
        mock_ak.stock_hot_rank_em.side_effect = RuntimeError("down")
        mock_ak.stock_margin_detail_sse.side_effect = RuntimeError("down")
        result = engine.analyze_sentiment_overheating("600519")
        assert result["data_quality"] == "unavailable"
        assert result["overheating_score"] == 0.0
        assert result["overheating_level"] == "low"
