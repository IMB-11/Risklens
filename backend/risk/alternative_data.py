"""Alternative data engine: search hotness, dragon-tiger, margin trading signals."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import akshare as ak


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_code(stock_code: str) -> str:
    """Strip leading zeros and whitespace from stock codes."""
    return stock_code.strip().lstrip("0") if stock_code else ""


class AlternativeDataEngine:
    """Fetch and analyse alternative data (hotness, LHB, margin) via akshare."""

    # ------------------------------------------------------------------
    # Search hotness (Xueqiu / Snowball search ranking)
    # ------------------------------------------------------------------

    def fetch_search_hotness(self, stock_code: str) -> Dict[str, Any]:
        """Return Xueqiu search hotness data for *stock_code*.

        Returns a dict with keys: stock_code, hot_rank, hot_value,
        data_quality, fetch_time, raw_data.
        """
        base: Dict[str, Any] = {
            "stock_code": stock_code,
            "hot_rank": 0,
            "hot_value": 0.0,
            "data_quality": "unavailable",
            "fetch_time": datetime.now(timezone.utc).isoformat(),
            "raw_data": [],
        }
        try:
            df = ak.stock_hot_rank_em()
            if df is None or df.empty:
                return base

            code_col = next(
                (c for c in df.columns if "代码" in c or "code" in c.lower()),
                df.columns[0] if len(df.columns) > 0 else None,
            )
            if code_col is None:
                return base

            matched = df[df[code_col].astype(str).str.contains(stock_code)]
            if matched.empty:
                base["data_quality"] = "not_found"
                return base

            row = matched.iloc[0]
            rank_col = next(
                (c for c in df.columns if "排名" in c or "rank" in c.lower()), None
            )
            value_col = next(
                (c for c in df.columns if "热度" in c or "value" in c.lower()), None
            )
            base["hot_rank"] = int(_to_float(row.get(rank_col, 0))) if rank_col else 0
            base["hot_value"] = (
                _to_float(row.get(value_col, 0.0)) if value_col else 0.0
            )
            base["data_quality"] = "good"
            base["raw_data"] = row.to_dict()
            return base
        except Exception as exc:  # noqa: BLE001
            base["error"] = str(exc)
            return base

    # ------------------------------------------------------------------
    # Dragon-tiger list (龙虎榜)
    # ------------------------------------------------------------------

    def fetch_dragon_tiger(self, stock_code: str) -> Dict[str, Any]:
        """Return dragon-tiger list data for *stock_code*.

        Returns a dict with keys: stock_code, records_count,
        total_buy_amount, total_sell_amount, data_quality, fetch_time,
        records.
        """
        base: Dict[str, Any] = {
            "stock_code": stock_code,
            "records_count": 0,
            "total_buy_amount": 0.0,
            "total_sell_amount": 0.0,
            "data_quality": "unavailable",
            "fetch_time": datetime.now(timezone.utc).isoformat(),
            "records": [],
        }
        try:
            today_str = datetime.now().strftime("%Y%m%d")
            df = ak.stock_lhb_detail_em(
                start_date=today_str, end_date=today_str
            )
            if df is None or df.empty:
                base["data_quality"] = "not_found"
                return base

            code_col = next(
                (c for c in df.columns if "代码" in c or "code" in c.lower()),
                df.columns[0] if len(df.columns) > 0 else None,
            )
            if code_col is None:
                return base

            matched = df[df[code_col].astype(str).str.contains(stock_code)]
            if matched.empty:
                base["data_quality"] = "not_found"
                return base

            buy_col = next(
                (c for c in df.columns if "买入" in c or "buy" in c.lower()), None
            )
            sell_col = next(
                (c for c in df.columns if "卖出" in c or "sell" in c.lower()), None
            )
            total_buy = matched[buy_col].apply(_to_float).sum() if buy_col else 0.0
            total_sell = matched[sell_col].apply(_to_float).sum() if sell_col else 0.0

            base["records_count"] = len(matched)
            base["total_buy_amount"] = float(total_buy)
            base["total_sell_amount"] = float(total_sell)
            base["data_quality"] = "good"
            base["records"] = matched.to_dict(orient="records")
            return base
        except Exception as exc:  # noqa: BLE001
            base["error"] = str(exc)
            return base

    # ------------------------------------------------------------------
    # Margin trading (融资融券)
    # ------------------------------------------------------------------

    def fetch_margin_trading(self, stock_code: str) -> Dict[str, Any]:
        """Return margin trading data for *stock_code* (SSE).

        Returns a dict with keys: stock_code, margin_buy_balance,
        short_sell_balance, margin_ratio, data_quality, fetch_time,
        raw_data.
        """
        base: Dict[str, Any] = {
            "stock_code": stock_code,
            "margin_buy_balance": 0.0,
            "short_sell_balance": 0.0,
            "margin_ratio": 0.0,
            "data_quality": "unavailable",
            "fetch_time": datetime.now(timezone.utc).isoformat(),
            "raw_data": [],
        }
        try:
            df = ak.stock_margin_detail_sse()
            if df is None or df.empty:
                return base

            code_col = next(
                (c for c in df.columns if "代码" in c or "code" in c.lower()),
                df.columns[0] if len(df.columns) > 0 else None,
            )
            if code_col is None:
                return base

            matched = df[df[code_col].astype(str).str.contains(stock_code)]
            if matched.empty:
                base["data_quality"] = "not_found"
                return base

            buy_col = next(
                (c for c in df.columns if "融资余额" in c or "margin" in c.lower()),
                None,
            )
            sell_col = next(
                (c for c in df.columns if "融券余额" in c or "short" in c.lower()),
                None,
            )
            row = matched.iloc[0]
            margin_buy = _to_float(row.get(buy_col, 0.0)) if buy_col else 0.0
            short_sell = _to_float(row.get(sell_col, 0.0)) if sell_col else 0.0

            base["margin_buy_balance"] = margin_buy
            base["short_sell_balance"] = short_sell
            if margin_buy > 0:
                base["margin_ratio"] = round(
                    short_sell / (margin_buy + short_sell), 4
                )
            base["data_quality"] = "good"
            base["raw_data"] = row.to_dict()
            return base
        except Exception as exc:  # noqa: BLE001
            base["error"] = str(exc)
            return base

    # ------------------------------------------------------------------
    # Sentiment overheating analysis
    # ------------------------------------------------------------------

    def analyze_sentiment_overheating(
        self, stock_code: str
    ) -> Dict[str, Any]:
        """Combine search hotness + margin data to compute an overheating score.

        Returns a dict with keys: stock_code, overheating_score (0-1),
        overheating_level, signals, hotness_data, margin_data, analyse_time.
        """
        hotness = self.fetch_search_hotness(stock_code)
        margin = self.fetch_margin_trading(stock_code)

        signals: List[str] = []
        score_components: List[float] = []

        # --- hotness component (weight 0.5) ---
        # Lower rank number = more popular, so invert the score.
        if hotness["data_quality"] == "good":
            rank = hotness["hot_rank"]
            if rank > 0:
                hotness_score = _clip(1.0 - rank / 500.0)
            else:
                hotness_score = _clip(hotness["hot_value"] / 100000.0)
            score_components.append(hotness_score * 0.5)
            if rank > 0 and rank <= 50:
                signals.append("search_hotness:extremely_popular")
            elif rank > 0 and rank <= 200:
                signals.append("search_hotness:popular")
        elif hotness["data_quality"] == "not_found":
            score_components.append(0.0)

        # --- margin component (weight 0.5) ---
        if margin["data_quality"] == "good":
            ratio = margin["margin_ratio"]
            margin_score = _clip(ratio * 2.0)
            score_components.append(margin_score * 0.5)
            if ratio > 0.3:
                signals.append("margin:high_short_ratio")
            elif ratio > 0.15:
                signals.append("margin:elevated_short")
        elif margin["data_quality"] == "not_found":
            score_components.append(0.0)

        # If both sources unavailable, mark overall quality accordingly
        overall_quality = "good"
        if (
            hotness["data_quality"] == "unavailable"
            and margin["data_quality"] == "unavailable"
        ):
            overall_quality = "unavailable"

        overheating_score = (
            round(sum(score_components), 4) if score_components else 0.0
        )

        if overheating_score >= 0.8:
            level = "critical"
        elif overheating_score >= 0.5:
            level = "high"
        elif overheating_score >= 0.2:
            level = "medium"
        else:
            level = "low"

        return {
            "stock_code": stock_code,
            "overheating_score": overheating_score,
            "overheating_level": level,
            "signals": signals,
            "hotness_data": hotness,
            "margin_data": margin,
            "data_quality": overall_quality,
            "analyse_time": datetime.now(timezone.utc).isoformat(),
        }


# Module-level singleton used by RiskControlEngine
alternative_data_engine = AlternativeDataEngine()
