"""
FinanceLM-centered prediction coordinator.

This module intentionally uses FinanceLM as the only prediction model and keeps
response schema compatible with existing API consumers.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import os
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except Exception:  # pragma: no cover - dependency may be absent in lightweight runtime
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None

from backend.crawler.runtime_adapters import AsyncNewsCrawlerAdapter, AsyncSocialMediaCrawlerAdapter


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


CONFIG = {
    "model_paths": {
        "financeLM": "financeLM_outputpath_stock_movement_prediction__5",
    },
    "enabled_models": {
        # Single-model prediction path: FinanceLM only.
        "financeLM": _env_flag("ENABLE_MODEL_FINANCELM", True),
        "np_lstm": _env_flag("ENABLE_MODEL_NP_LSTM", False),
        "sentiment": _env_flag("ENABLE_MODEL_SENTIMENT", False),
    },
    "default_weights": {
        "financeLM": 1.0,
        "np_lstm": 0.0,
        "sentiment": 0.0,
    },
    "prediction": {
        "steps": 7,
        "confidence_threshold": 0.58,
    },
}


class LightweightSentimentEngine:
    """News-driven lightweight sentiment aggregator without model dependencies."""

    def __init__(self) -> None:
        self.news_crawler = AsyncNewsCrawlerAdapter()
        self.social_crawler = AsyncSocialMediaCrawlerAdapter()

    async def analyze_sentiment(self, stock_name: str, limit: int = 30) -> Dict[str, Any]:
        tasks = [
            asyncio.create_task(self.news_crawler.crawl(stock_name, limit=max(5, min(limit, 20)))),
            asyncio.create_task(self.social_crawler.crawl(stock_name, limit=max(3, min(limit // 2, 12)))),
        ]
        payloads = await asyncio.gather(*tasks, return_exceptions=True)
        items: List[Dict[str, Any]] = []
        for payload in payloads:
            if isinstance(payload, Exception):
                continue
            items.extend(payload)

        dedup: Dict[str, Dict[str, Any]] = {}
        for item in items:
            digest = item.get("content_hash") or item.get("title") or str(len(dedup))
            if digest not in dedup:
                dedup[digest] = item
        merged = list(dedup.values())
        merged.sort(key=lambda x: -_safe_float(x.get("weight"), 0.5))

        if not merged:
            return {
                "total_news": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "sentiment_score": 0.0,
                "confidence": 0.4,
                "top_news": [],
                "news_items": [],
            }

        positive_count = sum(1 for n in merged if n.get("sentiment_type") == "positive")
        negative_count = sum(1 for n in merged if n.get("sentiment_type") == "negative")
        neutral_count = max(0, len(merged) - positive_count - negative_count)
        sentiment_score = (positive_count - negative_count) / max(1, len(merged))
        avg_weight = sum(_safe_float(n.get("weight"), 0.5) for n in merged) / len(merged)
        confidence = _clip(avg_weight * 0.65 + (abs(sentiment_score) + 0.25) * 0.35, 0.2, 0.95)

        return {
            "total_news": len(merged),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "sentiment_score": round(sentiment_score, 4),
            "confidence": round(confidence, 4),
            "top_news": merged[:5],
            "news_items": merged[:12],
        }


class FinanceLMModel:
    """Wrapper for local FinanceLM model."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or CONFIG["model_paths"]["financeLM"]
        self.tokenizer = None
        self.model = None
        self.generator = None
        self.device = "cpu"
        self._initialized = False
        self._init_error: Optional[str] = None
        self._initialize()

    def _initialize(self) -> None:
        if not os.path.exists(self.model_path):
            self._init_error = f"model path not found: {self.model_path}"
            print(f"[WARN] FinanceLM init skipped: {self._init_error}")
            return
        if AutoTokenizer is None or AutoModelForCausalLM is None or pipeline is None or torch is None:
            self._init_error = "transformers/torch unavailable"
            print(f"[WARN] FinanceLM init skipped: {self._init_error}")
            return

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[INFO] Loading FinanceLM from: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            self.model = self.model.to(self.device)
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                max_new_tokens=220,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.08,
                return_full_text=False,
            )
            self._initialized = True
            print("[OK] FinanceLM ready")
        except Exception as exc:
            self._initialized = False
            self._init_error = str(exc)
            print(f"[ERR] FinanceLM init failed: {str(exc)[:160]}")

    def predict(self, stock_name: str, news_items: List[Dict[str, Any]], stock_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self._initialized or self.generator is None:
            fallback = self._build_fallback_predictions(None, stock_data)
            return {
                "success": False,
                "message": f"FinanceLM unavailable: {self._init_error or 'not initialized'}",
                "trend": "neutral",
                "confidence": 0.35,
                "target_price": fallback[-1] if fallback else None,
                "support_level": min(fallback) if fallback else None,
                "resistance_level": max(fallback) if fallback else None,
                "predictions": fallback,
                "suggestion": "hold",
                "raw_response": "",
            }

        try:
            prompt = self._build_prompt(stock_name, news_items, stock_data)
            response = self.generator(prompt)
            text = response[0]["generated_text"] if response else ""
            return self._parse_response(text, stock_data)
        except Exception as exc:
            fallback = self._build_fallback_predictions(None, stock_data)
            return {
                "success": False,
                "message": str(exc)[:200],
                "trend": "neutral",
                "confidence": 0.35,
                "target_price": fallback[-1] if fallback else None,
                "support_level": min(fallback) if fallback else None,
                "resistance_level": max(fallback) if fallback else None,
                "predictions": fallback,
                "suggestion": "hold",
                "raw_response": "",
            }

    def _build_prompt(self, stock_name: str, news_items: List[Dict[str, Any]], stock_data: Optional[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for i, news in enumerate(news_items[:8]):
            title = str(news.get("title", "")).strip()[:72]
            if not title:
                continue
            sentiment = str(news.get("sentiment_type", "neutral")).strip()
            weight = _safe_float(news.get("weight"), 0.5)
            lines.append(f"{i+1}. [{sentiment}] {title} (权重={weight:.2f})")

        if not lines:
            lines.append("暂无高质量新闻，请给出谨慎预测并明确置信度。")

        latest_price = (stock_data or {}).get("latest_price", "未知")
        change_percent = (stock_data or {}).get("change_percent", "未知")
        volume = (stock_data or {}).get("volume", "未知")

        return f"""
你是资深中文金融分析助手，请仅用中文输出结论。

股票: {stock_name}
当前价格: {latest_price}
当日涨跌幅(%): {change_percent}
成交量: {volume}

相关新闻:
{chr(10).join(lines)}

请严格按以下格式输出（每项都必须出现）：
1) 趋势判断: 上涨/下跌/震荡
2) 置信度: 0-1之间小数
3) 目标价: 数字
4) 支撑位: 数字
5) 阻力位: 数字
6) 未来7天预测: [p1, p2, p3, p4, p5, p6, p7]
7) 投资建议: 买入/持有/卖出

不要输出与模板无关内容。
""".strip()

    def _parse_response(self, response_text: str, stock_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        parsed: Dict[str, Any] = {
            "success": True,
            "raw_response": response_text or "",
            "trend": "neutral",
            "confidence": 0.5,
            "target_price": None,
            "support_level": None,
            "resistance_level": None,
            "predictions": [],
            "suggestion": "hold",
        }
        text = response_text or ""

        try:
            if any(k in text for k in ("上涨", "看涨", "多头", "buy", "bull", "up")):
                parsed["trend"] = "up"
            elif any(k in text for k in ("下跌", "看跌", "空头", "sell", "bear", "down")):
                parsed["trend"] = "down"
            elif any(k in text for k in ("震荡", "横盘", "neutral", "sideways")):
                parsed["trend"] = "sideways"

            confidence_patterns = [
                r"(?:置信度|confidence)\s*[:：=]?\s*([01](?:\.\d+)?)",
                r"(?:置信度|confidence)\s*[:：=]?\s*(\d{1,3}(?:\.\d+)?)\s*%",
            ]
            for pattern in confidence_patterns:
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if match:
                    val = _safe_float(match.group(1), 0.5)
                    if "%" in match.group(0):
                        val = val / 100.0
                    parsed["confidence"] = _clip(val, 0.01, 0.99)
                    break

            for pattern, key in [
                (r"(?:目标价|目标价格|target price)\s*[:：=]?\s*([0-9]+(?:\.[0-9]+)?)", "target_price"),
                (r"(?:支撑位|支撑价格|support)\s*[:：=]?\s*([0-9]+(?:\.[0-9]+)?)", "support_level"),
                (r"(?:阻力位|压力位|resistance)\s*[:：=]?\s*([0-9]+(?:\.[0-9]+)?)", "resistance_level"),
            ]:
                m = re.search(pattern, text, flags=re.IGNORECASE)
                if m:
                    parsed[key] = _safe_float(m.group(1), None)

            week = re.search(
                r"(?:未来\s*7\s*天预测|7天预测|predictions?\s*7d)\s*[:：=]?\s*(.*)",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if week:
                nums = re.findall(r"-?\d+(?:\.\d+)?", week.group(1))
                parsed["predictions"] = [_safe_float(n, 0.0) for n in nums[:7]]

            if not parsed["predictions"]:
                nums = re.findall(r"-?\d+(?:\.\d+)?", text)
                tail = [_safe_float(n, 0.0) for n in nums[-7:]]
                if len(tail) >= 3:
                    parsed["predictions"] = tail[:7]

            if any(k in text for k in ("买入", "加仓", "增持", "buy", "long")):
                parsed["suggestion"] = "buy"
            elif any(k in text for k in ("卖出", "减仓", "止损", "sell", "short")):
                parsed["suggestion"] = "sell"
            elif any(k in text for k in ("持有", "观望", "hold", "wait")):
                parsed["suggestion"] = "hold"

            if not parsed["predictions"]:
                parsed["predictions"] = self._build_fallback_predictions(parsed, stock_data)

            if parsed["target_price"] is None and parsed["predictions"]:
                parsed["target_price"] = parsed["predictions"][-1]
            if parsed["support_level"] is None and parsed["predictions"]:
                parsed["support_level"] = min(parsed["predictions"])
            if parsed["resistance_level"] is None and parsed["predictions"]:
                parsed["resistance_level"] = max(parsed["predictions"])
        except Exception:
            parsed["success"] = False
            parsed["predictions"] = self._build_fallback_predictions(parsed, stock_data)

        return parsed

    def _build_fallback_predictions(self, parsed: Optional[Dict[str, Any]], stock_data: Optional[Dict[str, Any]]) -> List[float]:
        base_price = _safe_float((stock_data or {}).get("latest_price"), 100.0)
        trend = (parsed or {}).get("trend", "neutral")
        confidence = _safe_float((parsed or {}).get("confidence"), 0.5)
        daily = 0.003 + confidence * 0.008
        if trend == "up":
            sign = 1.0
        elif trend == "down":
            sign = -1.0
        else:
            sign = 0.0
        out: List[float] = []
        p = base_price
        for _ in range(CONFIG["prediction"]["steps"]):
            p = p * (1.0 + sign * daily)
            out.append(round(p, 3))
        return out


class MultiModelCoordinator:
    """Single-model coordinator that keeps legacy output schema."""

    def __init__(self):
        self.enabled_models = CONFIG["enabled_models"].copy()
        self.finance_lm = FinanceLMModel() if self.enabled_models.get("financeLM", True) else None
        self.sentiment_engine = LightweightSentimentEngine()
        self.weights = CONFIG["default_weights"].copy()
        print("[OK] MultiModelCoordinator initialized in FinanceLM-only mode")
        print(
            f"   enabled: financeLM={self.enabled_models.get('financeLM')}, "
            f"np_lstm={self.enabled_models.get('np_lstm')}, sentiment={self.enabled_models.get('sentiment')}"
        )

    async def analyze_and_predict(self, stock_name: str, stock_data: Optional[Dict[str, Any]] = None, days: int = 60) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "stock_name": stock_name,
            "analysis_time": datetime.now().isoformat(),
            "model_predictions": {},
            "sentiment_analysis": {},
            "market_state": {},
            "combined_prediction": {},
            "current_suggestion": {},
            "future_forecast": {},
            "confidence": 0.5,
            "weights_used": {"financeLM": 1.0, "np_lstm": 0.0, "sentiment": 0.0},
        }

        sentiment_result = await self._get_sentiment(stock_name)
        result["sentiment_analysis"] = sentiment_result

        market_state = self._build_market_state(sentiment_result)
        result["market_state"] = market_state

        news_items = self._extract_news_items(sentiment_result)
        flm_pred = await self._predict_finance_lm(stock_name, news_items, stock_data)

        model_predictions = {
            "financeLM": flm_pred,
            "np_lstm": {
                "success": False,
                "disabled": True,
                "message": "np_lstm disabled by design (FinanceLM-only prediction pipeline)",
            },
        }
        result["model_predictions"] = model_predictions

        combined = self._combine_predictions(flm_pred)
        result["combined_prediction"] = combined

        suggestion = self._generate_current_suggestion(combined, market_state)
        result["current_suggestion"] = suggestion

        forecast = self._generate_future_forecast(combined, model_predictions)
        result["future_forecast"] = forecast

        result["confidence"] = combined.get("confidence", 0.5)
        return result

    async def _get_sentiment(self, stock_name: str) -> Dict[str, Any]:
        try:
            return await self.sentiment_engine.analyze_sentiment(stock_name, limit=30)
        except Exception as exc:
            return {
                "total_news": 0,
                "positive_count": 0,
                "negative_count": 0,
                "sentiment_score": 0.0,
                "confidence": 0.4,
                "error": str(exc)[:200],
            }

    def _build_market_state(self, sentiment_result: Dict[str, Any]) -> Dict[str, Any]:
        sentiment_score = _safe_float(sentiment_result.get("sentiment_score"), 0.0)
        confidence = _clip(0.45 + abs(sentiment_score) * 0.4, 0.2, 0.9)

        if sentiment_score > 0.25:
            raw_state = "bull"
        elif sentiment_score < -0.25:
            raw_state = "bear"
        elif abs(sentiment_score) < 0.08:
            raw_state = "sideways"
        else:
            raw_state = "uncertain"

        label_map = {
            "bull": "牛市",
            "bear": "熊市",
            "sideways": "震荡市",
            "volatile": "高波动",
            "quiet": "低波动",
            "uncertain": "不确定",
        }

        return {
            "state": raw_state,
            "state_label": label_map.get(raw_state, "不确定"),
            "confidence": _clip(confidence, 0.05, 0.99),
        }

    async def _predict_finance_lm(self, stock_name: str, news_items: List[Dict[str, Any]], stock_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if self.finance_lm is None:
            return {
                "success": False,
                "message": "FinanceLM model instance unavailable",
                "trend": "neutral",
                "confidence": 0.35,
                "predictions": [],
            }
        return await asyncio.to_thread(self.finance_lm.predict, stock_name, news_items, stock_data)

    def _extract_news_items(self, sentiment_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        news_items: List[Dict[str, Any]] = []
        if "top_news" in sentiment_result and isinstance(sentiment_result["top_news"], list):
            news_items.extend(sentiment_result["top_news"])
        if "news_items" in sentiment_result and isinstance(sentiment_result["news_items"], list):
            news_items.extend(sentiment_result["news_items"])
        if not news_items:
            sentiment_score = _safe_float(sentiment_result.get("sentiment_score"), 0.0)
            news_items = [{
                "title": "舆情汇总样本",
                "sentiment_type": "positive" if sentiment_score > 0.2 else "negative" if sentiment_score < -0.2 else "neutral",
                "weight": _safe_float(sentiment_result.get("confidence"), 0.5),
                "publish_time": datetime.now().isoformat(),
            }]
        return news_items[:10]

    def _combine_predictions(self, flm_pred: Dict[str, Any]) -> Dict[str, Any]:
        predictions = flm_pred.get("predictions") or []
        trend = str(flm_pred.get("trend", "neutral")).lower()
        confidence = _clip(_safe_float(flm_pred.get("confidence"), 0.5), 0.05, 0.99)

        if trend not in {"up", "down", "sideways"}:
            if len(predictions) >= 2 and predictions[0]:
                change = (predictions[-1] - predictions[0]) / predictions[0] * 100
                trend = "up" if change > 0.8 else "down" if change < -0.8 else "sideways"
            else:
                trend = "sideways"

        return {
            "trend": trend,
            "confidence": confidence,
            "target_price": flm_pred.get("target_price"),
            "support_level": flm_pred.get("support_level"),
            "resistance_level": flm_pred.get("resistance_level"),
            "predictions_7d": predictions[:7],
            "sentiment_score": 0.0,  # prediction logic is FinanceLM-only
            "active_weights": {"financeLM": 1.0, "np_lstm": 0.0, "sentiment": 0.0},
        }

    def _generate_current_suggestion(self, combined_prediction: Dict[str, Any], market_state: Dict[str, Any]) -> Dict[str, Any]:
        trend = combined_prediction.get("trend", "sideways")
        confidence = _safe_float(combined_prediction.get("confidence"), 0.5)
        market_state_label = market_state.get("state_label", "不确定")

        if trend == "up":
            score = 0.50 + confidence * 0.38
        elif trend == "down":
            score = 0.50 - confidence * 0.38
        else:
            score = 0.50

        if score >= 0.64:
            action, action_code = "买入", "buy"
        elif score >= 0.56:
            action, action_code = "增持", "buy"
        elif score <= 0.36:
            action, action_code = "卖出", "sell"
        elif score <= 0.44:
            action, action_code = "减持", "sell"
        else:
            action, action_code = "持有观望", "hold"

        reason = f"FinanceLM趋势={trend}, 置信度={confidence:.1%}, 市场状态={market_state_label}"
        risk_level = "low" if confidence > 0.72 else "medium" if confidence > 0.52 else "high"
        risk_advice = {
            "low": "模型置信度较高，可按纪律执行仓位计划",
            "medium": "模型存在不确定性，建议分批和止损并行",
            "high": "置信度偏低，建议降低仓位并等待确认",
        }

        return {
            "action": action,
            "action_code": action_code,
            "action_detail": reason,
            "reason": reason,
            "confidence": confidence,
            "score": round(score, 4),
            "risk_level": risk_level,
            "risk_advice": risk_advice[risk_level],
        }

    def _generate_future_forecast(self, combined_prediction: Dict[str, Any], model_predictions: Dict[str, Any]) -> Dict[str, Any]:
        predictions = combined_prediction.get("predictions_7d", [])[:7]
        today = datetime.now().date()
        trading_days = self._get_trading_days(today, 7)

        predictions_by_day = []
        for i, price in enumerate(predictions):
            date = trading_days[i] if i < len(trading_days) else today + timedelta(days=i + 1)
            predictions_by_day.append(
                {
                    "date": date.isoformat(),
                    "day_of_week": date.strftime("%A"),
                    "predicted_price": round(price, 2),
                    "price_change": self._calculate_daily_change(predictions, i),
                    "confidence": self._calculate_daily_confidence(combined_prediction, i),
                }
            )

        trend_summary, trend_type, movement = self._summarize_movement(predictions)
        key_levels = self._extract_key_levels(combined_prediction, model_predictions)
        risk_assessment = self._assess_forecast_risk(combined_prediction, predictions)
        model_contributions = {
            "financeLM": {"contribution": 1.0, "status": "active", "confidence": combined_prediction.get("confidence", 0.5)},
            "np_lstm": {"contribution": 0.0, "status": "disabled"},
            "sentiment_analysis": {"contribution": 0.0, "status": "disabled"},
        }

        return {
            "predictions_by_day": predictions_by_day,
            "predictions_7d": predictions,
            "forecast_period": {
                "start_date": trading_days[0].isoformat() if trading_days else None,
                "end_date": trading_days[-1].isoformat() if len(trading_days) > 1 else None,
                "duration_days": len(predictions),
                "update_time": datetime.now().isoformat(),
            },
            "trend_summary": trend_summary,
            "trend_type": trend_type,
            "key_levels": key_levels,
            "expected_movement": movement,
            "weekly_summary": self._generate_weekly_summary(predictions, combined_prediction),
            "risk_assessment": risk_assessment,
            "model_contributions": model_contributions,
        }

    def _get_trading_days(self, start_date, days: int):
        trading_days = []
        current = start_date
        while len(trading_days) < days:
            if current.weekday() < 5:
                trading_days.append(current)
            current += timedelta(days=1)
        return trading_days

    def _calculate_daily_change(self, predictions: List[float], index: int) -> Dict[str, Any]:
        if index == 0 or index >= len(predictions):
            return {"percent": None, "direction": None}
        prev_val = predictions[index - 1]
        curr_val = predictions[index]
        if not prev_val:
            return {"percent": None, "direction": None}
        change = (curr_val - prev_val) / prev_val * 100
        return {
            "percent": round(change, 2),
            "direction": "up" if change > 0 else "down" if change < 0 else "flat",
            "value": round(curr_val - prev_val, 2),
        }

    def _calculate_daily_confidence(self, combined_prediction: Dict[str, Any], day_index: int) -> float:
        base = _safe_float(combined_prediction.get("confidence"), 0.5)
        decay = max(0.7, 1 - day_index * 0.03)
        return round(_clip(base * decay, 0.2, 0.95), 4)

    def _summarize_movement(self, predictions: List[float]) -> Tuple[str, str, Dict[str, Any]]:
        if len(predictions) < 2 or not predictions[0]:
            return "预测样本不足", "insufficient", {}
        first = predictions[0]
        last = predictions[-1]
        change_pct = (last - first) / first * 100
        volatility_pct = ((max(predictions) - min(predictions)) / first * 100) if first else 0.0
        trend_type = "bullish" if change_pct > 1 else "bearish" if change_pct < -1 else "sideways"
        if trend_type == "bullish":
            summary = f"预计未来7天偏强，区间涨幅约 {change_pct:.2f}%"
        elif trend_type == "bearish":
            summary = f"预计未来7天偏弱，区间跌幅约 {abs(change_pct):.2f}%"
        else:
            summary = f"预计未来7天震荡，波动约 {volatility_pct:.2f}%"
        movement = {
            "change_percent": round(change_pct, 2),
            "change_direction": "up" if change_pct > 0 else "down" if change_pct < 0 else "sideways",
            "volatility_percent": round(volatility_pct, 2),
            "volatility_level": "low" if volatility_pct < 3 else "medium" if volatility_pct < 6 else "high",
            "min_price": round(min(predictions), 2),
            "max_price": round(max(predictions), 2),
            "avg_price": round(sum(predictions) / len(predictions), 2),
            "range_high_low": round(((max(predictions) - min(predictions)) / min(predictions) * 100), 2) if min(predictions) else None,
        }
        return summary, trend_type, movement

    def _extract_key_levels(self, combined_prediction: Dict[str, Any], model_predictions: Dict[str, Any]) -> Dict[str, float]:
        levels: Dict[str, float] = {}
        if combined_prediction.get("support_level") is not None:
            levels["support_primary"] = round(_safe_float(combined_prediction.get("support_level")), 2)
        if combined_prediction.get("resistance_level") is not None:
            levels["resistance_primary"] = round(_safe_float(combined_prediction.get("resistance_level")), 2)
        preds = combined_prediction.get("predictions_7d") or []
        if preds:
            levels["intraday_support"] = round(min(preds) * 0.99, 2)
            levels["intraday_resistance"] = round(max(preds) * 1.01, 2)
        finance_pred = model_predictions.get("financeLM", {})
        if finance_pred.get("support_level") is not None:
            levels["support_secondary"] = round(_safe_float(finance_pred.get("support_level")), 2)
        if finance_pred.get("resistance_level") is not None:
            levels["resistance_secondary"] = round(_safe_float(finance_pred.get("resistance_level")), 2)
        return levels

    def _generate_weekly_summary(self, predictions: List[float], combined_prediction: Dict[str, Any]) -> Dict[str, Any]:
        if not predictions or not predictions[0]:
            return {}
        first = predictions[0]
        last = predictions[-1]
        change = (last - first) / first * 100
        confidence = _safe_float(combined_prediction.get("confidence"), 0.5)
        if change > 2 and confidence > 0.6:
            rec = "本周偏强，建议顺势但控制回撤"
        elif change < -2 and confidence > 0.6:
            rec = "本周偏弱，建议控制仓位并观察确认"
        else:
            rec = "本周偏震荡，建议区间策略"
        return {
            "weekly_return": round(change, 2),
            "expected_high": round(max(predictions), 2),
            "expected_low": round(min(predictions), 2),
            "target_price": combined_prediction.get("target_price"),
            "target_confidence": confidence,
            "recommendation": rec,
        }

    def _assess_forecast_risk(self, combined_prediction: Dict[str, Any], predictions: List[float]) -> Dict[str, Any]:
        confidence = _safe_float(combined_prediction.get("confidence"), 0.5)
        model_risk = 1 - confidence
        volatility_risk = 0.2
        if len(predictions) >= 2 and min(predictions) > 0:
            volatility = (max(predictions) - min(predictions)) / min(predictions) * 100
            volatility_risk = 0.4 if volatility > 8 else 0.2 if volatility > 4 else 0.1
        total_risk = (model_risk + volatility_risk + 0.2) / 3
        if total_risk < 0.3:
            risk_level, risk_color = "low", "green"
            advice = "风险较低，可按计划执行"
        elif total_risk < 0.5:
            risk_level, risk_color = "medium", "yellow"
            advice = "风险中等，建议分批并设置止损"
        else:
            risk_level, risk_color = "high", "red"
            advice = "风险较高，建议降低仓位或等待确认"
        return {
            "risk_score": round(total_risk, 2),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "risk_factors": {
                "model_confidence": round(model_risk, 4),
                "volatility_risk": round(volatility_risk, 4),
                "market_risk": 0.2,
            },
            "advice": advice,
            "position_sizing": self._calculate_position_sizing(total_risk),
        }

    def _calculate_position_sizing(self, risk_score: float) -> Dict[str, float]:
        base_position = 1 - (risk_score * 1.2)
        return {
            "recommended_max_position": min(0.9, max(0.2, round(base_position, 2))),
            "conservative_position": min(0.7, max(0.1, round(base_position * 0.7, 2))),
            "aggressive_position": min(1.0, max(0.3, round(base_position * 1.3, 2))),
        }


multi_model_coordinator = MultiModelCoordinator()


async def get_multi_model_prediction(stock_name: str, stock_data: Optional[Dict[str, Any]] = None, days: int = 60) -> Dict[str, Any]:
    return await multi_model_coordinator.analyze_and_predict(stock_name, stock_data, days)
