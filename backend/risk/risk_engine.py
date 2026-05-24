"""Risk engine with quantile-style VaR/CVaR scoring and portfolio controls."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import random
import statistics
from typing import Any, Dict, List, Optional, Tuple

from .event_risk_engine import event_risk_engine
from .market_state_engine import market_state_engine_2
from .pretrained_signal_model import pretrained_headline_risk_scorer


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalized_entropy(probabilities: List[float]) -> float:
    clean = [p for p in probabilities if p > 0]
    if not clean:
        return 0.0
    entropy = -sum(p * math.log(p) for p in clean)
    max_entropy = math.log(len(probabilities)) if probabilities else 1.0
    return _clip(entropy / max_entropy) if max_entropy > 0 else 0.0


def _safe_mean(values: List[float], default: float = 0.0) -> float:
    return statistics.mean(values) if values else default


def _safe_pstdev(values: List[float], default: float = 0.0) -> float:
    return statistics.pstdev(values) if len(values) > 1 else default


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    q = _clip(q, 0.0, 1.0)
    idx = (len(sorted_vals) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = idx - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _normalize_weights(raw: List[float]) -> List[float]:
    if not raw:
        return []
    gross = sum(abs(v) for v in raw)
    if gross <= 1e-12:
        n = len(raw)
        return [1.0 / n for _ in raw]
    return [v / gross for v in raw]


@dataclass
class Alert:
    code: str
    severity: str
    message: str
    metric: str
    value: float
    threshold: float
    stage: str
    escalation_action: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "metric": self.metric,
            "value": round(self.value, 6),
            "threshold": round(self.threshold, 6),
            "stage": self.stage,
            "escalation_action": self.escalation_action,
        }


class RiskControlEngine:
    """Risk engine with single-asset and portfolio-level controls."""

    PROFILE_CONFIG: Dict[str, Dict[str, Any]] = {
        "aggressive": {
            "weights": {
                "sentiment": 0.20,
                "trend": 0.17,
                "volatility": 0.20,
                "event": 0.16,
                "uncertainty": 0.10,
                "market_regime": 0.09,
                "reasoning": 0.08,
            },
            "thresholds": (42.0, 62.0, 80.0),
            "var_alpha": 0.90,
            "cvar_alpha": 0.90,
            "var_cap": 0.050,
            "cvar_cap": 0.085,
        },
        "moderate": {
            "weights": {
                "sentiment": 0.22,
                "trend": 0.16,
                "volatility": 0.18,
                "event": 0.16,
                "uncertainty": 0.10,
                "market_regime": 0.10,
                "reasoning": 0.08,
            },
            "thresholds": (36.0, 56.0, 74.0),
            "var_alpha": 0.95,
            "cvar_alpha": 0.95,
            "var_cap": 0.040,
            "cvar_cap": 0.070,
        },
        "conservative": {
            "weights": {
                "sentiment": 0.24,
                "trend": 0.15,
                "volatility": 0.17,
                "event": 0.17,
                "uncertainty": 0.09,
                "market_regime": 0.10,
                "reasoning": 0.08,
            },
            "thresholds": (30.0, 50.0, 68.0),
            "var_alpha": 0.975,
            "cvar_alpha": 0.975,
            "var_cap": 0.032,
            "cvar_cap": 0.058,
        },
    }

    DIRECTION_RISK = {
        "down": 0.95,
        "bear": 0.90,
        "bearish": 0.90,
        "up": 0.18,
        "bull": 0.20,
        "bullish": 0.20,
        "stable": 0.45,
        "sideways": 0.52,
        "neutral": 0.50,
        "uncertain": 0.65,
    }

    MARKET_STATE_RISK = {
        "bear": 0.86,
        "volatile": 0.90,
        "uncertain": 0.66,
        "sideways": 0.54,
        "quiet": 0.30,
        "bull": 0.22,
    }

    SEVERITY_TO_STAGE = {
        "low": "L1",
        "medium": "L2",
        "high": "L3",
        "critical": "L4",
    }

    STAGE_ESCALATION = {
        "L1": "Log and monitor",
        "L2": "Escalate to strategy owner within same cycle",
        "L3": "Freeze new risk and notify risk manager",
        "L4": "Immediate kill-switch and human override required",
    }

    def __init__(self) -> None:
        self.market_state_engine = market_state_engine_2
        self._last_single_regime = "sideways"
        self._last_portfolio_regime = "sideways"

    def assess(
        self,
        stock_name: str,
        risk_profile: str,
        news_items: List[Dict[str, Any]],
        inference_result: Optional[Dict[str, Any]] = None,
        multi_model_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        profile = self._normalize_profile(risk_profile)
        config = self.PROFILE_CONFIG[profile]
        base_weights = config["weights"].copy()

        inference_result = inference_result or {}
        multi_model_result = multi_model_result or {}

        sentiment = self._score_sentiment(news_items)
        trend = self._score_trend(inference_result, multi_model_result)
        volatility = self._score_volatility(multi_model_result)
        event = self._score_event_impact(news_items, inference_result)
        reasoning = self._score_reasoning(inference_result)
        uncertainty = self._score_uncertainty(inference_result, multi_model_result)
        data_quality = self._score_data_quality(news_items, inference_result, multi_model_result)

        prelim_factors = {
            "sentiment": sentiment,
            "trend": trend,
            "volatility": volatility,
            "event": event,
            "reasoning": reasoning,
            "uncertainty": uncertainty,
        }

        scenario_returns = self._build_return_scenarios(multi_model_result, prelim_factors)
        macro_factors = self.market_state_engine.extract_macro_factors(inference_result, multi_model_result)
        regime_info = self.market_state_engine.infer_regime(
            scenario_returns,
            macro_factors,
            previous_regime=self._last_single_regime,
        )
        self._last_single_regime = regime_info.get("regime", self._last_single_regime)

        adjusted_weights_pack = self.market_state_engine.adjusted_factor_weights(base_weights, regime_info)
        weights = adjusted_weights_pack.get("weights", base_weights)
        market_regime = self._score_market_regime(multi_model_result, regime_info)

        factors = {
            "sentiment": sentiment,
            "trend": trend,
            "volatility": volatility,
            "event": event,
            "reasoning": reasoning,
            "uncertainty": uncertainty,
            "market_regime": market_regime,
        }

        quantile_risk = self._compute_quantile_risk(scenario_returns, profile)
        dynamic_thresholds = self._calibrate_dynamic_thresholds(
            config["thresholds"],
            factors,
            data_quality,
            quantile_risk,
            regime_info=regime_info,
        )
        risk_score = self._compose_quantile_risk_score(
            quantile_risk=quantile_risk,
            factors=factors,
            weights=weights,
            data_quality=data_quality,
            profile=profile,
            regime_info=regime_info,
        )
        risk_level = self._map_risk_level(risk_score, dynamic_thresholds["thresholds"])
        alerts = self._build_layered_alerts(
            factors=factors,
            data_quality=data_quality,
            quantile_risk=quantile_risk,
            risk_score=risk_score,
            risk_level=risk_level,
            dynamic_thresholds=dynamic_thresholds["thresholds"],
        )
        escalation = self._build_escalation_strategy(alerts, risk_level, risk_score)
        controls = self._build_controls(
            level=risk_level,
            profile=profile,
            risk_score=risk_score,
            factors=factors,
            alerts=alerts,
            quantile_risk=quantile_risk,
        )

        weighted_contrib = {
            name: factors[name]["score"] * weights[name] for name in factors
        }
        factor_breakdown: Dict[str, Any] = {}
        for name, payload in factors.items():
            factor_breakdown[name] = {
                "score": round(payload["score"] * 100.0, 2),
                "weight": round(weights[name], 4),
                "contribution": round(weighted_contrib[name] * 100.0, 2),
                "signals": payload.get("signals", {}),
            }
        top_risk_drivers = [
            key
            for key, _ in sorted(
                factor_breakdown.items(),
                key=lambda item: item[1]["contribution"],
                reverse=True,
            )[:3]
        ]

        return {
            "stock_name": stock_name,
            "risk_profile": profile,
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "risk_label": self._risk_label(risk_level),
            "risk_method": "quantile_var_cvar_with_event_chain_hmm_and_reasoning_fusion",
            "market_state_engine": {
                "version": "2.0",
                "regime_info": regime_info,
                "effective_weights": weights,
                "weight_adjustment_meta": adjusted_weights_pack,
            },
            "quantile_risk": quantile_risk,
            "dynamic_thresholds": dynamic_thresholds,
            "factor_breakdown": factor_breakdown,
            "top_risk_drivers": top_risk_drivers,
            "alerts": [alert.to_dict() for alert in alerts],
            "escalation_strategy": escalation,
            "control_actions": controls,
            "data_quality": data_quality,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def assess_portfolio(
        self,
        assets: List[Dict[str, Any]],
        risk_profile: str = "moderate",
        covariance_matrix: Optional[List[List[float]]] = None,
        gross_leverage: Optional[float] = None,
        net_leverage: Optional[float] = None,
    ) -> Dict[str, Any]:
        profile = self._normalize_profile(risk_profile)
        config = self.PROFILE_CONFIG[profile]

        if not assets:
            return self._build_empty_portfolio_assessment(profile)

        normalized_assets = self._normalize_portfolio_assets(assets)
        weights = [item["weight"] for item in normalized_assets]
        names = [item["stock_name"] for item in normalized_assets]

        return_series = [self._asset_return_series(item) for item in normalized_assets]
        aligned_series = self._align_return_series(return_series)
        cov = self._resolve_covariance_matrix(aligned_series, covariance_matrix)
        portfolio_returns = self._portfolio_return_series(weights, aligned_series)
        if len(portfolio_returns) < 7:
            synthetic = self._portfolio_synthetic_tail(portfolio_returns, cov, weights)
            portfolio_returns.extend(synthetic)

        quantile_risk = self._compute_quantile_risk(portfolio_returns, profile)

        macro_factors = self._portfolio_macro_factors(normalized_assets)
        regime_info = self.market_state_engine.infer_regime(
            portfolio_returns,
            macro_factors,
            previous_regime=self._last_portfolio_regime,
        )
        self._last_portfolio_regime = regime_info.get("regime", self._last_portfolio_regime)

        concentration = self._portfolio_concentration(weights)
        crowding = self._portfolio_crowding(normalized_assets)
        industry_exposure = self._portfolio_bucket_exposure(normalized_assets, "sector")
        style_exposure = self._portfolio_bucket_exposure(normalized_assets, "style")
        leverage = self._portfolio_leverage(normalized_assets, gross_leverage, net_leverage)
        covariance_metrics = self._portfolio_covariance_metrics(cov, weights)

        exposure_scores = {
            "concentration": concentration["score"],
            "crowding": crowding["score"],
            "industry_exposure": industry_exposure["score"],
            "style_exposure": style_exposure["score"],
            "leverage_exposure": leverage["score"],
            "covariance_cluster": covariance_metrics["score"],
        }
        exposure_composite = _clip(
            0.24 * exposure_scores["concentration"] +
            0.16 * exposure_scores["crowding"] +
            0.16 * exposure_scores["industry_exposure"] +
            0.14 * exposure_scores["style_exposure"] +
            0.20 * exposure_scores["leverage_exposure"] +
            0.10 * exposure_scores["covariance_cluster"]
        )

        quantile_norm = _clip(
            0.55 * quantile_risk["var_norm"] + 0.45 * quantile_risk["cvar_norm"]
        )
        regime_stress = _clip(_to_float(regime_info.get("state_stress"), 0.5))
        risk_score = _clip(
            0.52 * quantile_norm + 0.34 * exposure_composite + 0.14 * regime_stress
        ) * 100.0

        pseudo_factors = {
            "volatility": {"score": _clip(quantile_norm)},
            "uncertainty": {"score": _clip(covariance_metrics["score"])},
            "market_regime": {"score": regime_stress},
        }
        pseudo_quality = {"quality_score": 1.0}
        dynamic_thresholds = self._calibrate_dynamic_thresholds(
            config["thresholds"],
            pseudo_factors,
            pseudo_quality,
            quantile_risk,
            regime_info=regime_info,
        )
        risk_level = self._map_risk_level(risk_score, dynamic_thresholds["thresholds"])

        alerts = self._build_portfolio_alerts(
            risk_score=risk_score,
            risk_level=risk_level,
            quantile_risk=quantile_risk,
            concentration=concentration,
            crowding=crowding,
            industry_exposure=industry_exposure,
            style_exposure=style_exposure,
            leverage=leverage,
            covariance_metrics=covariance_metrics,
            thresholds=dynamic_thresholds["thresholds"],
        )
        escalation = self._build_escalation_strategy(alerts, risk_level, risk_score)

        return {
            "portfolio_size": len(normalized_assets),
            "asset_names": names,
            "risk_profile": profile,
            "risk_method": "portfolio_quantile_var_cvar_with_hmm_regime_and_exposure_controls",
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "risk_label": self._risk_label(risk_level),
            "market_state_engine": {
                "version": "2.0",
                "regime_info": regime_info,
            },
            "quantile_risk": quantile_risk,
            "dynamic_thresholds": dynamic_thresholds,
            "portfolio_metrics": {
                "expected_return_1d": round(_safe_mean(portfolio_returns), 6),
                "volatility_1d": round(math.sqrt(max(covariance_metrics["portfolio_variance"], 0.0)), 6),
                "covariance_matrix": cov,
            },
            "exposure_breakdown": {
                "concentration": concentration,
                "crowding": crowding,
                "industry_exposure": industry_exposure,
                "style_exposure": style_exposure,
                "leverage_exposure": leverage,
                "covariance_cluster": covariance_metrics,
                "composite_exposure_score": round(exposure_composite, 4),
            },
            "alerts": [alert.to_dict() for alert in alerts],
            "escalation_strategy": escalation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def build_empty_assessment(self, stock_name: str, risk_profile: str) -> Dict[str, Any]:
        profile = self._normalize_profile(risk_profile)
        return {
            "stock_name": stock_name,
            "risk_profile": profile,
            "risk_score": 50.0,
            "risk_level": "high",
            "risk_label": self._risk_label("high"),
            "risk_method": "quantile_var_cvar_with_event_chain_hmm_and_reasoning_fusion",
            "market_state_engine": {
                "version": "2.0",
                "regime_info": {
                    "regime": "unknown",
                    "regime_label": "Unknown",
                    "confidence": 0.0,
                },
            },
            "quantile_risk": {
                "var_alpha": self.PROFILE_CONFIG[profile]["var_alpha"],
                "cvar_alpha": self.PROFILE_CONFIG[profile]["cvar_alpha"],
                "var_1d": 0.03,
                "cvar_1d": 0.05,
                "tail_loss_mean": 0.05,
                "tail_loss_std": 0.0,
                "downside_prob": 1.0,
                "sample_size": 0,
                "var_norm": 0.75,
                "cvar_norm": 0.75,
            },
            "factor_breakdown": {},
            "top_risk_drivers": ["data_unavailable"],
            "alerts": [
                Alert(
                    code="DATA_EMPTY",
                    severity="high",
                    message="No data available; risk assumed elevated.",
                    metric="news_count",
                    value=0.0,
                    threshold=1.0,
                    stage="L3",
                    escalation_action=self.STAGE_ESCALATION["L3"],
                ).to_dict()
            ],
            "escalation_strategy": {
                "current_stage": "L3",
                "next_action": self.STAGE_ESCALATION["L3"],
                "rule": "data_unavailable_fallback",
            },
            "control_actions": {
                "position_limit": 0.2,
                "action": "reduce_exposure",
                "risk_budget_hint": 0.25,
                "next_steps": [
                    "Re-fetch data sources and re-run risk assessment.",
                    "Suspend new positions until confidence improves.",
                ],
            },
            "data_quality": {
                "quality_score": 0.0,
                "news_count": 0,
                "source_count": 0,
                "inference_ready": False,
                "multi_model_ready": False,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _build_empty_portfolio_assessment(self, profile: str) -> Dict[str, Any]:
        return {
            "portfolio_size": 0,
            "asset_names": [],
            "risk_profile": profile,
            "risk_method": "portfolio_quantile_var_cvar_with_hmm_regime_and_exposure_controls",
            "market_state_engine": {
                "version": "2.0",
                "regime_info": {
                    "regime": "unknown",
                    "regime_label": "Unknown",
                    "confidence": 0.0,
                },
            },
            "risk_score": 55.0,
            "risk_level": "high",
            "risk_label": self._risk_label("high"),
            "alerts": [
                Alert(
                    code="PORTFOLIO_EMPTY",
                    severity="high",
                    message="No assets provided for portfolio risk.",
                    metric="asset_count",
                    value=0.0,
                    threshold=1.0,
                    stage="L3",
                    escalation_action=self.STAGE_ESCALATION["L3"],
                ).to_dict()
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _normalize_profile(self, risk_profile: str) -> str:
        profile = (risk_profile or "moderate").lower().strip()
        return profile if profile in self.PROFILE_CONFIG else "moderate"

    def _map_risk_level(self, score: float, thresholds: Tuple[float, float, float]) -> str:
        low_cap, medium_cap, high_cap = thresholds
        if score < low_cap:
            return "low"
        if score < medium_cap:
            return "medium"
        if score < high_cap:
            return "high"
        return "critical"

    def _risk_label(self, level: str) -> str:
        labels = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH", "critical": "CRITICAL"}
        return labels.get(level, "UNKNOWN")

    def _score_sentiment(self, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(news_items)
        if total == 0:
            return {
                "score": 0.55,
                "signals": {
                    "news_count": 0,
                    "negative_ratio": 0.0,
                    "weighted_negative_ratio": 0.0,
                    "entropy": 1.0,
                },
            }

        weighted_total = sum(max(_to_float(item.get("weight"), 0.5), 0.01) for item in news_items)
        positive = [item for item in news_items if item.get("sentiment_type") == "positive"]
        negative = [item for item in news_items if item.get("sentiment_type") == "negative"]
        neutral_count = total - len(positive) - len(negative)

        neg_weight = sum(max(_to_float(item.get("weight"), 0.5), 0.01) for item in negative)
        weighted_negative_ratio = neg_weight / weighted_total if weighted_total > 0 else 0.0
        negative_ratio = len(negative) / total
        probs = [len(positive) / total, len(negative) / total, neutral_count / total]
        entropy = _normalized_entropy(probs)

        # 按数据源类型做平衡，避免某一类来源（如search_signal）过度主导情绪分值。
        source_type_bucket: Dict[str, Dict[str, float]] = {}
        source_type_counter: Dict[str, int] = {}
        for item in news_items:
            source_type = str(item.get("data_source_type", "news") or "news")
            source_type_counter[source_type] = source_type_counter.get(source_type, 0) + 1
            bucket = source_type_bucket.setdefault(source_type, {"total_w": 0.0, "neg_w": 0.0})
            w = max(_to_float(item.get("weight"), 0.5), 0.01)
            bucket["total_w"] += w
            if str(item.get("sentiment_type", "")).lower() == "negative":
                bucket["neg_w"] += w

        source_type_negative_ratios: List[float] = []
        for val in source_type_bucket.values():
            total_w = _to_float(val.get("total_w"), 0.0)
            neg_w = _to_float(val.get("neg_w"), 0.0)
            source_type_negative_ratios.append(neg_w / total_w if total_w > 0 else 0.0)
        balanced_negative_ratio = _safe_mean(source_type_negative_ratios, weighted_negative_ratio)
        source_type_diversity = _clip(len(source_type_bucket) / 4.0)
        source_type_dominance = max(source_type_counter.values()) / total if source_type_counter else 1.0
        source_dominance_penalty = _clip((source_type_dominance - 0.60) / 0.35)
        search_signal_share = (
            source_type_counter.get("search_signal", 0) / total if total > 0 else 0.0
        )

        blended_negative_ratio = _clip(
            0.72 * weighted_negative_ratio +
            0.28 * balanced_negative_ratio
        )

        semantic = pretrained_headline_risk_scorer.score(news_items)
        semantic_risk = _to_float(semantic.get("semantic_risk"), 0.5)
        semantic_consensus = _clip(_to_float(semantic.get("semantic_consensus"), 0.0))
        semantic_dispersion = _clip(_to_float(semantic.get("risk_dispersion"), 0.0) / 0.30)
        semantic_weight = (0.16 + 0.18 * semantic_consensus) if semantic.get("available") else 0.0
        base_score = _clip(
            0.62 * blended_negative_ratio +
            0.20 * entropy +
            0.10 * negative_ratio +
            0.05 * source_dominance_penalty +
            0.03 * _clip(1.0 - source_type_diversity)
        )
        score = _clip(
            base_score * (1.0 - semantic_weight) +
            semantic_risk * semantic_weight +
            0.03 * semantic_dispersion
        )
        return {
            "score": score,
            "signals": {
                "news_count": total,
                "negative_ratio": round(negative_ratio, 4),
                "weighted_negative_ratio": round(weighted_negative_ratio, 4),
                "balanced_negative_ratio": round(balanced_negative_ratio, 4),
                "blended_negative_ratio": round(blended_negative_ratio, 4),
                "entropy": round(entropy, 4),
                "source_type_count": len(source_type_bucket),
                "source_type_diversity": round(source_type_diversity, 4),
                "source_type_dominance": round(source_type_dominance, 4),
                "source_dominance_penalty": round(source_dominance_penalty, 4),
                "search_signal_share": round(search_signal_share, 4),
                "semantic_model_available": bool(semantic.get("available")),
                "semantic_model_alias": semantic.get("model_alias"),
                "semantic_model_count": int(semantic.get("model_count", 0)),
                "semantic_model_aliases": semantic.get("model_aliases", []),
                "semantic_risk": round(semantic_risk, 4),
                "semantic_consensus": round(semantic_consensus, 4),
                "semantic_dispersion": round(_to_float(semantic.get("risk_dispersion"), 0.0), 4),
                "semantic_sample_count": semantic.get("sample_count", 0),
                "semantic_error": semantic.get("error"),
            },
        }

    def _score_trend(self, inference_result: Dict[str, Any], multi_model_result: Dict[str, Any]) -> Dict[str, Any]:
        trend_predictions = inference_result.get("trend_predictions", {})
        horizon_weights = {"1h": 0.15, "1d": 0.50, "1w": 0.35}
        trend_score = 0.0
        signals: Dict[str, Any] = {}
        for horizon, weight in horizon_weights.items():
            payload = trend_predictions.get(horizon, {})
            direction = str(payload.get("direction", "uncertain")).lower()
            confidence = _clip(_to_float(payload.get("confidence"), 0.5))
            direction_risk = self.DIRECTION_RISK.get(direction, 0.65)
            horizon_risk = direction_risk * confidence + (1.0 - confidence) * 0.50
            trend_score += horizon_risk * weight
            signals[f"{horizon}_direction"] = direction
            signals[f"{horizon}_confidence"] = round(confidence, 4)

        action = str(multi_model_result.get("current_suggestion", {}).get("action", "")).lower()
        action_map = {"buy": -0.08, "hold": 0.0, "sell": 0.08}
        trend_score = _clip(trend_score + action_map.get(action, 0.0))
        signals["current_action"] = action or "unknown"
        return {"score": trend_score, "signals": signals}

    def _score_volatility(self, multi_model_result: Dict[str, Any]) -> Dict[str, Any]:
        forecast = multi_model_result.get("future_forecast", {})
        predictions = forecast.get("predictions_7d", [])
        values = [_to_float(v, 0.0) for v in predictions if _to_float(v, 0.0) > 0]
        if len(values) >= 2:
            mean_price = _safe_mean(values, 1.0)
            std_price = _safe_pstdev(values, 0.0)
            range_pct = (max(values) - min(values)) / mean_price if mean_price > 0 else 0.0
            std_pct = std_price / mean_price if mean_price > 0 else 0.0
        else:
            movement = forecast.get("expected_movement", {})
            range_pct = abs(_to_float(movement.get("change_percent"), 0.0)) / 100.0
            std_pct = _to_float(movement.get("volatility_percent"), 0.0) / 100.0

        # fallback: 当模型预测不可用时，使用真实行情波动率与收益率回撤估算
        source = "model"
        if std_pct == 0 and range_pct == 0:
            market_data = multi_model_result.get("market_data", {})
            equity_vol = _to_float(market_data.get("equity_volatility"), 0.0)
            stock_returns = market_data.get("stock_returns", [])
            if equity_vol > 0 and len(stock_returns) >= 10:
                std_pct = equity_vol
                cum = 1.0
                peak = 1.0
                max_dd = 0.0
                for r in stock_returns:
                    cum *= (1.0 + _to_float(r, 0.0))
                    peak = max(peak, cum)
                    dd = (peak - cum) / peak if peak > 0 else 0.0
                    max_dd = max(max_dd, dd)
                range_pct = max_dd if max_dd > 0 else equity_vol * 2.0
                source = "real_market_data"

        score = _clip(0.55 * _clip(range_pct / 0.08) + 0.45 * _clip(std_pct / 0.03))
        return {
            "score": score,
            "signals": {
                "prediction_count": len(values),
                "range_pct": round(range_pct, 4),
                "std_pct": round(std_pct, 4),
                "source": source,
            },
        }

    def _score_event_impact(
        self,
        news_items: List[Dict[str, Any]],
        inference_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        inference_events = inference_result.get("event_impacts", [])
        event_pack = event_risk_engine.assess(news_items, inference_events)
        score = _clip(_to_float(event_pack.get("aggregate_risk"), 0.45))
        propagation = event_pack.get("propagation_scores", {}) or {}
        return {
            "score": score,
            "signals": {
                "event_engine_version": "1.0",
                "event_count": int(event_pack.get("event_count", 0)),
                "event_type_distribution": event_pack.get("event_type_distribution", {}),
                "event_type_scores": event_pack.get("event_type_scores", {}),
                "dominant_event_type": event_pack.get("dominant_event_type", "none"),
                "severity_avg": round(_to_float(event_pack.get("severity_avg"), 0.0), 6),
                "timeliness_avg": round(_to_float(event_pack.get("timeliness_avg"), 0.0), 6),
                "reversibility_avg": round(_to_float(event_pack.get("reversibility_avg"), 0.0), 6),
                "irreversibility_avg": round(_to_float(event_pack.get("irreversibility_avg"), 0.0), 6),
                "propagation_company": round(_to_float(propagation.get("company"), 0.0), 6),
                "propagation_sector": round(_to_float(propagation.get("sector"), 0.0), 6),
                "propagation_index": round(_to_float(propagation.get("index"), 0.0), 6),
                "chain_composite_avg": round(_to_float(event_pack.get("chain_composite_avg"), 0.0), 6),
                "adverse_event_flow": round(_to_float(event_pack.get("adverse_event_flow"), 0.0), 6),
                "positive_event_flow": round(_to_float(event_pack.get("positive_event_flow"), 0.0), 6),
                "net_event_flow": round(_to_float(event_pack.get("net_event_flow"), 0.0), 6),
                "adverse_event_ratio": round(_to_float(event_pack.get("adverse_event_ratio"), 0.0), 6),
                "tail_event_flag": bool(event_pack.get("tail_event_flag", False)),
                "negative_event_density": round(_to_float(event_pack.get("adverse_event_ratio"), 0.0), 6),
                "top_events": event_pack.get("top_events", []),
            },
        }

    def _score_reasoning(self, inference_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        推理质量风险分：
        - NLI矛盾占比
        - 多周期趋势分歧
        - 事件情景尾部风险
        - 推理链路健康度（health）
        """
        reasoning_meta = inference_result.get("reasoning_meta", {}) or {}
        summary = inference_result.get("summary", {}) or {}

        contradiction_ratio = _clip(_to_float(reasoning_meta.get("nli_contradiction_ratio"), 0.0))
        trend_divergence = _clip(_to_float(reasoning_meta.get("trend_divergence"), 0.0))
        tail_risk = _clip(_to_float(reasoning_meta.get("scenario_tail_risk"), 0.0))
        reasoning_health = _clip(_to_float(reasoning_meta.get("reasoning_health"), 0.5))
        nli_confidence = _clip(_to_float(reasoning_meta.get("nli_confidence"), 0.5))

        # 回退：如果没有reasoning_meta，直接从原字段估计。
        if not reasoning_meta:
            inferences = inference_result.get("inference_results", []) or []
            n_total = len(inferences)
            if n_total > 0:
                contradiction = sum(
                    1 for row in inferences
                    if str(row.get("inference_type", "")).lower() == "contradiction"
                )
                contradiction_ratio = _clip(contradiction / n_total)
                nli_confidence = _clip(_safe_mean([_to_float(r.get("confidence"), 0.5) for r in inferences], 0.5))
            trend_predictions = inference_result.get("trend_predictions", {}) or {}
            dirs = []
            for h in ("1h", "1d", "1w"):
                d = str((trend_predictions.get(h, {}) or {}).get("direction", "uncertain")).lower()
                if d in {"up", "down", "stable"}:
                    dirs.append(d)
            if dirs:
                up = sum(1 for d in dirs if d == "up")
                down = sum(1 for d in dirs if d == "down")
                stable = sum(1 for d in dirs if d == "stable")
                trend_divergence = _clip(1.0 - max(up, down, stable) / len(dirs))
            tail_risk = _clip(_to_float(summary.get("scenario_tail_risk"), 0.0))
            reasoning_health = _clip(0.60 - 0.45 * contradiction_ratio - 0.25 * trend_divergence)

        key_events = _clip(_to_float(summary.get("key_events_count"), 0.0) / 8.0)
        # 越高越风险。
        score = _clip(
            0.31 * contradiction_ratio +
            0.27 * trend_divergence +
            0.23 * tail_risk +
            0.13 * (1.0 - reasoning_health) +
            0.06 * (1.0 - nli_confidence) +
            0.04 * key_events
        )
        return {
            "score": score,
            "signals": {
                "nli_contradiction_ratio": round(contradiction_ratio, 6),
                "trend_divergence": round(trend_divergence, 6),
                "scenario_tail_risk": round(tail_risk, 6),
                "reasoning_health": round(reasoning_health, 6),
                "nli_confidence": round(nli_confidence, 6),
                "reasoning_meta_available": bool(reasoning_meta),
            },
        }

    def _score_uncertainty(self, inference_result: Dict[str, Any], multi_model_result: Dict[str, Any]) -> Dict[str, Any]:
        summary_conf = _clip(_to_float(inference_result.get("summary", {}).get("confidence"), 0.5))
        ensemble_conf = _clip(_to_float(multi_model_result.get("confidence"), 0.5))
        confidence = _clip(0.45 * summary_conf + 0.55 * ensemble_conf)
        score = _clip(1.0 - confidence)
        return {
            "score": score,
            "signals": {"combined_confidence": round(confidence, 4)},
        }

    def _score_market_regime(
        self,
        multi_model_result: Dict[str, Any],
        regime_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        market_state = multi_model_result.get("market_state", {})
        state = str(market_state.get("state", "")).lower()
        confidence = _clip(_to_float(market_state.get("confidence"), 0.5))
        base = self.MARKET_STATE_RISK.get(state, 0.58)
        hmm_score = self.market_state_engine.regime_risk_score(regime_info or {}, fallback=base)
        hmm_conf = _clip(_to_float((regime_info or {}).get("confidence"), 0.5))
        blend = _clip(0.35 + 0.45 * hmm_conf)
        score = _clip((1.0 - blend) * (base * confidence + (1.0 - confidence) * 0.55) + blend * hmm_score)
        return {
            "score": score,
            "signals": {
                "market_state": state or "unknown",
                "market_confidence": round(confidence, 4),
                "hmm_regime": (regime_info or {}).get("regime", "unknown"),
                "hmm_confidence": round(hmm_conf, 4),
                "blend": round(blend, 4),
            },
        }

    def _score_data_quality(
        self,
        news_items: List[Dict[str, Any]],
        inference_result: Dict[str, Any],
        multi_model_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        news_count = len(news_items)
        source_count = len({item.get("source", "unknown") for item in news_items if item.get("source")})
        source_type_count = len({
            str(item.get("data_source_type", "news") or "news")
            for item in news_items
        })
        signal_type_count = len({
            str(item.get("signal_type", "")).strip()
            for item in news_items
            if str(item.get("signal_type", "")).strip()
        })
        search_signal_count = sum(
            1 for item in news_items
            if str(item.get("data_source_type", "")).strip() == "search_signal"
        )
        official_count = sum(
            1 for item in news_items
            if str(item.get("data_source_type", "")).strip() == "official_announcement"
        )

        inference_ready = bool(inference_result)
        multi_model_ready = bool(multi_model_result)
        reasoning_meta_available = bool(inference_result.get("reasoning_meta"))
        source_type_diversity = _clip(source_type_count / 4.0)
        signal_type_diversity = _clip(signal_type_count / 3.0)
        coverage_score = _clip(0.60 * source_type_diversity + 0.40 * signal_type_diversity)
        search_signal_share = search_signal_count / max(news_count, 1)
        official_share = official_count / max(news_count, 1)
        search_dominance_penalty = _clip((search_signal_share - 0.65) / 0.30)
        official_bonus = 0.12 * _clip(official_share / 0.20)

        quality_score = _clip(
            0.30 * _clip(news_count / 30.0) +
            0.12 * _clip(source_count / 6.0) +
            0.14 * coverage_score +
            0.16 * (1.0 if inference_ready else 0.0) +
            0.16 * (1.0 if multi_model_ready else 0.0) +
            0.08 * (1.0 if reasoning_meta_available else 0.0) +
            official_bonus -
            0.06 * search_dominance_penalty
        )
        return {
            "quality_score": round(quality_score, 4),
            "news_count": news_count,
            "source_count": source_count,
            "source_type_count": source_type_count,
            "signal_type_count": signal_type_count,
            "coverage_score": round(coverage_score, 4),
            "search_signal_share": round(search_signal_share, 4),
            "official_announcement_share": round(official_share, 4),
            "search_dominance_penalty": round(search_dominance_penalty, 4),
            "inference_ready": inference_ready,
            "multi_model_ready": multi_model_ready,
            "reasoning_meta_available": reasoning_meta_available,
        }

    def _build_return_scenarios(
        self,
        multi_model_result: Dict[str, Any],
        factors: Dict[str, Dict[str, Any]],
    ) -> List[float]:
        # Priority 1: 真实收益率（风控链路注入）
        real_returns = multi_model_result.get("real_returns", [])
        if real_returns and len(real_returns) >= 10:
            clean = [_to_float(r, float("nan")) for r in real_returns]
            clean = [r for r in clean if math.isfinite(r)]
            if len(clean) >= 10:
                return clean

        # Priority 2: 模型预测收益率
        returns: List[float] = []
        forecast = multi_model_result.get("future_forecast", {})
        predictions = [p for p in forecast.get("predictions_7d", []) if _to_float(p, 0.0) > 0]
        for i in range(1, len(predictions)):
            prev_p = _to_float(predictions[i - 1], 0.0)
            curr_p = _to_float(predictions[i], 0.0)
            if prev_p > 0:
                returns.append((curr_p - prev_p) / prev_p)

        # Priority 3: 参数化合成场景
        movement = forecast.get("expected_movement", {})
        change_pct = _to_float(movement.get("change_percent"), 0.0) / 100.0
        vol_pct = _to_float(movement.get("volatility_percent"), 0.0) / 100.0

        if len(returns) < 5:
            daily_mu = change_pct / 7.0
            daily_sigma = max(vol_pct / math.sqrt(7.0), factors["volatility"]["score"] * 0.02)
            z_grid = [-2.2, -1.8, -1.4, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.4, 1.8, 2.2]
            downside_bias = (
                0.35 * factors["sentiment"]["score"] +
                0.35 * factors["trend"]["score"] +
                0.30 * factors["event"]["score"]
            )
            shift = -0.006 * downside_bias
            returns.extend([daily_mu + shift + daily_sigma * z for z in z_grid])

        return returns

    def _compute_quantile_risk(self, returns: List[float], profile: str) -> Dict[str, Any]:
        cfg = self.PROFILE_CONFIG[profile]
        var_alpha = cfg["var_alpha"]
        cvar_alpha = cfg["cvar_alpha"]
        losses = [-r for r in returns] if returns else [0.0]

        # Method 1: historical quantile
        hist_var = max(0.0, _quantile(losses, var_alpha))
        hist_cvar_threshold = max(0.0, _quantile(losses, cvar_alpha))
        hist_tail = [x for x in losses if x >= hist_cvar_threshold]
        hist_cvar = max(hist_var, _safe_mean(hist_tail, hist_var))

        # Method 2: parametric (t-style tail fit)
        param_var, param_cvar = self._parametric_var_cvar(losses, var_alpha, cvar_alpha)

        # Method 3: Monte Carlo
        mc_var, mc_cvar = self._monte_carlo_var_cvar(losses, var_alpha, cvar_alpha, n_sims=5000)

        # Readme-aligned fusion weights: 40% historical + 30% parametric + 30% Monte Carlo
        var_val = 0.40 * hist_var + 0.30 * param_var + 0.30 * mc_var
        cvar_val = 0.40 * hist_cvar + 0.30 * param_cvar + 0.30 * mc_cvar

        tail_std = _safe_pstdev(hist_tail, 0.0)
        downside_prob = len([r for r in returns if r < 0]) / max(len(returns), 1)
        var_norm = _clip(var_val / max(cfg["var_cap"], 1e-8))
        cvar_norm = _clip(cvar_val / max(cfg["cvar_cap"], 1e-8))
        return {
            "var_alpha": var_alpha,
            "cvar_alpha": cvar_alpha,
            "var_1d": round(var_val, 6),
            "cvar_1d": round(cvar_val, 6),
            "tail_loss_mean": round(_safe_mean(hist_tail, cvar_val), 6),
            "tail_loss_std": round(tail_std, 6),
            "downside_prob": round(downside_prob, 6),
            "sample_size": len(returns),
            "var_norm": round(var_norm, 6),
            "cvar_norm": round(cvar_norm, 6),
            "var_methods": {
                "historical": {"var": round(hist_var, 6), "cvar": round(hist_cvar, 6), "weight": 0.40},
                "parametric": {"var": round(param_var, 6), "cvar": round(param_cvar, 6), "weight": 0.30},
                "monte_carlo": {"var": round(mc_var, 6), "cvar": round(mc_cvar, 6), "weight": 0.30},
            },
        }

    def _parametric_var_cvar(self, losses: List[float], var_alpha: float, cvar_alpha: float) -> Tuple[float, float]:
        if len(losses) < 4:
            return _quantile(losses, var_alpha), _quantile(losses, cvar_alpha)

        mu = _safe_mean(losses)
        sigma = _safe_pstdev(losses, 0.001)

        kurtosis = sum((x - mu) ** 4 for x in losses) / max(len(losses), 1) / max(sigma ** 4, 1e-12)
        excess_kurt = kurtosis - 3.0
        if excess_kurt > 0.5:
            df = max(3.1, min(30.0, 6.0 / excess_kurt + 2.0))
        else:
            df = 30.0

        t_val = self._t_ppf(var_alpha, df)
        var_val = max(0.0, mu + sigma * t_val)

        t_tail = self._t_ppf(cvar_alpha, df)
        pdf_val = self._t_pdf(t_tail, df)
        cvar_multiplier = (pdf_val / (1.0 - cvar_alpha)) * (df + t_tail ** 2) / (df - 1) if df > 2 else t_tail
        cvar_val = max(var_val, mu + sigma * cvar_multiplier)
        return var_val, cvar_val

    def _t_ppf(self, p: float, df: float) -> float:
        if p <= 0.0 or p >= 1.0:
            return 0.0
        t = math.sqrt(-2.0 * math.log(min(p, 1.0 - p)))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        normal_ppf = t - (c0 + c1 * t + c2 * t ** 2) / (1.0 + d1 * t + d2 * t ** 2 + d3 * t ** 3)
        if p < 0.5:
            normal_ppf = -normal_ppf
        z = normal_ppf
        g1 = (z ** 3 + z) / (4.0 * df)
        g2 = (5.0 * z ** 5 + 16.0 * z ** 3 + 3.0 * z) / (96.0 * df ** 2)
        return z + g1 + g2

    def _t_pdf(self, x: float, df: float) -> float:
        coeff = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
        return coeff * (1.0 + x ** 2 / df) ** (-(df + 1) / 2)

    def _monte_carlo_var_cvar(
        self,
        losses: List[float],
        var_alpha: float,
        cvar_alpha: float,
        n_sims: int = 5000,
    ) -> Tuple[float, float]:
        if len(losses) < 2:
            return _quantile(losses, var_alpha), _quantile(losses, cvar_alpha)
        mu = _safe_mean(losses)
        sigma = max(_safe_pstdev(losses, 0.001), 1e-6)
        simulated = [mu + sigma * random.gauss(0, 1) for _ in range(n_sims)]
        mc_var = max(0.0, _quantile(simulated, var_alpha))
        mc_cvar_threshold = _quantile(simulated, cvar_alpha)
        mc_tail = [x for x in simulated if x >= mc_cvar_threshold]
        mc_cvar = max(mc_var, _safe_mean(mc_tail, mc_var))
        return mc_var, mc_cvar

    def _calibrate_dynamic_thresholds(
        self,
        base_thresholds: Tuple[float, float, float],
        factors: Dict[str, Dict[str, Any]],
        data_quality: Dict[str, Any],
        quantile_risk: Dict[str, Any],
        regime_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        vol = _to_float(factors.get("volatility", {}).get("score"), 0.5)
        uncertainty = _to_float(factors.get("uncertainty", {}).get("score"), 0.5)
        reasoning = _to_float(factors.get("reasoning", {}).get("score"), 0.5)
        stress = 0.50 * vol + 0.30 * uncertainty + 0.20 * reasoning
        tail_norm = 0.55 * _to_float(quantile_risk.get("var_norm"), 0.5) + 0.45 * _to_float(quantile_risk.get("cvar_norm"), 0.5)
        quality_penalty = 1.0 - _to_float(data_quality.get("quality_score"), 0.8)
        regime_stress = _clip(_to_float((regime_info or {}).get("state_stress"), 0.5))
        adjustment = 10.0 * stress + 7.5 * tail_norm + 5.5 * quality_penalty + 6.0 * regime_stress + 2.5 * reasoning

        low_base, med_base, high_base = base_thresholds
        low = max(8.0, low_base - 0.34 * adjustment)
        med = max(low + 6.0, med_base - 0.28 * adjustment)
        high = max(med + 6.0, high_base - 0.24 * adjustment)

        regime_shift = self.market_state_engine.threshold_shift_points(regime_info or {})
        low = max(8.0, low + _to_float(regime_shift.get("low"), 0.0))
        med = max(low + 6.0, med + _to_float(regime_shift.get("medium"), 0.0))
        high = max(med + 6.0, high + _to_float(regime_shift.get("high"), 0.0))

        return {
            "base_thresholds": {
                "low": round(low_base, 2),
                "medium": round(med_base, 2),
                "high": round(high_base, 2),
            },
            "thresholds": (low, med, high),
            "calibration": {
                "market_stress": round(stress, 4),
                "reasoning_stress": round(reasoning, 4),
                "tail_risk_norm": round(tail_norm, 4),
                "quality_penalty": round(quality_penalty, 4),
                "regime_stress": round(regime_stress, 4),
                "adjustment": round(adjustment, 4),
                "regime_shift": regime_shift,
            },
        }

    def _compose_quantile_risk_score(
        self,
        quantile_risk: Dict[str, Any],
        factors: Dict[str, Dict[str, Any]],
        weights: Dict[str, float],
        data_quality: Dict[str, Any],
        profile: str,
        regime_info: Optional[Dict[str, Any]] = None,
    ) -> float:
        var_norm = _to_float(quantile_risk.get("var_norm"), 0.5)
        cvar_norm = _to_float(quantile_risk.get("cvar_norm"), 0.5)
        downside_prob = _to_float(quantile_risk.get("downside_prob"), 0.5)
        quantile_component = _clip(0.46 * var_norm + 0.44 * cvar_norm + 0.10 * downside_prob)

        weighted_factors = sum(_to_float(factors[name]["score"]) * _to_float(weights.get(name), 0.0) for name in factors)
        quality_penalty = 1.0 - _to_float(data_quality.get("quality_score"), 1.0)
        regime_stress = _clip(_to_float((regime_info or {}).get("state_stress"), 0.5))

        profile_bias = {"aggressive": -0.02, "moderate": 0.0, "conservative": 0.02}.get(profile, 0.0)
        raw = _clip(
            0.60 * quantile_component +
            0.24 * weighted_factors +
            0.10 * regime_stress +
            0.08 * quality_penalty +
            profile_bias
        )
        return raw * 100.0

    def _build_layered_alerts(
        self,
        factors: Dict[str, Dict[str, Any]],
        data_quality: Dict[str, Any],
        quantile_risk: Dict[str, Any],
        risk_score: float,
        risk_level: str,
        dynamic_thresholds: Tuple[float, float, float],
    ) -> List[Alert]:
        alerts: List[Alert] = []

        def make_alert(code: str, severity: str, message: str, metric: str, value: float, threshold: float) -> None:
            stage = self.SEVERITY_TO_STAGE.get(severity, "L1")
            alerts.append(
                Alert(
                    code=code,
                    severity=severity,
                    message=message,
                    metric=metric,
                    value=value,
                    threshold=threshold,
                    stage=stage,
                    escalation_action=self.STAGE_ESCALATION[stage],
                )
            )

        if _to_float(quantile_risk.get("var_norm"), 0.0) >= 0.85:
            make_alert("VAR_TAIL_BREACH", "high", "VaR tail loss exceeded calibrated budget.", "var_norm", _to_float(quantile_risk.get("var_norm"), 0.0), 0.85)
        if _to_float(quantile_risk.get("cvar_norm"), 0.0) >= 0.92:
            make_alert("CVAR_EXTREME_TAIL", "critical", "CVaR tail loss indicates extreme downside cluster.", "cvar_norm", _to_float(quantile_risk.get("cvar_norm"), 0.0), 0.92)
        if _to_float(factors["volatility"]["score"], 0.0) >= 0.72:
            make_alert("VOLATILITY_SPIKE", "high", "Volatility factor is above dynamic budget.", "volatility_score", _to_float(factors["volatility"]["score"]), 0.72)
        if _to_float(factors["sentiment"]["signals"].get("weighted_negative_ratio"), 0.0) >= 0.56:
            make_alert("SENTIMENT_NEG_CLUSTER", "high", "Negative sentiment cluster is concentrated.", "weighted_negative_ratio", _to_float(factors["sentiment"]["signals"].get("weighted_negative_ratio")), 0.56)
        if _to_float(factors["event"]["score"], 0.0) >= 0.62:
            make_alert("EVENT_RISK_SURGE", "high", "Event-chain risk score is elevated.", "event_score", _to_float(factors["event"]["score"]), 0.62)
        if bool(factors["event"]["signals"].get("tail_event_flag", False)):
            make_alert("EVENT_TAIL_CLUSTER", "critical", "Tail event cluster detected with high irreversibility.", "tail_event_flag", 1.0, 1.0)
        if _to_float(factors["event"]["signals"].get("propagation_index"), 0.0) >= 0.26:
            make_alert("EVENT_INDEX_TRANSMISSION", "medium", "Event transmission has reached index layer.", "propagation_index", _to_float(factors["event"]["signals"].get("propagation_index")), 0.26)
        if _to_float(factors["event"]["signals"].get("irreversibility_avg"), 0.0) >= 0.72:
            make_alert("EVENT_IRREVERSIBILITY_HIGH", "high", "Event set shows high irreversibility.", "irreversibility_avg", _to_float(factors["event"]["signals"].get("irreversibility_avg")), 0.72)
        if _to_float(factors.get("reasoning", {}).get("score"), 0.0) >= 0.64:
            make_alert("REASONING_CONFLICT_HIGH", "high", "Reasoning layer reports high conflict / tail scenario risk.", "reasoning_score", _to_float(factors.get("reasoning", {}).get("score")), 0.64)
        if _to_float((factors.get("reasoning", {}).get("signals", {}) or {}).get("nli_contradiction_ratio"), 0.0) >= 0.45:
            make_alert("NLI_CONTRADICTION_CLUSTER", "medium", "NLI contradiction ratio is elevated.", "nli_contradiction_ratio", _to_float((factors.get("reasoning", {}).get("signals", {}) or {}).get("nli_contradiction_ratio")), 0.45)
        if _to_float(factors["uncertainty"]["score"], 0.0) >= 0.66:
            make_alert("MODEL_UNCERTAINTY_HIGH", "medium", "Model uncertainty is high.", "uncertainty_score", _to_float(factors["uncertainty"]["score"]), 0.66)
        if _to_float(data_quality.get("search_signal_share"), 0.0) >= 0.78:
            make_alert(
                "DATA_SOURCE_IMBALANCE",
                "medium",
                "Search-signal share is too dominant; increase official/primary coverage.",
                "search_signal_share",
                _to_float(data_quality.get("search_signal_share")),
                0.78,
            )
        if _to_float(data_quality.get("search_dominance_penalty"), 0.0) >= 0.70:
            make_alert(
                "DATA_SOURCE_DOMINANCE_HIGH",
                "high",
                "Single source-type dominance is high and may bias risk inference.",
                "search_dominance_penalty",
                _to_float(data_quality.get("search_dominance_penalty")),
                0.70,
            )
        if _to_float(data_quality.get("quality_score"), 0.0) <= 0.35:
            make_alert("DATA_QUALITY_DEGRADED", "high", "Data quality is degraded.", "quality_score", _to_float(data_quality.get("quality_score")), 0.35)

        high_threshold = dynamic_thresholds[2]
        if risk_score >= high_threshold + 8.0:
            make_alert("RISK_SCORE_OVERSHOOT", "critical", "Risk score is materially above dynamic threshold.", "risk_score", risk_score, high_threshold + 8.0)
        elif risk_level == "high":
            make_alert("RISK_SCORE_HIGH", "high", "Risk score entered high-risk regime.", "risk_score", risk_score, high_threshold)

        severity_rank = {"critical": 3, "high": 2, "medium": 1, "low": 0}
        alerts.sort(key=lambda item: severity_rank.get(item.severity, 0), reverse=True)
        return alerts

    def _build_escalation_strategy(self, alerts: List[Alert], risk_level: str, risk_score: float) -> Dict[str, Any]:
        stage_order = {"L1": 1, "L2": 2, "L3": 3, "L4": 4}
        current_stage = "L1"
        if alerts:
            current_stage = max((alert.stage for alert in alerts), key=lambda s: stage_order.get(s, 1))
        if risk_level == "critical":
            current_stage = "L4"
        elif risk_level == "high" and stage_order.get(current_stage, 1) < 3:
            current_stage = "L3"

        grouped: Dict[str, int] = {}
        for alert in alerts:
            grouped[alert.stage] = grouped.get(alert.stage, 0) + 1

        return {
            "current_stage": current_stage,
            "next_action": self.STAGE_ESCALATION[current_stage],
            "alert_stage_count": grouped,
            "risk_score": round(risk_score, 2),
            "rule": "layered_escalation_policy_v1",
        }

    def _build_controls(
        self,
        level: str,
        profile: str,
        risk_score: float,
        factors: Dict[str, Dict[str, Any]],
        alerts: List[Alert],
        quantile_risk: Dict[str, Any],
    ) -> Dict[str, Any]:
        position_caps = {"low": 0.82, "medium": 0.58, "high": 0.35, "critical": 0.15}
        action_map = {
            "low": "normal_monitoring",
            "medium": "heightened_monitoring",
            "high": "risk_alert",
            "critical": "human_review_and_reduce",
        }
        profile_scale = {"aggressive": 1.1, "moderate": 1.0, "conservative": 0.85}
        cap = _clip(position_caps.get(level, 0.35) * profile_scale.get(profile, 1.0), 0.08, 0.90)
        dominant = max(factors.items(), key=lambda item: item[1]["score"])[0]

        steps: List[str] = []
        if level in {"high", "critical"}:
            steps.append("Reduce gross exposure and tighten stop-loss band.")
            steps.append("Suspend new adds until two consecutive rechecks improve.")
        else:
            steps.append("Keep monitoring cadence and re-evaluate at next cycle.")
        if dominant == "volatility":
            steps.append("Deploy hedge overlay and lower intraday limits.")
        elif dominant == "sentiment":
            steps.append("Increase headline refresh cadence and event watchlist.")
        elif dominant == "event":
            steps.append("Harden event playbook and tighten exposure on impacted chain.")
        elif dominant == "reasoning":
            steps.append("Run reasoning drill-down and require multi-source confirmation before new exposure.")
        if any(alert.severity == "critical" for alert in alerts):
            steps.append("Trigger manual override runbook immediately.")

        return {
            "action": action_map.get(level, "risk_alert"),
            "position_limit": round(cap, 2),
            "risk_budget_hint": round(max(0.05, 1.0 - risk_score / 100.0), 2),
            "tail_budget_hint": round(max(0.01, 1.0 - _to_float(quantile_risk.get("cvar_norm"), 0.5)), 2),
            "dominant_factor": dominant,
            "next_steps": steps[:4],
        }

    def _normalize_portfolio_assets(self, assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raw_weights = [_to_float(item.get("weight"), 0.0) for item in assets]
        norm = _normalize_weights(raw_weights)
        normalized = []
        for i, item in enumerate(assets):
            normalized.append(
                {
                    "stock_name": str(item.get("stock_name") or item.get("symbol") or f"asset_{i+1}"),
                    "weight": norm[i] if i < len(norm) else 0.0,
                    "sector": str(item.get("sector") or "unknown"),
                    "style": str(item.get("style") or "unknown"),
                    "leverage": max(0.0, _to_float(item.get("leverage"), 1.0)),
                    "crowding": _clip(_to_float(item.get("crowding"), 0.5)),
                    "returns": item.get("returns"),
                    "predictions_7d": item.get("predictions_7d"),
                    "expected_return": _to_float(item.get("expected_return"), 0.0),
                    "volatility": max(0.0, _to_float(item.get("volatility"), 0.02)),
                }
            )
        return normalized

    def _asset_return_series(self, item: Dict[str, Any]) -> List[float]:
        raw_returns = item.get("returns")
        if isinstance(raw_returns, list):
            parsed = [_to_float(v, 0.0) for v in raw_returns]
            clean = [v for v in parsed if math.isfinite(v)]
            if len(clean) >= 2:
                return clean

        preds = item.get("predictions_7d")
        if isinstance(preds, list):
            clean_preds = [_to_float(v, 0.0) for v in preds if _to_float(v, 0.0) > 0]
            if len(clean_preds) >= 2:
                out = []
                for i in range(1, len(clean_preds)):
                    prev_p = clean_preds[i - 1]
                    curr_p = clean_preds[i]
                    if prev_p > 0:
                        out.append((curr_p - prev_p) / prev_p)
                if len(out) >= 2:
                    return out

        mu = _to_float(item.get("expected_return"), 0.0)
        sigma = max(0.0001, _to_float(item.get("volatility"), 0.02))
        z_grid = [-2.0, -1.3, -0.7, -0.2, 0.2, 0.7, 1.3, 2.0]
        return [mu + sigma * z for z in z_grid]

    def _align_return_series(self, series_list: List[List[float]]) -> List[List[float]]:
        if not series_list:
            return []
        min_len = min(len(s) for s in series_list if s)
        min_len = max(2, min_len)
        aligned = []
        for seq in series_list:
            if len(seq) >= min_len:
                aligned.append(seq[-min_len:])
            else:
                padded = seq[:] + [seq[-1] if seq else 0.0] * (min_len - len(seq))
                aligned.append(padded[-min_len:])
        return aligned

    def _resolve_covariance_matrix(
        self,
        aligned_series: List[List[float]],
        provided: Optional[List[List[float]]],
    ) -> List[List[float]]:
        n = len(aligned_series)
        if n == 0:
            return []
        if provided and len(provided) == n and all(isinstance(row, list) and len(row) == n for row in provided):
            return [[_to_float(v, 0.0) for v in row] for row in provided]
        return self._covariance_matrix(aligned_series)

    def _covariance_matrix(self, aligned_series: List[List[float]]) -> List[List[float]]:
        n = len(aligned_series)
        m = len(aligned_series[0]) if aligned_series else 0
        means = [_safe_mean(s, 0.0) for s in aligned_series]
        cov = [[0.0 for _ in range(n)] for _ in range(n)]
        if m <= 1:
            for i in range(n):
                cov[i][i] = 0.0004
            return cov
        denom = m - 1
        for i in range(n):
            for j in range(n):
                acc = 0.0
                for t in range(m):
                    acc += (aligned_series[i][t] - means[i]) * (aligned_series[j][t] - means[j])
                cov[i][j] = acc / denom
        return cov

    def _portfolio_return_series(self, weights: List[float], aligned_series: List[List[float]]) -> List[float]:
        if not aligned_series:
            return []
        m = len(aligned_series[0])
        portfolio_returns = []
        for t in range(m):
            val = 0.0
            for i, w in enumerate(weights):
                val += w * aligned_series[i][t]
            portfolio_returns.append(val)
        return portfolio_returns

    def _portfolio_synthetic_tail(self, portfolio_returns: List[float], cov: List[List[float]], weights: List[float]) -> List[float]:
        if not weights:
            return []
        port_var = 0.0
        for i, wi in enumerate(weights):
            for j, wj in enumerate(weights):
                port_var += wi * wj * _to_float(cov[i][j], 0.0)
        sigma = math.sqrt(max(port_var, 1e-6))
        mu = _safe_mean(portfolio_returns, 0.0)
        z_grid = [-3.0, -2.4, -2.0, -1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 3.0]
        return [mu + sigma * z for z in z_grid]

    def _portfolio_concentration(self, weights: List[float]) -> Dict[str, Any]:
        n = max(1, len(weights))
        hhi = sum((abs(w) ** 2) for w in weights)
        if n > 1:
            normalized_hhi = _clip((hhi - 1.0 / n) / (1.0 - 1.0 / n))
        else:
            normalized_hhi = 1.0
        return {
            "hhi": round(hhi, 6),
            "normalized_hhi": round(normalized_hhi, 6),
            "largest_weight": round(max([abs(w) for w in weights] or [0.0]), 6),
            "score": round(normalized_hhi, 6),
        }

    def _portfolio_crowding(self, assets: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not assets:
            return {"weighted_crowding": 0.0, "score": 0.0}
        weighted = sum(abs(a["weight"]) * _clip(_to_float(a.get("crowding"), 0.5)) for a in assets)
        gross = sum(abs(a["weight"]) for a in assets)
        crowd = weighted / gross if gross > 0 else 0.0
        return {
            "weighted_crowding": round(crowd, 6),
            "score": round(_clip(crowd), 6),
        }

    def _portfolio_bucket_exposure(self, assets: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
        buckets: Dict[str, float] = {}
        for asset in assets:
            key = str(asset.get(field) or "unknown").lower()
            buckets[key] = buckets.get(key, 0.0) + _to_float(asset.get("weight"), 0.0)
        max_abs = max([abs(v) for v in buckets.values()] or [0.0])
        return {
            "buckets": {k: round(v, 6) for k, v in buckets.items()},
            "max_absolute_exposure": round(max_abs, 6),
            "score": round(_clip(max_abs), 6),
        }

    def _portfolio_leverage(
        self,
        assets: List[Dict[str, Any]],
        gross_leverage: Optional[float],
        net_leverage: Optional[float],
    ) -> Dict[str, Any]:
        implied_gross = sum(abs(a["weight"]) * max(0.0, _to_float(a.get("leverage"), 1.0)) for a in assets)
        implied_net = sum(a["weight"] * max(0.0, _to_float(a.get("leverage"), 1.0)) for a in assets)
        gross = _to_float(gross_leverage, implied_gross)
        net = _to_float(net_leverage, implied_net)
        gross_score = _clip((gross - 1.0) / 1.5, 0.0, 1.0)
        net_score = _clip(abs(net) / 1.2, 0.0, 1.0)
        score = _clip(0.72 * gross_score + 0.28 * net_score)
        return {
            "gross_leverage": round(gross, 6),
            "net_leverage": round(net, 6),
            "score": round(score, 6),
        }

    def _portfolio_covariance_metrics(self, cov: List[List[float]], weights: List[float]) -> Dict[str, Any]:
        n = len(weights)
        if n == 0 or not cov:
            return {"portfolio_variance": 0.0, "avg_abs_corr": 0.0, "score": 0.0}
        port_var = 0.0
        for i in range(n):
            for j in range(n):
                port_var += weights[i] * weights[j] * _to_float(cov[i][j], 0.0)

        corrs = []
        vars_diag = [max(_to_float(cov[i][i], 0.0), 1e-12) for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                denom = math.sqrt(vars_diag[i] * vars_diag[j])
                corr = _to_float(cov[i][j], 0.0) / denom if denom > 0 else 0.0
                corrs.append(abs(corr))
        avg_abs_corr = _safe_mean(corrs, 0.0)
        vol = math.sqrt(max(port_var, 0.0))
        score = _clip(0.55 * _clip(avg_abs_corr) + 0.45 * _clip(vol / 0.045))
        return {
            "portfolio_variance": round(port_var, 8),
            "avg_abs_corr": round(avg_abs_corr, 6),
            "score": round(score, 6),
        }

    def _portfolio_macro_factors(self, assets: List[Dict[str, Any]]) -> Dict[str, float]:
        if not assets:
            return {
                "rate_stress": 0.5,
                "credit_stress": 0.5,
                "inflation_stress": 0.5,
                "fx_stress": 0.5,
                "vix_stress": 0.5,
                "liquidity_stress": 0.5,
                "policy_uncertainty": 0.5,
                "macro_stress": 0.5,
            }
        gross = sum(abs(a["weight"]) for a in assets) or 1.0
        avg_leverage = sum(abs(a["weight"]) * _to_float(a.get("leverage"), 1.0) for a in assets) / gross
        avg_crowding = sum(abs(a["weight"]) * _clip(_to_float(a.get("crowding"), 0.5)) for a in assets) / gross
        cyclical_ratio = sum(
            abs(a["weight"]) for a in assets
            if str(a.get("sector", "")).lower() in {"tech", "consumer", "materials", "industrials"}
        ) / gross

        rate_stress = _clip((avg_leverage - 1.0) / 1.5 + 0.45, 0.0, 1.0)
        credit_stress = _clip(0.35 + 0.55 * avg_leverage, 0.0, 1.0)
        inflation_stress = _clip(0.25 + 0.50 * cyclical_ratio, 0.0, 1.0)
        fx_stress = _clip(0.30 + 0.30 * cyclical_ratio + 0.20 * avg_crowding, 0.0, 1.0)
        vix_stress = _clip(0.35 + 0.55 * avg_crowding, 0.0, 1.0)
        liquidity_stress = _clip(0.30 + 0.65 * avg_crowding + 0.25 * max(0.0, avg_leverage - 1.0), 0.0, 1.0)
        policy_uncertainty = _clip(0.35 + 0.30 * cyclical_ratio, 0.0, 1.0)
        macro_stress = _clip(
            0.18 * rate_stress +
            0.18 * credit_stress +
            0.14 * inflation_stress +
            0.10 * fx_stress +
            0.16 * vix_stress +
            0.14 * liquidity_stress +
            0.10 * policy_uncertainty
        )
        return {
            "rate_stress": round(rate_stress, 6),
            "credit_stress": round(credit_stress, 6),
            "inflation_stress": round(inflation_stress, 6),
            "fx_stress": round(fx_stress, 6),
            "vix_stress": round(vix_stress, 6),
            "liquidity_stress": round(liquidity_stress, 6),
            "policy_uncertainty": round(policy_uncertainty, 6),
            "macro_stress": round(macro_stress, 6),
        }

    def _build_portfolio_alerts(
        self,
        risk_score: float,
        risk_level: str,
        quantile_risk: Dict[str, Any],
        concentration: Dict[str, Any],
        crowding: Dict[str, Any],
        industry_exposure: Dict[str, Any],
        style_exposure: Dict[str, Any],
        leverage: Dict[str, Any],
        covariance_metrics: Dict[str, Any],
        thresholds: Tuple[float, float, float],
    ) -> List[Alert]:
        alerts: List[Alert] = []

        def add(code: str, severity: str, message: str, metric: str, value: float, threshold: float) -> None:
            stage = self.SEVERITY_TO_STAGE.get(severity, "L1")
            alerts.append(
                Alert(
                    code=code,
                    severity=severity,
                    message=message,
                    metric=metric,
                    value=value,
                    threshold=threshold,
                    stage=stage,
                    escalation_action=self.STAGE_ESCALATION[stage],
                )
            )

        if _to_float(quantile_risk.get("cvar_norm"), 0.0) >= 0.90:
            add("PORT_CVAR_TAIL", "critical", "Portfolio CVaR tail risk is extreme.", "cvar_norm", _to_float(quantile_risk.get("cvar_norm")), 0.90)
        if _to_float(concentration.get("normalized_hhi"), 0.0) >= 0.62:
            add("PORT_CONCENTRATION_HIGH", "high", "Portfolio concentration (HHI) exceeded policy.", "normalized_hhi", _to_float(concentration.get("normalized_hhi")), 0.62)
        if _to_float(industry_exposure.get("max_absolute_exposure"), 0.0) >= 0.42:
            add("INDUSTRY_EXPOSURE_HIGH", "high", "Industry exposure is concentrated.", "industry_max_exposure", _to_float(industry_exposure.get("max_absolute_exposure")), 0.42)
        if _to_float(style_exposure.get("max_absolute_exposure"), 0.0) >= 0.45:
            add("STYLE_EXPOSURE_HIGH", "medium", "Style exposure concentration is elevated.", "style_max_exposure", _to_float(style_exposure.get("max_absolute_exposure")), 0.45)
        if _to_float(crowding.get("score"), 0.0) >= 0.70:
            add("CROWDING_HIGH", "high", "Crowding factor is high across positions.", "crowding_score", _to_float(crowding.get("score")), 0.70)
        if _to_float(leverage.get("score"), 0.0) >= 0.65:
            add("LEVERAGE_EXCESS", "high", "Leverage exposure is above risk budget.", "leverage_score", _to_float(leverage.get("score")), 0.65)
        if _to_float(covariance_metrics.get("score"), 0.0) >= 0.66:
            add("COVARIANCE_CLUSTER", "medium", "Cross-asset covariance cluster is elevated.", "covariance_cluster_score", _to_float(covariance_metrics.get("score")), 0.66)

        high_threshold = thresholds[2]
        if risk_score >= high_threshold + 6.0:
            add("PORT_SCORE_OVERSHOOT", "critical", "Portfolio risk score overshot dynamic high threshold.", "risk_score", risk_score, high_threshold + 6.0)
        elif risk_level == "high":
            add("PORT_SCORE_HIGH", "high", "Portfolio risk score entered high-risk regime.", "risk_score", risk_score, high_threshold)

        severity_rank = {"critical": 3, "high": 2, "medium": 1, "low": 0}
        alerts.sort(key=lambda item: severity_rank.get(item.severity, 0), reverse=True)
        return alerts


risk_control_engine = RiskControlEngine()
