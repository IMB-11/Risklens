"""Risk output enhancement:
1) lightweight risk_sklearn fusion (low weight)
2) DeepSeek API-based narrative rendering with deterministic fallback.
3) DeepSeek API coordination memo across inference/prediction (no decision override).
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

try:
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:  # pragma: no cover
    InconsistentVersionWarning = Warning

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from scipy import sparse as sp
except Exception:  # pragma: no cover
    sp = None

try:
    import requests as _requests
except Exception:  # pragma: no cover
    _requests = None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _normalize_label(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if text in {"0", "low", "l", "低", "low_risk"}:
        return "low"
    if text in {"1", "medium", "mid", "m", "中", "moderate"}:
        return "medium"
    if text in {"2", "high", "h", "高", "critical"}:
        return "high"
    return "medium"


def _risk_label_from_score(score: float, thresholds: List[float]) -> str:
    low_up, mid_up, high_up = thresholds
    if score < low_up:
        return "low"
    if score < mid_up:
        return "medium"
    if score < high_up:
        return "high"
    return "critical"


def _normalize_trend(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if text in {"up", "bull", "buy", "上涨", "看涨", "多头"}:
        return "up"
    if text in {"down", "bear", "sell", "下跌", "看跌", "空头"}:
        return "down"
    if text in {"sideways", "stable", "flat", "hold", "震荡", "中性", "观望"}:
        return "sideways"
    return "unknown"


def _trend_to_action(trend: str) -> str:
    if trend == "up":
        return "buy"
    if trend == "down":
        return "sell"
    return "hold"


class RiskSklearnAdapter:
    """Low-cost sklearn risk classifier adapter."""

    SCORE_HINT = {"low": 32.0, "medium": 56.0, "high": 78.0}

    def __init__(self) -> None:
        self.model_path = os.getenv("RISK_SKLEARN_MODEL_PATH", "./risk_sklearn/model.joblib")
        self.weight = _clip(_safe_float(os.getenv("RISK_SKLEARN_WEIGHT", "0.08"), 0.08), 0.0, 0.20)
        self._model = None
        self._loaded = False
        self._error: Optional[str] = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if joblib is None:
            self._error = "joblib unavailable"
            return
        if not Path(self.model_path).exists():
            self._error = f"model path not found: {self.model_path}"
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InconsistentVersionWarning)
                self._model = joblib.load(self.model_path)
        except Exception as exc:  # pragma: no cover
            self._error = str(exc)

    def _build_text(self, stock_name: str, assessment: Dict[str, Any], news_items: List[Dict[str, Any]]) -> str:
        top_news = []
        for row in (news_items or [])[:8]:
            title = str(row.get("title", "") or "").strip()
            if title:
                top_news.append(title)

        quantile = assessment.get("quantile_risk", {}) or {}
        drivers = ", ".join(assessment.get("top_risk_drivers", [])[:6])
        factors = assessment.get("factor_breakdown", {}) or {}
        lines = [
            f"股票:{stock_name}",
            f"risk_score:{_safe_float(assessment.get('risk_score'), 50.0):.4f}",
            f"risk_level:{assessment.get('risk_level', 'medium')}",
            f"var_1d:{_safe_float(quantile.get('var_1d'), 0.0):.6f}",
            f"cvar_1d:{_safe_float(quantile.get('cvar_1d'), 0.0):.6f}",
            f"drivers:{drivers}",
            "factors:" + json.dumps({k: _safe_float((v or {}).get('score'), 0.5) for k, v in factors.items()}, ensure_ascii=False),
        ]
        if top_news:
            lines.append("news:" + " | ".join(top_news))
        return "\n".join(lines)

    def _build_numeric_features(self, assessment: Dict[str, Any]) -> List[float]:
        factors = assessment.get("factor_breakdown", {}) or {}
        quantile = assessment.get("quantile_risk", {}) or {}
        diagnostics = assessment.get("diagnostics", {}) or {}
        news_snapshot = diagnostics.get("news_snapshot", {}) or {}
        alerts = assessment.get("alerts", []) or []
        return [
            _clip(_safe_float(assessment.get("risk_score"), 50.0) / 100.0),                      # 1
            _clip(_safe_float(quantile.get("var_norm"), 0.5)),                                    # 2
            _clip(_safe_float(quantile.get("cvar_norm"), 0.5)),                                   # 3
            _clip(_safe_float(quantile.get("downside_prob"), 0.5)),                               # 4
            _clip((_safe_float(news_snapshot.get("sentiment_score"), 0.0) + 1.0) / 2.0),         # 5
            _clip(_safe_float((factors.get("trend", {}) or {}).get("score"), 0.5)),              # 6
            _clip(_safe_float((factors.get("volatility", {}) or {}).get("score"), 0.5)),         # 7
            _clip(_safe_float((factors.get("event", {}) or {}).get("score"), 0.5)),              # 8
            _clip(_safe_float((factors.get("uncertainty", {}) or {}).get("score"), 0.5)),        # 9
            _clip(_safe_float((factors.get("market_regime", {}) or {}).get("score"), 0.5)),      # 10
            _clip(len(alerts) / 10.0),                                                            # 11
            _clip(_safe_float(news_snapshot.get("total_news"), 0.0) / 50.0),                      # 12
        ]

    def predict(self, stock_name: str, assessment: Dict[str, Any], news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        self._ensure_loaded()
        if self._model is None:
            return {
                "available": False,
                "error": self._error,
                "weight": self.weight,
            }

        text = self._build_text(stock_name, assessment, news_items)
        try:
            pred: Any = None
            confidence = 0.5

            # Format A: packaged dict {"vectorizer", "scaler", "classifier", ...}
            if isinstance(self._model, dict):
                vectorizer = self._model.get("vectorizer")
                scaler = self._model.get("scaler")
                classifier = self._model.get("classifier")
                labels = self._model.get("labels")
                if vectorizer is None or classifier is None:
                    raise ValueError("risk_sklearn bundle missing vectorizer/classifier")

                X = vectorizer.transform([text])
                if scaler is not None:
                    if np is None or sp is None:
                        raise ValueError("numpy/scipy unavailable for sklearn numeric feature fusion")
                    numeric = np.array([self._build_numeric_features(assessment)], dtype=float)
                    numeric_scaled = scaler.transform(numeric)
                    X = sp.hstack([X, sp.csr_matrix(numeric_scaled)], format="csr")
                pred = classifier.predict(X)[0]
                if isinstance(pred, (int, float)) and isinstance(labels, (list, tuple)):
                    idx = int(pred)
                    if 0 <= idx < len(labels):
                        pred = labels[idx]
                elif isinstance(pred, (int, float)) and isinstance(labels, dict):
                    inv = {int(v): str(k) for k, v in labels.items()}
                    pred = inv.get(int(pred), pred)
                if hasattr(classifier, "predict_proba"):
                    proba = classifier.predict_proba(X)[0]
                    confidence = float(max(proba))
            else:
                # Format B: plain estimator/pipeline
                pred = self._model.predict([text])[0]
                if hasattr(self._model, "predict_proba"):
                    proba = self._model.predict_proba([text])[0]
                    confidence = float(max(proba))

            label = _normalize_label(pred)
            return {
                "available": True,
                "label": label,
                "confidence": round(_clip(confidence), 6),
                "risk_score_hint": self.SCORE_HINT.get(label, 56.0),
                "weight": self.weight,
                "input_length": len(text),
            }
        except Exception as exc:
            return {
                "available": False,
                "error": str(exc),
                "weight": self.weight,
            }


class QwenNarrativeFormatter:
    """Narrative formatter using DeepSeek API with deterministic fallback."""

    API_URL = "https://api.deepseek.com/v1/chat/completions"
    MODEL = "deepseek-chat"

    def __init__(self) -> None:
        self.enabled = os.getenv("ENABLE_QWEN_NARRATIVE", "true").strip().lower() in {"1", "true", "yes", "on"}
        self.coord_enabled = os.getenv("ENABLE_QWEN_COORDINATION", "true").strip().lower() in {"1", "true", "yes", "on"}
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        self.timeout_seconds = max(5.0, _safe_float(os.getenv("QWEN_NARRATIVE_TIMEOUT_SECONDS", "20"), 20.0))
        self.coord_timeout_seconds = max(4.0, _safe_float(os.getenv("QWEN_COORDINATION_TIMEOUT_SECONDS", "12"), 12.0))
        self._init_attempted = False
        self._init_error: Optional[str] = None

    def _ensure_loaded(self) -> None:
        if self._init_attempted:
            return
        self._init_attempted = True
        if not (self.enabled or self.coord_enabled):
            self._init_error = "narrative_disabled_by_env"
            return
        if _requests is None:
            self._init_error = "requests library unavailable"
            return
        if not self.api_key:
            self._init_error = "DEEPSEEK_API_KEY not set"

    def _call_api(self, user_content: str, timeout: float) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.MODEL,
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": 600,
            "temperature": 0.25,
            "top_p": 0.9,
        }
        resp = _requests.post(self.API_URL, json=body, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()

    def _fallback(self, stock_name: str, assessment: Dict[str, Any], sklearn_signal: Dict[str, Any]) -> str:
        q = assessment.get("quantile_risk", {}) or {}
        drivers = assessment.get("top_risk_drivers", [])[:5]
        actions = (assessment.get("control_actions", {}) or {}).get("next_steps", [])[:4]
        base_score = _safe_float(assessment.get("risk_score"), 50.0)
        fused_score = _safe_float(assessment.get("risk_score_fused"), base_score)
        lines = [
            f"【风控摘要】{stock_name}",
            f"- 当前风险分: {base_score:.2f}，融合后风险分: {fused_score:.2f}，等级: {assessment.get('risk_level', 'medium')}",
            f"- 分位风险: VaR(1d)={_safe_float(q.get('var_1d'), 0.0):.4f}, CVaR(1d)={_safe_float(q.get('cvar_1d'), 0.0):.4f}",
            f"- 关键驱动: {', '.join(drivers) if drivers else '暂无'}",
            f"- 小模型信号: {sklearn_signal.get('label', 'n/a')} (conf={_safe_float(sklearn_signal.get('confidence'), 0.0):.2f}, w={_safe_float(sklearn_signal.get('weight'), 0.0):.2f})",
            f"- 建议动作: {'; '.join(actions) if actions else '保持监控并复核数据源'}",
        ]
        return "\n".join(lines)

    def _build_prompt(self, stock_name: str, assessment: Dict[str, Any], news_items: List[Dict[str, Any]], sklearn_signal: Dict[str, Any]) -> str:
        top_news = []
        for row in (news_items or [])[:6]:
            title = str(row.get("title", "") or "").strip()
            source = str(row.get("source", "") or "").strip()
            sentiment = str(row.get("sentiment_type", "neutral") or "neutral")
            if title:
                top_news.append({"title": title[:96], "source": source, "sentiment": sentiment})

        compact = {
            "stock_name": stock_name,
            "risk_score": _safe_float(assessment.get("risk_score"), 50.0),
            "risk_score_fused": _safe_float(assessment.get("risk_score_fused"), _safe_float(assessment.get("risk_score"), 50.0)),
            "risk_level": assessment.get("risk_level", "medium"),
            "risk_label": assessment.get("risk_label", "MEDIUM"),
            "quantile_risk": assessment.get("quantile_risk", {}),
            "top_risk_drivers": assessment.get("top_risk_drivers", []),
            "factor_breakdown": {
                k: round(_safe_float((v or {}).get("score"), 0.5) * 100, 2)
                for k, v in (assessment.get("factor_breakdown", {}) or {}).items()
            },
            "alerts": (assessment.get("alerts", []) or [])[:8],
            "next_steps": (assessment.get("control_actions", {}) or {}).get("next_steps", []),
            "sklearn_signal": sklearn_signal,
            "top_news": top_news,
        }
        payload = json.dumps(compact, ensure_ascii=False)

        return (
            "你是中文证券风控专家。基于下方JSON输出一份结构化且专业的风控说明，"
            "要求: 1) 分点清晰 2) 强调风险证据 3) 给出可执行建议和优先级 4) 不要虚构数据。\n\n"
            "输出格式固定为：\n"
            "一、核心结论\n二、关键风险证据\n三、分位风险与阈值解读\n四、动作建议（按优先级）\n五、监控清单（未来24小时）\n\n"
            f"JSON:\n{payload}\n"
        )

    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        raw = (text or "").strip()
        if not raw:
            return None
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        block = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
        if block:
            try:
                obj = json.loads(block.group(1))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        span = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
        if span:
            snippet = span.group(1)
            try:
                obj = json.loads(snippet)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return None
        return None

    def _coord_prompt(self, payload: Dict[str, Any]) -> str:
        compact = json.dumps(payload, ensure_ascii=False)
        return (
            "你是风控链路协调器。请基于输入JSON，仅输出一个JSON对象，不要任何额外文本。\n"
            "重要约束：你不能做交易决策，不给买卖结论，不改风险分，只做一致性与协同建议。\n"
            "输出字段必须包含：\n"
            "{\n"
            "  \"consistency_score\": 0到1,\n"
            "  \"conflict_level\": \"low|medium|high\",\n"
            "  \"coordination_focus\": [\"...\"],\n"
            "  \"data_contract_checks\": [\"...\"],\n"
            "  \"rationale\": \"一句话，说明协同要点\"\n"
            "}\n"
            f"输入JSON:\n{compact}"
        )

    async def format(
        self,
        stock_name: str,
        assessment: Dict[str, Any],
        news_items: List[Dict[str, Any]],
        sklearn_signal: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "used_qwen": False,
                "error": "narrative_disabled",
                "text": self._fallback(stock_name, assessment, sklearn_signal),
            }
        self._ensure_loaded()
        if self._init_error:
            return {
                "used_qwen": False,
                "error": self._init_error,
                "text": self._fallback(stock_name, assessment, sklearn_signal),
            }

        prompt = self._build_prompt(stock_name, assessment, news_items, sklearn_signal)
        try:
            text = await asyncio.wait_for(
                asyncio.to_thread(self._call_api, prompt, self.timeout_seconds),
                timeout=self.timeout_seconds + 5,
            )
            if not text:
                return {"used_qwen": False, "error": "empty_generation", "text": self._fallback(stock_name, assessment, sklearn_signal)}
            return {"used_qwen": True, "error": None, "text": text}
        except Exception as exc:
            return {
                "used_qwen": False,
                "error": str(exc),
                "text": self._fallback(stock_name, assessment, sklearn_signal),
            }

    async def coordinate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_loaded()
        if not self.coord_enabled:
            return {"used_qwen": False, "error": "coordination_disabled", "patch": {}}
        if self._init_error:
            return {"used_qwen": False, "error": self._init_error, "patch": {}}
        prompt = self._coord_prompt(payload)
        try:
            text = await asyncio.wait_for(
                asyncio.to_thread(self._call_api, prompt, self.coord_timeout_seconds),
                timeout=self.coord_timeout_seconds + 5,
            )
            patch = self._extract_json_object(text) or {}
            if not isinstance(patch, dict):
                patch = {}
            return {"used_qwen": bool(patch), "error": None if patch else "empty_or_invalid_json", "patch": patch}
        except Exception as exc:
            return {"used_qwen": False, "error": str(exc), "patch": {}}


class RiskOutputEnhancer:
    """Orchestrates small-model fusion + narrative output."""

    def __init__(self) -> None:
        self.risk_sklearn = RiskSklearnAdapter()
        self.qwen_formatter = QwenNarrativeFormatter()
        # 默认关闭“小模型改决策”，仅保留旁路参考分。
        self.sklearn_decision_override = (
            os.getenv("ENABLE_RISK_SKLEARN_DECISION_OVERRIDE", "false").strip().lower()
            in {"1", "true", "yes", "on"}
        )

    async def coordinate_chain(
        self,
        stock_name: str,
        news_items: List[Dict[str, Any]],
        inference_result: Dict[str, Any],
        multi_model_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Qwen作为“协调器”而非“决策器”：
        - 不直接改风险分、不改买卖动作；
        - 只做一致性审计、结构补全建议、解释增强元信息输出。
        """
        inf = copy.deepcopy(inference_result or {})
        mm = copy.deepcopy(multi_model_result or {})

        infer_1d = (inf.get("trend_predictions", {}) or {}).get("1d", {}) or {}
        infer_trend = _normalize_trend(infer_1d.get("direction"))
        infer_conf = _clip(_safe_float(infer_1d.get("confidence"), 0.5))

        combined = mm.get("combined_prediction", {}) or {}
        suggestion = mm.get("current_suggestion", {}) or {}
        movement = (mm.get("future_forecast", {}) or {}).get("expected_movement", {}) or {}
        model_trend = _normalize_trend(
            combined.get("trend")
            or movement.get("change_direction")
            or suggestion.get("action_code")
        )
        model_conf = _clip(_safe_float(mm.get("confidence"), _safe_float(combined.get("confidence"), 0.5)))

        sentiment_score = _safe_float((mm.get("sentiment_analysis", {}) or {}).get("sentiment_score"), 0.0)
        sentiment_trend = "up" if sentiment_score > 0.20 else "down" if sentiment_score < -0.20 else "sideways"

        votes = [x for x in [infer_trend, model_trend, sentiment_trend] if x in {"up", "down", "sideways"}]
        mode = max(votes, key=votes.count) if votes else "sideways"
        agreement = votes.count(mode) / max(len(votes), 1)
        conflict_index = round(1.0 - agreement, 6)

        algorithmic_notes: List[str] = []
        if infer_trend != "unknown" and model_trend != "unknown" and infer_trend != model_trend and infer_conf > 0.60 and model_conf > 0.60:
            algorithmic_notes.append("trend_conflict_between_inference_and_prediction")
        if sentiment_trend != "sideways" and model_trend != "unknown" and sentiment_trend != model_trend and abs(sentiment_score) >= 0.35:
            algorithmic_notes.append("sentiment_trend_conflict")
        if not algorithmic_notes and agreement >= 0.67:
            algorithmic_notes.append("cross_module_consistency_good")

        # Qwen only writes coordination memo; it does NOT modify decision fields.
        qwen_payload = {
            "stock_name": stock_name,
            "inference": {
                "trend_1d": infer_trend,
                "confidence_1d": infer_conf,
                "reasoning_meta": inf.get("reasoning_meta", {}),
                "summary": inf.get("summary", {}),
            },
            "prediction": {
                "trend": model_trend,
                "confidence": model_conf,
                "suggestion_action": suggestion.get("action_code"),
                "market_state": mm.get("market_state", {}),
            },
            "sentiment": {
                "score": sentiment_score,
                "trend": sentiment_trend,
                "top_news_titles": [str((n or {}).get("title", ""))[:80] for n in (news_items or [])[:5]],
            },
            "agreement": {
                "mode": mode,
                "agreement_ratio": agreement,
                "conflict_index": conflict_index,
                "algorithmic_notes": algorithmic_notes,
            },
        }
        qwen_coord = await self.qwen_formatter.coordinate(qwen_payload)

        # Put coordination report into metadata only (no decision override).
        coordination_meta = {
            "mode": "coordination_only_no_decision_override",
            "algorithmic": {
                "infer_trend_1d": infer_trend,
                "infer_confidence_1d": round(infer_conf, 6),
                "model_trend": model_trend,
                "model_confidence": round(model_conf, 6),
                "sentiment_trend": sentiment_trend,
                "sentiment_score": round(sentiment_score, 6),
                "agreement_ratio": round(agreement, 6),
                "conflict_index": conflict_index,
                "notes": algorithmic_notes,
            },
            "qwen": {
                "used_qwen": bool(qwen_coord.get("used_qwen", False)),
                "error": qwen_coord.get("error"),
                "coord_patch": qwen_coord.get("patch", {}),
            },
        }
        mm.setdefault("coordination_meta", coordination_meta)
        inf.setdefault("coordination_meta", coordination_meta)
        return {
            "inference_result": inf,
            "multi_model_result": mm,
            "coordination_meta": coordination_meta,
        }

    async def enhance(
        self,
        stock_name: str,
        assessment: Dict[str, Any],
        news_items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        base_score = _safe_float(assessment.get("risk_score"), 50.0)
        sklearn_signal = self.risk_sklearn.predict(stock_name, assessment, news_items)

        shadow_score = base_score
        fused_score = base_score
        applied = False
        if sklearn_signal.get("available"):
            hint = _safe_float(sklearn_signal.get("risk_score_hint"), base_score)
            w = _clip(_safe_float(sklearn_signal.get("weight"), 0.0), 0.0, 0.20)
            shadow_score = (1.0 - w) * base_score + w * hint
            shadow_score = round(_clip(shadow_score, 0.0, 100.0), 4)
            if self.sklearn_decision_override:
                fused_score = shadow_score
                applied = True

        narrative = await self.qwen_formatter.format(stock_name, assessment, news_items, sklearn_signal)

        thresholds = ((assessment.get("dynamic_thresholds") or {}).get("thresholds") or [36.0, 56.0, 74.0])[:3]
        if len(thresholds) < 3:
            thresholds = [36.0, 56.0, 74.0]
        fused_level = _risk_label_from_score(fused_score, [float(thresholds[0]), float(thresholds[1]), float(thresholds[2])])
        shadow_level = _risk_label_from_score(shadow_score, [float(thresholds[0]), float(thresholds[1]), float(thresholds[2])])

        return {
            "risk_sklearn": sklearn_signal,
            "risk_sklearn_applied": applied,
            "risk_sklearn_override_enabled": self.sklearn_decision_override,
            "risk_score_base": round(base_score, 4),
            "risk_score_fused": fused_score,
            "risk_level_fused": fused_level,
            "risk_score_shadow": shadow_score,
            "risk_level_shadow": shadow_level,
            "narrative": narrative,
        }


risk_output_enhancer = RiskOutputEnhancer()
