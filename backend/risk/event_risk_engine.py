"""Event-risk layer with event type, severity, time decay, reversibility and propagation chain."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Tuple


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_parse_time(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    for fmt in (None, "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            dt = datetime.fromisoformat(text) if fmt is None else datetime.strptime(text, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


class EventRiskEngine:
    """Algorithmic event-risk scorer, no extra pretrained model required."""

    EVENT_LIBRARY: Dict[str, Dict[str, Any]] = {
        "regulatory_penalty": {
            "keywords": [
                "处罚", "罚款", "立案", "调查", "违规", "监管", "约谈", "问询", "停牌",
                "penalty", "probe", "regulatory",
            ],
            "base_severity": 0.78,
            "half_life_hours": 96.0,
            "reversibility": 0.22,
            "sector_beta": 0.62,
            "index_beta": 0.34,
            "direction": +1.0,
        },
        "financial_distress": {
            "keywords": [
                "违约", "暴雷", "破产", "退市", "资不抵债", "流动性危机", "债务", "爆仓",
                "default", "bankruptcy", "distress",
            ],
            "base_severity": 0.90,
            "half_life_hours": 168.0,
            "reversibility": 0.12,
            "sector_beta": 0.70,
            "index_beta": 0.42,
            "direction": +1.0,
        },
        "earnings_miss": {
            "keywords": [
                "亏损", "业绩下滑", "业绩预警", "不及预期", "利润下滑", "营收下滑",
                "miss", "guidance cut",
            ],
            "base_severity": 0.64,
            "half_life_hours": 72.0,
            "reversibility": 0.38,
            "sector_beta": 0.48,
            "index_beta": 0.22,
            "direction": +1.0,
        },
        "litigation_governance": {
            "keywords": [
                "诉讼", "仲裁", "刑事", "高管离职", "舞弊", "造假", "内控缺陷", "审计意见",
                "lawsuit", "fraud", "governance",
            ],
            "base_severity": 0.80,
            "half_life_hours": 144.0,
            "reversibility": 0.20,
            "sector_beta": 0.54,
            "index_beta": 0.30,
            "direction": +1.0,
        },
        "supply_chain_shock": {
            "keywords": [
                "停产", "断供", "事故", "火灾", "召回", "供应链", "断链", "物流中断",
                "shutdown", "recall", "supply chain",
            ],
            "base_severity": 0.70,
            "half_life_hours": 84.0,
            "reversibility": 0.42,
            "sector_beta": 0.58,
            "index_beta": 0.26,
            "direction": +1.0,
        },
        "macro_policy_tightening": {
            "keywords": [
                "加息", "收紧", "去杠杆", "监管趋严", "窗口指导", "压降",
                "rate hike", "tightening",
            ],
            "base_severity": 0.66,
            "half_life_hours": 96.0,
            "reversibility": 0.36,
            "sector_beta": 0.72,
            "index_beta": 0.56,
            "direction": +1.0,
        },
        "policy_support": {
            "keywords": [
                "降息", "降准", "政策支持", "补贴", "回购", "增持", "利好", "放松监管",
                "stimulus", "supportive policy", "easing",
            ],
            "base_severity": 0.42,
            "half_life_hours": 72.0,
            "reversibility": 0.74,
            "sector_beta": 0.44,
            "index_beta": 0.30,
            "direction": -1.0,
        },
        "earnings_beat": {
            "keywords": [
                "超预期", "创新高", "大增", "扭亏", "订单增长", "盈利改善",
                "beat", "record high",
            ],
            "base_severity": 0.38,
            "half_life_hours": 56.0,
            "reversibility": 0.70,
            "sector_beta": 0.32,
            "index_beta": 0.18,
            "direction": -1.0,
        },
        "mna_restructuring": {
            "keywords": [
                "并购", "重组", "资产出售", "拆分", "战略合作", "引战",
                "m&a", "restructuring",
            ],
            "base_severity": 0.52,
            "half_life_hours": 96.0,
            "reversibility": 0.56,
            "sector_beta": 0.36,
            "index_beta": 0.18,
            "direction": +0.40,
        },
        "generic_news": {
            "keywords": [],
            "base_severity": 0.50,
            "half_life_hours": 48.0,
            "reversibility": 0.50,
            "sector_beta": 0.30,
            "index_beta": 0.14,
            "direction": +0.20,
        },
    }

    INTENSITY_UP = {
        "重大": 0.07,
        "严重": 0.08,
        "突发": 0.06,
        "连续": 0.05,
        "扩大": 0.06,
        "暴雷": 0.12,
        "破产": 0.15,
        "刑事": 0.10,
        "巨额": 0.08,
        "紧急": 0.05,
    }
    INTENSITY_DOWN = {
        "澄清": -0.05,
        "恢复": -0.06,
        "和解": -0.06,
        "复产": -0.06,
        "改善": -0.05,
        "回购": -0.04,
    }

    REVERSIBILITY_OVERRIDE_LOW = ("破产", "退市", "刑事", "造假", "资不抵债")
    REVERSIBILITY_OVERRIDE_HIGH = ("回购", "补贴", "恢复", "复产", "和解", "改善")

    def assess(
        self,
        news_items: List[Dict[str, Any]],
        inference_events: Optional[List[Dict[str, Any]]] = None,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        event_records: List[Dict[str, Any]] = []

        for item in news_items:
            event_records.append(self._score_news_item(item, now))
        for ev in (inference_events or [])[:20]:
            event_records.append(self._score_inference_event(ev))

        if not event_records:
            return {
                "aggregate_risk": 0.45,
                "event_count": 0,
                "event_type_distribution": {},
                "event_type_scores": {},
                "dominant_event_type": "none",
                "severity_avg": 0.0,
                "timeliness_avg": 0.0,
                "reversibility_avg": 0.0,
                "irreversibility_avg": 0.0,
                "propagation_scores": {"company": 0.0, "sector": 0.0, "index": 0.0},
                "chain_composite_avg": 0.0,
                "adverse_event_flow": 0.0,
                "positive_event_flow": 0.0,
                "net_event_flow": 0.0,
                "adverse_event_ratio": 0.0,
                "tail_event_flag": False,
                "top_events": [],
            }

        event_type_distribution: Dict[str, int] = {}
        event_type_scores: Dict[str, float] = {}
        severity_list: List[float] = []
        decay_list: List[float] = []
        reversibility_list: List[float] = []
        irreversibility_list: List[float] = []
        chain_company: List[float] = []
        chain_sector: List[float] = []
        chain_index: List[float] = []
        chain_composite: List[float] = []
        adverse_flows: List[float] = []
        beneficial_flows: List[float] = []
        adverse_count = 0

        for rec in event_records:
            event_type = rec["event_type"]
            event_type_distribution[event_type] = event_type_distribution.get(event_type, 0) + 1
            event_type_scores[event_type] = event_type_scores.get(event_type, 0.0) + _to_float(rec.get("adverse_score"), 0.0)
            severity_list.append(_to_float(rec["severity"]))
            decay_list.append(_to_float(rec["timeliness_decay"]))
            reversibility_list.append(_to_float(rec["reversibility"]))
            irreversibility_list.append(_to_float(rec["irreversibility"]))
            chain_company.append(_to_float(rec["chain"]["company"]))
            chain_sector.append(_to_float(rec["chain"]["sector"]))
            chain_index.append(_to_float(rec["chain"]["index"]))
            chain_composite.append(_to_float(rec["chain"]["composite"]))
            adverse_flows.append(_to_float(rec["adverse_score"], 0.0))
            beneficial_flows.append(_to_float(rec["beneficial_score"], 0.0))
            if _to_float(rec.get("direction"), 0.0) > 0:
                adverse_count += 1

        event_count = len(event_records)
        severity_avg = sum(severity_list) / event_count
        decay_avg = sum(decay_list) / event_count
        reversibility_avg = sum(reversibility_list) / event_count
        irreversibility_avg = sum(irreversibility_list) / event_count
        company_avg = sum(chain_company) / event_count
        sector_avg = sum(chain_sector) / event_count
        index_avg = sum(chain_index) / event_count
        chain_avg = sum(chain_composite) / event_count
        adverse_flow = sum(adverse_flows) / event_count
        positive_flow = sum(beneficial_flows) / event_count
        net_event_flow = adverse_flow - positive_flow
        adverse_ratio = adverse_count / event_count

        top_events = sorted(event_records, key=lambda item: _to_float(item.get("adverse_score"), 0.0), reverse=True)[:6]
        tail_event_flag = any(
            _to_float(item.get("adverse_score"), 0.0) >= 0.78 and _to_float(item.get("irreversibility"), 0.0) >= 0.70
            for item in event_records
        )
        tail_bonus = 0.08 if tail_event_flag else 0.0

        aggregate_risk = _clip(
            0.50 * adverse_flow +
            0.17 * irreversibility_avg +
            0.18 * company_avg +
            0.10 * sector_avg +
            0.05 * index_avg -
            0.10 * positive_flow +
            tail_bonus
        )
        dominant_event_type = max(event_type_scores.items(), key=lambda item: item[1])[0] if event_type_scores else "none"

        return {
            "aggregate_risk": round(aggregate_risk, 6),
            "event_count": event_count,
            "event_type_distribution": event_type_distribution,
            "event_type_scores": {key: round(val, 6) for key, val in event_type_scores.items()},
            "dominant_event_type": dominant_event_type,
            "severity_avg": round(severity_avg, 6),
            "timeliness_avg": round(decay_avg, 6),
            "reversibility_avg": round(reversibility_avg, 6),
            "irreversibility_avg": round(irreversibility_avg, 6),
            "propagation_scores": {
                "company": round(company_avg, 6),
                "sector": round(sector_avg, 6),
                "index": round(index_avg, 6),
            },
            "chain_composite_avg": round(chain_avg, 6),
            "adverse_event_flow": round(adverse_flow, 6),
            "positive_event_flow": round(positive_flow, 6),
            "net_event_flow": round(net_event_flow, 6),
            "adverse_event_ratio": round(adverse_ratio, 6),
            "tail_event_flag": bool(tail_event_flag),
            "top_events": top_events,
        }

    def _score_news_item(self, item: Dict[str, Any], now: datetime) -> Dict[str, Any]:
        title = str(item.get("title", "") or "")
        content = str(item.get("content", "") or "")
        summary = str(item.get("summary", "") or "")
        text = f"{title} {summary} {content}".strip().lower()
        if not text:
            text = "generic news"
        signal_type = str(item.get("signal_type", "") or "").strip().lower()
        data_source_type = str(item.get("data_source_type", "") or "").strip().lower()
        category = str(item.get("category", "") or "").strip().lower()

        event_type_hint = str(item.get("event_type", "") or "").strip().lower()
        if event_type_hint in self.EVENT_LIBRARY:
            event_type, match_count = event_type_hint, 1
        else:
            event_type, match_count = self._classify_event_type(text)
        cfg = self.EVENT_LIBRARY[event_type]
        source_weight = _clip(_to_float(item.get("weight"), 0.5))
        sentiment = str(item.get("sentiment_type", "neutral")).lower()

        intensity = 0.0
        for token, val in self.INTENSITY_UP.items():
            if token in text:
                intensity += val
        for token, val in self.INTENSITY_DOWN.items():
            if token in text:
                intensity += val

        sentiment_boost = 0.08 if sentiment == "negative" else -0.06 if sentiment == "positive" else 0.0
        match_boost = min(0.16, 0.03 * max(0, match_count - 1))
        severity = _clip(
            _to_float(cfg["base_severity"]) +
            intensity +
            sentiment_boost +
            0.10 * (source_weight - 0.5) +
            match_boost
        )

        published_at = _safe_parse_time(item.get("publish_time") or item.get("published_at") or item.get("time"))
        age_hours = self._age_hours(now, published_at)
        half_life = max(6.0, _to_float(cfg["half_life_hours"], 48.0))
        timeliness_decay = _clip(math.exp(-math.log(2.0) * age_hours / half_life), 0.05, 1.0)

        reversibility = _clip(_to_float(cfg["reversibility"], 0.5))
        for token in self.REVERSIBILITY_OVERRIDE_LOW:
            if token in text:
                reversibility = min(reversibility, 0.12)
        for token in self.REVERSIBILITY_OVERRIDE_HIGH:
            if token in text:
                reversibility = max(reversibility, 0.72)
        irreversibility = 1.0 - reversibility

        # 多类型检索信号参与传播链：公告/研报/资金流采用不同传播偏置
        company_mult = 1.0
        sector_mult = 1.0
        index_mult = 1.0
        if signal_type == "official_disclosure" or category == "disclosure":
            severity = _clip(severity + 0.04)
            company_mult = 1.10
            sector_mult = 1.06
        elif signal_type == "capital_flow" or category == "capital_flow":
            sector_mult = 1.18
            index_mult = 1.18
        elif signal_type == "analyst_research" or category == "research":
            reversibility = _clip(reversibility + 0.06)
            irreversibility = 1.0 - reversibility
            sector_mult = 1.06

        if data_source_type == "search_signal":
            source_weight = _clip(source_weight * 1.02)

        company_score = _clip(severity * timeliness_decay * company_mult)
        sector_score = _clip(
            company_score * _to_float(cfg["sector_beta"], 0.3) * (0.70 + 0.30 * irreversibility) * sector_mult
        )
        index_score = _clip(
            company_score * _to_float(cfg["index_beta"], 0.2) * (0.60 + 0.40 * irreversibility) * index_mult
        )
        chain_score = _clip(0.52 * company_score + 0.30 * sector_score + 0.18 * index_score)
        event_score = _clip(chain_score * (0.88 + 0.12 * source_weight))
        direction = _to_float(cfg.get("direction"), +1.0)
        adverse_score = event_score if direction > 0 else 0.0
        beneficial_score = event_score if direction < 0 else 0.0

        return {
            "event_type": event_type,
            "severity": round(severity, 6),
            "timeliness_decay": round(timeliness_decay, 6),
            "reversibility": round(reversibility, 6),
            "irreversibility": round(irreversibility, 6),
            "direction": round(direction, 6),
            "event_score": round(event_score, 6),
            "adverse_score": round(adverse_score, 6),
            "beneficial_score": round(beneficial_score, 6),
            "source_weight": round(source_weight, 6),
            "age_hours": round(age_hours, 4),
            "match_count": match_count,
            "chain": {
                "company": round(company_score, 6),
                "sector": round(sector_score, 6),
                "index": round(index_score, 6),
                "composite": round(chain_score, 6),
            },
            "title": title[:140],
            "source": str(item.get("source", "") or ""),
            "signal_type": signal_type,
            "data_source_type": data_source_type,
        }

    def _score_inference_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        impact = _to_float(event.get("impact_score"), 0.0)
        confidence = _clip(_to_float(event.get("confidence"), 0.5))
        severity = _clip(abs(impact) * (0.65 + 0.35 * confidence))
        direction = +1.0 if impact < 0 else -1.0 if impact > 0 else 0.0
        company = _clip(severity)
        sector = _clip(0.45 * severity)
        index = _clip(0.25 * severity)
        chain = _clip(0.52 * company + 0.30 * sector + 0.18 * index)
        adverse_score = chain if direction > 0 else 0.0
        beneficial_score = chain if direction < 0 else 0.0
        return {
            "event_type": str(event.get("event_type", "inference_event"))[:64] or "inference_event",
            "severity": round(severity, 6),
            "timeliness_decay": 0.85,
            "reversibility": 0.50,
            "irreversibility": 0.50,
            "direction": round(direction, 6),
            "event_score": round(chain, 6),
            "adverse_score": round(adverse_score, 6),
            "beneficial_score": round(beneficial_score, 6),
            "source_weight": round(confidence, 6),
            "age_hours": 6.0,
            "match_count": 1,
            "chain": {
                "company": round(company, 6),
                "sector": round(sector, 6),
                "index": round(index, 6),
                "composite": round(chain, 6),
            },
            "title": str(event.get("event_type", "inference_event"))[:140],
            "source": "inference",
        }

    def _classify_event_type(self, text: str) -> Tuple[str, int]:
        best_type = "generic_news"
        best_score = 0
        for event_type, cfg in self.EVENT_LIBRARY.items():
            keywords = cfg.get("keywords", [])
            if not keywords:
                continue
            score = sum(1 for keyword in keywords if keyword and keyword in text)
            if score > best_score:
                best_score = score
                best_type = event_type
        return best_type, best_score

    def _age_hours(self, now: datetime, published_at: Optional[datetime]) -> float:
        if published_at is None:
            return 24.0
        return max(0.0, (now - published_at).total_seconds() / 3600.0)


event_risk_engine = EventRiskEngine()
