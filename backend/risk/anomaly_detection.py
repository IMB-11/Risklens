"""Anomaly trading detection engine for Chinese securities risk control.

Pure-algorithm module -- no external network calls or akshare dependency.
Detects suspicious trading patterns via keyword matching, volume-price
z-score analysis, temporal clustering, and sentiment spike scoring.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp *value* into [lower, upper]."""
    return max(lower, min(upper, value))


def _safe_mean(values: List[float], default: float = 0.0) -> float:
    return statistics.mean(values) if values else default


def _safe_stdev(values: List[float], default: float = 0.0) -> float:
    return statistics.pstdev(values) if len(values) > 1 else default


def _z_scores(values: List[float]) -> List[float]:
    """Return population z-scores for *values* (empty list if insufficient)."""
    mu = _safe_mean(values)
    sigma = _safe_stdev(values)
    if sigma < 1e-12:
        return [0.0] * len(values)
    return [(v - mu) / sigma for v in values]


# ---------------------------------------------------------------------------
# Anomaly type keyword dictionaries
# ---------------------------------------------------------------------------

KEYWORD_DICT: Dict[str, List[str]] = {
    "high_frequency": [
        "频繁撤单", "幌骗", "诱多", "诱空", "撤单",
    ],
    "wash_trade": [
        "自买自卖", "倒仓", "对敲", "对倒",
    ],
    "large_order_anomaly": [
        "大单买入", "大单卖出", "巨量", "封板",
    ],
    "insider_trading": [
        "内幕", "泄露", "突击入股", "知情交易",
    ],
    "market_manipulation": [
        "操纵", "坐庄", "拉高出货", "杀跌",
    ],
    "regulatory_risk": [
        "监管", "处罚", "警告", "立案", "调查",
    ],
}

# Weight assigned to each anomaly type when computing the composite score.
_ANOMALY_WEIGHTS: Dict[str, float] = {
    "high_frequency": 0.18,
    "wash_trade": 0.22,
    "large_order_anomaly": 0.12,
    "insider_trading": 0.20,
    "market_manipulation": 0.18,
    "regulatory_risk": 0.10,
}

# Anomaly level thresholds (score -> level).
_LEVEL_THRESHOLDS: List[Tuple[float, str]] = [
    (0.70, "critical"),
    (0.45, "high"),
    (0.25, "medium"),
    (0.0, "low"),
]


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnomalyAlert:
    """Single anomaly alert generated during detection."""

    alert_type: str          # anomaly category key
    severity: str            # low / medium / high / critical
    message: str
    matched_keywords: List[str] = field(default_factory=list)
    confidence: float = 0.0  # 0-1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "matched_keywords": list(self.matched_keywords),
            "confidence": round(self.confidence, 4),
        }


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class AnomalyDetectionEngine:
    """Detect anomalous trading behaviour via multiple signal channels."""

    def __init__(self) -> None:
        self.keyword_dict = KEYWORD_DICT
        self.weights = dict(_ANOMALY_WEIGHTS)
        self.alerts: List[AnomalyAlert] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        stock_name: str,
        news_text: str = "",
        volume_data: Optional[List[float]] = None,
        price_data: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Run anomaly detection and return a result dictionary.

        Parameters
        ----------
        stock_name : str
            Human-readable stock name (e.g. "贵州茅台").
        news_text : str
            Free-text news or announcement content to scan.
        volume_data : list[float] | None
            Historical volume series (e.g. daily volumes over N days).
        price_data : list[float] | None
            Historical price series (e.g. close prices over N days).

        Returns
        -------
        dict with keys: anomaly_score, anomaly_level, trading_anomalies,
        compliance_risks, alerts.
        """
        self.alerts = []
        score_components: Dict[str, float] = {}

        # 1. Keyword matching on news text
        keyword_results = self._keyword_match(news_text)
        for atype, info in keyword_results.items():
            score_components[atype] = info["score"]
            if info["matched"]:
                self.alerts.append(
                    AnomalyAlert(
                        alert_type=atype,
                        severity=self._score_to_severity(info["score"]),
                        message=(
                            f"{stock_name} 新闻文本命中 {atype} "
                            f"关键词: {', '.join(info['matched'])}"
                        ),
                        matched_keywords=info["matched"],
                        confidence=info["score"],
                    )
                )

        # 2. Volume-price anomaly detection
        vp_score = self._volume_price_anomaly(volume_data, price_data)
        if vp_score > 0.1:
            score_components["volume_price_anomaly"] = vp_score
            self.alerts.append(
                AnomalyAlert(
                    alert_type="volume_price_anomaly",
                    severity=self._score_to_severity(vp_score),
                    message=(
                        f"{stock_name} 量价异常指标偏高 "
                        f"(score={vp_score:.3f})"
                    ),
                    confidence=vp_score,
                )
            )

        # 3. Temporal clustering
        tc_score = self._temporal_clustering(volume_data)
        if tc_score > 0.1:
            score_components["temporal_clustering"] = tc_score
            self.alerts.append(
                AnomalyAlert(
                    alert_type="temporal_clustering",
                    severity=self._score_to_severity(tc_score),
                    message=(
                        f"{stock_name} 时间聚集性异常 "
                        f"(score={tc_score:.3f})"
                    ),
                    confidence=tc_score,
                )
            )

        # 4. Sentiment spike
        ss_score = self._sentiment_spike(volume_data, price_data)
        if ss_score > 0.1:
            score_components["sentiment_spike"] = ss_score
            self.alerts.append(
                AnomalyAlert(
                    alert_type="sentiment_spike",
                    severity=self._score_to_severity(ss_score),
                    message=(
                        f"{stock_name} 情绪突变指标偏高 "
                        f"(score={ss_score:.3f})"
                    ),
                    confidence=ss_score,
                )
            )

        # Composite score
        anomaly_score = self._composite_score(score_components)
        anomaly_level = self._score_to_level(anomaly_score)

        # Split alerts into trading vs compliance categories
        trading_types = {
            "high_frequency", "wash_trade", "large_order_anomaly",
            "volume_price_anomaly", "temporal_clustering",
        }
        compliance_types = {
            "insider_trading", "market_manipulation", "regulatory_risk",
            "sentiment_spike",
        }

        trading_anomalies = [
            a.to_dict() for a in self.alerts if a.alert_type in trading_types
        ]
        compliance_risks = [
            a.to_dict() for a in self.alerts if a.alert_type in compliance_types
        ]

        return {
            "anomaly_score": round(anomaly_score, 4),
            "anomaly_level": anomaly_level,
            "trading_anomalies": trading_anomalies,
            "compliance_risks": compliance_risks,
            "alerts": [a.to_dict() for a in self.alerts],
        }

    # ------------------------------------------------------------------
    # Internal detectors
    # ------------------------------------------------------------------

    def _keyword_match(
        self, text: str
    ) -> Dict[str, Dict[str, Any]]:
        """Scan *text* for anomaly keywords.

        Returns a dict keyed by anomaly type, each value being
        ``{"matched": [...], "score": float}``.
        """
        results: Dict[str, Dict[str, Any]] = {}
        if not text:
            for atype in self.keyword_dict:
                results[atype] = {"matched": [], "score": 0.0}
            return results

        for atype, keywords in self.keyword_dict.items():
            matched: List[str] = []
            for kw in keywords:
                if kw in text:
                    matched.append(kw)
            # Score: proportion of keywords matched, capped at 1.0
            score = min(1.0, len(matched) / max(len(keywords), 1))
            results[atype] = {"matched": matched, "score": score}
        return results

    def _volume_price_anomaly(
        self,
        volume_data: Optional[List[float]],
        price_data: Optional[List[float]],
    ) -> float:
        """Z-score based volume-price anomaly.

        Flags days where volume z-score and absolute price change z-score
        are both elevated simultaneously.
        """
        if not volume_data or not price_data:
            return 0.0
        if len(volume_data) < 3 or len(price_data) < 3:
            return 0.0

        vol_z = _z_scores(volume_data)

        # Compute daily absolute price returns
        returns: List[float] = []
        for i in range(1, len(price_data)):
            prev = price_data[i - 1]
            if abs(prev) < 1e-12:
                returns.append(0.0)
            else:
                returns.append(abs((price_data[i] - prev) / prev))
        if not returns:
            return 0.0
        ret_z = _z_scores(returns)

        # Align lengths (vol_z has same length as volume_data, ret_z is shorter)
        min_len = min(len(vol_z), len(ret_z))
        if min_len == 0:
            return 0.0
        vol_z = vol_z[-min_len:]
        ret_z = ret_z[-min_len:]

        # Joint anomaly: days where both z-scores exceed threshold
        threshold = 1.5
        joint_count = sum(
            1 for vz, rz in zip(vol_z, ret_z)
            if abs(vz) > threshold and abs(rz) > threshold
        )
        anomaly_ratio = joint_count / min_len
        return _clip(anomaly_ratio * 3.0)  # scale up, cap at 1.0

    def _temporal_clustering(
        self, volume_data: Optional[List[float]]
    ) -> float:
        """Detect temporal clustering of high-volume days.

        Uses autocorrelation at lag-1 as a proxy for clustering.
        """
        if not volume_data or len(volume_data) < 4:
            return 0.0

        mu = _safe_mean(volume_data)
        sigma = _safe_stdev(volume_data)
        if sigma < 1e-12:
            return 0.0

        normalised = [(v - mu) / sigma for v in volume_data]

        # Lag-1 autocorrelation
        n = len(normalised)
        numerator = sum(
            normalised[i] * normalised[i + 1] for i in range(n - 1)
        )
        denominator = sum(v * v for v in normalised)
        if abs(denominator) < 1e-12:
            return 0.0
        autocorr = numerator / denominator

        # Map autocorrelation to 0-1 score (positive autocorr = clustering)
        return _clip(max(0.0, autocorr))

    def _sentiment_spike(
        self,
        volume_data: Optional[List[float]],
        price_data: Optional[List[float]],
    ) -> float:
        """Detect abrupt volume/price spikes as sentiment-change proxy.

        Looks at the rolling coefficient-of-volume-change and large single-bar
        price moves relative to the rest of the series.
        """
        if not volume_data or not price_data:
            return 0.0
        if len(volume_data) < 5 or len(price_data) < 5:
            return 0.0

        # Volume change ratios
        vol_changes: List[float] = []
        for i in range(1, len(volume_data)):
            prev = volume_data[i - 1]
            if abs(prev) < 1e-12:
                vol_changes.append(0.0)
            else:
                vol_changes.append(
                    abs((volume_data[i] - prev) / prev)
                )

        # Price change ratios
        price_changes: List[float] = []
        for i in range(1, len(price_data)):
            prev = price_data[i - 1]
            if abs(prev) < 1e-12:
                price_changes.append(0.0)
            else:
                price_changes.append(
                    abs((price_data[i] - prev) / prev)
                )

        min_len = min(len(vol_changes), len(price_changes))
        if min_len == 0:
            return 0.0
        vol_changes = vol_changes[-min_len:]
        price_changes = price_changes[-min_len:]

        # Spike: max of product of changes relative to mean
        products = [v * p for v, p in zip(vol_changes, price_changes)]
        mean_prod = _safe_mean(products)
        max_prod = max(products) if products else 0.0

        if mean_prod < 1e-12:
            return _clip(max_prod * 5.0)

        spike_ratio = max_prod / mean_prod
        return _clip(spike_ratio / 10.0)  # normalise roughly to 0-1

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _composite_score(self, components: Dict[str, float]) -> float:
        """Weighted composite of all anomaly signals."""
        if not components:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0
        for key, score in components.items():
            w = self.weights.get(key, 0.1)
            weighted_sum += w * score
            total_weight += w

        if total_weight < 1e-12:
            return 0.0
        return _clip(weighted_sum / total_weight)

    @staticmethod
    def _score_to_severity(score: float) -> str:
        if score >= 0.70:
            return "critical"
        if score >= 0.45:
            return "high"
        if score >= 0.25:
            return "medium"
        return "low"

    @staticmethod
    def _score_to_level(score: float) -> str:
        for threshold, level in _LEVEL_THRESHOLDS:
            if score >= threshold:
                return level
        return "low"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

anomaly_detection_engine = AnomalyDetectionEngine()
