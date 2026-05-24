"""Pretrained semantic signal scorer for financial risk."""

from __future__ import annotations

import os
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
except Exception:  # pragma: no cover - runtime fallback
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    pipeline = None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class PretrainedHeadlineRiskScorer:
    """
    Use local pretrained models to produce headline-level semantic downside risk.
    Supports auto-ensemble of multiple local models for better robustness.
    """

    MODEL_CANDIDATES = [
        ("financial_sentiment", "./bert-base-chinese-finetuning-financial-news-sentiment-v2"),
        ("roberta_large", "./chinese-roberta-wwm-ext-large"),
        ("macbert_base", "./chinese-macbert-base"),
        ("deberta_nli", "./nli-deberta-v3-base"),
    ]

    MODEL_PRIOR_WEIGHT = {
        "financial_sentiment": 0.55,
        "roberta_large": 0.22,
        "macbert_base": 0.13,
        "deberta_nli": 0.10,
    }

    def __init__(self, max_samples: int = 8, max_models: int = 3) -> None:
        self.max_samples = max_samples
        self.max_models = max(1, int(max_models))
        self._pipelines: List[Dict[str, Any]] = []
        self._model_alias: Optional[str] = None
        self._model_path: Optional[str] = None
        self._last_error: Optional[str] = None

    def score(self, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not news_items:
            return self._empty_result("empty_input")

        if not self._ensure_loaded():
            return self._empty_result(self._last_error or "model_unavailable")

        samples = news_items[: self.max_samples]
        model_rows: List[Dict[str, Any]] = []
        for pack in self._pipelines:
            model_rows.append(
                {
                    "alias": pack["alias"],
                    "path": pack["path"],
                    "weight": float(pack["weight"]),
                    "risk_probs": [],
                    "neg_probs": [],
                    "errors": 0,
                }
            )

        if not model_rows:
            return self._empty_result("no_loaded_models")

        for item in samples:
            text = self._to_text(item)
            if not text:
                continue
            for i, pack in enumerate(self._pipelines):
                try:
                    raw = pack["pipeline"](text, truncation=True, max_length=256)
                    probs = self._to_sentiment_probs(raw)
                    # Expected downside risk from sentiment distribution.
                    risk = probs["negative"] * 1.0 + probs["neutral"] * 0.35 + probs["positive"] * 0.05
                    model_rows[i]["risk_probs"].append(risk)
                    model_rows[i]["neg_probs"].append(probs["negative"])
                except Exception as e:  # pragma: no cover - inference runtime
                    self._last_error = str(e)
                    model_rows[i]["errors"] += 1

        valid_models = [row for row in model_rows if row["risk_probs"]]
        if not valid_models:
            return self._empty_result(self._last_error or "inference_empty")

        per_model: List[Dict[str, Any]] = []
        ensemble_risk = 0.0
        ensemble_neg = 0.0
        active_weight = 0.0
        risk_means = []
        for row in valid_models:
            risk_mean = mean(row["risk_probs"])
            neg_mean = mean(row["neg_probs"])
            weight = max(1e-6, _safe_float(row["weight"], 0.1))
            risk_means.append(risk_mean)
            active_weight += weight
            ensemble_risk += risk_mean * weight
            ensemble_neg += neg_mean * weight
            per_model.append(
                {
                    "alias": row["alias"],
                    "path": row["path"],
                    "weight": round(weight, 6),
                    "sample_count": len(row["risk_probs"]),
                    "semantic_risk": round(risk_mean, 6),
                    "negative_prob_mean": round(neg_mean, 6),
                    "error_count": int(row["errors"]),
                }
            )

        if active_weight <= 1e-12:
            active_weight = 1.0
        ensemble_risk /= active_weight
        ensemble_neg /= active_weight
        disagreement = pstdev(risk_means) if len(risk_means) > 1 else 0.0
        consensus = max(0.0, min(1.0, 1.0 - disagreement * 2.2))

        return {
            "available": True,
            "model_alias": self._model_alias,   # keep backward compatibility
            "model_path": self._model_path,     # keep backward compatibility
            "sample_count": max((len(r["risk_probs"]) for r in valid_models), default=0),
            "model_count": len(valid_models),
            "model_aliases": [row["alias"] for row in per_model],
            "semantic_risk": round(ensemble_risk, 4),
            "negative_prob_mean": round(ensemble_neg, 4),
            "risk_dispersion": round(disagreement, 4),
            "semantic_consensus": round(consensus, 4),
            "per_model": per_model,
            "error": None,
        }

    def _ensure_loaded(self) -> bool:
        if self._pipelines:
            return True
        if pipeline is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
            self._last_error = "transformers_not_available"
            return False

        candidates = self._resolve_local_models()
        if not candidates:
            self._last_error = "no_local_model_found"
            return False

        loaded: List[Dict[str, Any]] = []
        for alias, path in candidates[: self.max_models]:
            try:
                tokenizer = AutoTokenizer.from_pretrained(path)
                model = AutoModelForSequenceClassification.from_pretrained(
                    path,
                    num_labels=3,
                    ignore_mismatched_sizes=True,
                )
                clf = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    return_all_scores=True,
                    device=-1,
                )
                loaded.append(
                    {
                        "alias": alias,
                        "path": path,
                        "pipeline": clf,
                        "weight": self.MODEL_PRIOR_WEIGHT.get(alias, 0.1),
                    }
                )
            except Exception as e:  # pragma: no cover - load runtime
                self._last_error = str(e)

        if not loaded:
            self._pipelines = []
            return False

        # Normalize prior weights across loaded models.
        total_w = sum(float(row["weight"]) for row in loaded)
        if total_w <= 1e-12:
            total_w = float(len(loaded))
            for row in loaded:
                row["weight"] = 1.0
        for row in loaded:
            row["weight"] = float(row["weight"]) / total_w

        self._pipelines = loaded
        self._model_alias = loaded[0]["alias"]
        self._model_path = loaded[0]["path"]
        self._last_error = None
        return True

    def _resolve_local_models(self) -> List[Tuple[str, str]]:
        rows: List[Tuple[str, str]] = []
        # Optional manual override via env:
        # RISK_SEMANTIC_MODEL_PATHS="alias1:/path1,alias2:/path2"
        raw_env = str(os.getenv("RISK_SEMANTIC_MODEL_PATHS", "")).strip()
        if raw_env:
            for token in raw_env.split(","):
                part = token.strip()
                if not part:
                    continue
                if ":" in part:
                    alias, path = part.split(":", 1)
                    alias = alias.strip()
                    path = path.strip()
                else:
                    path = part
                    alias = os.path.basename(path.rstrip("\\/")) or "custom_model"
                if path and os.path.exists(path):
                    rows.append((alias, path))

        # Built-in candidates as fallback.
        for alias, path in self.MODEL_CANDIDATES:
            if os.path.exists(path) and all(path != p for _, p in rows):
                rows.append((alias, path))
        return rows

    def _to_text(self, item: Dict[str, Any]) -> str:
        title = str(item.get("title", "")).strip()
        content = str(item.get("content", "")).strip()
        if content:
            content = content[:180]
            return f"{title} {content}".strip()
        return title

    def _to_sentiment_probs(self, raw: Any) -> Dict[str, float]:
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            entries = raw[0]
        elif isinstance(raw, list):
            entries = raw
        else:
            entries = []

        neg = 0.0
        pos = 0.0
        neu = 0.0
        for entry in entries:
            label = str(entry.get("label", "")).lower()
            score = _safe_float(entry.get("score"), 0.0)
            if "negative" in label or label.endswith("_0") or label == "label_0":
                neg += score
            elif "positive" in label or label.endswith("_2") or label == "label_2":
                pos += score
            else:
                neu += score

        total = neg + pos + neu
        if total <= 0:
            return {"negative": 1 / 3, "neutral": 1 / 3, "positive": 1 / 3}

        return {
            "negative": neg / total,
            "neutral": neu / total,
            "positive": pos / total,
        }

    def _empty_result(self, reason: str) -> Dict[str, Any]:
        return {
            "available": False,
            "model_alias": self._model_alias,
            "model_path": self._model_path,
            "sample_count": 0,
            "model_count": 0,
            "model_aliases": [],
            "semantic_risk": 0.5,
            "negative_prob_mean": 0.0,
            "risk_dispersion": 0.0,
            "semantic_consensus": 0.0,
            "per_model": [],
            "error": reason,
        }


pretrained_headline_risk_scorer = PretrainedHeadlineRiskScorer()
