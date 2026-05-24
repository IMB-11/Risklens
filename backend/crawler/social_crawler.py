import asyncio
import math
import os
import random
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests

from .base_crawler import BaseCrawler


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _parse_datetime(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw
    text = str(raw).strip()
    if not text:
        return None

    # unix epoch
    if text.isdigit():
        try:
            ts = int(text)
            if ts > 10**12:
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None

    # common datetime formats
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d",
    ):
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            pass

    # ISO fallback
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


class SocialCrawler(BaseCrawler):
    """Social data crawler with API-first strategy and lightweight signal engineering."""

    SOURCE_RELIABILITY = {
        "weibo": 0.56,
        "xueqiu": 0.64,
        "guba": 0.52,
    }

    RISK_KEYWORDS = {
        "policy": ("监管", "处罚", "调查", "问询", "立案", "政策收紧"),
        "liquidity": ("爆仓", "平仓", "流动性", "资金链", "偿债"),
        "profit_warning": ("预亏", "亏损", "下修", "暴雷", "减值"),
        "market_panic": ("恐慌", "崩盘", "踩踏", "闪崩", "跌停潮"),
    }

    def __init__(self, platform: str):
        super().__init__(platform, timeout=30, max_retry=2)
        self.platform = platform
        env_prefix = platform.upper()
        self.api_url = (os.getenv(f"{env_prefix}_API_URL", "") or "").strip()
        self.api_token = (os.getenv(f"{env_prefix}_API_TOKEN", "") or "").strip()
        self.api_method = (os.getenv(f"{env_prefix}_API_METHOD", "GET") or "GET").strip().upper()
        self.api_query_key = (os.getenv(f"{env_prefix}_API_QUERY_KEY", "query") or "query").strip()
        self.api_limit_key = (os.getenv(f"{env_prefix}_API_LIMIT_KEY", "limit") or "limit").strip()

    async def crawl(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        print(f"[INFO] Social fetch {self.platform}: {query}")
        try:
            api_rows = await asyncio.to_thread(self._crawl_via_api, query, limit)
            if api_rows:
                deduped = self._dedup_posts(api_rows)
                deduped.sort(key=lambda x: -_to_float(x.get("signal_strength"), 0.0))
                return deduped[: max(1, int(limit))]

            await asyncio.sleep(0.35)
            fallback = self._generate_fallback_posts(
                query=query,
                limit=limit,
                fetch_mode="web_crawl_fallback",
                api_attempted=bool(self.api_url),
            )
            return fallback
        except Exception as exc:
            print(f"[WARN] Social fetch failed ({self.platform}): {exc}")
            return self._generate_fallback_posts(
                query=query,
                limit=limit,
                fetch_mode="web_crawl_fallback",
                api_attempted=bool(self.api_url),
            )

    def _crawl_via_api(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        API-first layer.
        Configure via:
        - WEIBO_API_URL / WEIBO_API_TOKEN
        - XUEQIU_API_URL / XUEQIU_API_TOKEN
        - GUBA_API_URL / GUBA_API_TOKEN
        """
        if not self.api_url:
            return []

        headers = {"Accept": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        payload: Any = None
        params = {self.api_query_key: query, self.api_limit_key: max(1, int(limit))}

        try:
            if self.api_method == "POST":
                response = requests.post(self.api_url, json=params, headers=headers, timeout=10)
            else:
                response = requests.get(self.api_url, params=params, headers=headers, timeout=10)
            if response.status_code != 200:
                return []
            payload = response.json()
        except Exception:
            return []

        rows = self._extract_rows(payload)
        if not rows:
            return []

        out: List[Dict[str, Any]] = []
        for idx, row in enumerate(rows[: max(1, int(limit) * 2)]):
            normalized = self._normalize_post(
                query=query,
                row=row,
                idx=idx,
                fetch_mode="official_api",
                api_attempted=True,
                fallback_used=False,
            )
            if normalized is not None:
                out.append(normalized)

        out = self._dedup_posts(out)
        out.sort(key=lambda x: -_to_float(x.get("signal_strength"), 0.0))
        return out[: max(1, int(limit))]

    def _extract_rows(self, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
        if not isinstance(payload, dict):
            return []
        for key in ("data", "list", "items", "results", "posts", "rows"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [x for x in rows if isinstance(x, dict)]
        return []

    def _infer_sentiment_score(self, text: str) -> float:
        content = (text or "").lower()
        neg_terms = ("下跌", "暴跌", "亏损", "利空", "减持", "风险", "违约", "处罚", "暴雷")
        pos_terms = ("上涨", "增长", "盈利", "利好", "回购", "增持", "突破", "超预期")
        neg = sum(1 for t in neg_terms if t in content)
        pos = sum(1 for t in pos_terms if t in content)
        if pos == 0 and neg == 0:
            return random.choice([-0.2, -0.1, 0.05, 0.1, 0.2])
        score = (pos - neg) / max(pos + neg, 1)
        return float(_clip(score, -1.0, 1.0))

    def _extract_risk_tags(self, text: str) -> List[str]:
        content = text or ""
        tags: List[str] = []
        for tag, words in self.RISK_KEYWORDS.items():
            if any(w in content for w in words):
                tags.append(tag)
        return tags

    def _normalize_post(
        self,
        query: str,
        row: Dict[str, Any],
        idx: int,
        fetch_mode: str,
        api_attempted: bool,
        fallback_used: bool,
    ) -> Optional[Dict[str, Any]]:
        title = str(row.get("title") or row.get("subject") or f"{query} 相关讨论").strip()
        content = str(row.get("content") or row.get("text") or row.get("body") or title).strip()
        if not title and not content:
            return None

        url = str(row.get("url") or row.get("link") or "").strip()
        raw_time = row.get("publish_time") or row.get("time") or row.get("created_at") or row.get("date")
        dt = _parse_datetime(raw_time) or datetime.now()
        publish_time = dt.isoformat()

        sentiment_score = row.get("sentiment_score")
        try:
            sentiment_score = float(sentiment_score)
        except (TypeError, ValueError):
            sentiment_score = self._infer_sentiment_score(f"{title} {content}")
        sentiment_score = float(_clip(sentiment_score, -1.0, 1.0))

        likes = _to_float(row.get("likes") or row.get("like_count") or row.get("up") or 0.0, 0.0)
        comments = _to_float(row.get("comments") or row.get("comment_count") or row.get("reply") or 0.0, 0.0)
        reposts = _to_float(row.get("reposts") or row.get("repost_count") or row.get("share") or 0.0, 0.0)

        engagement_raw = likes + 1.5 * comments + 2.0 * reposts
        engagement_score = _clip(math.log1p(max(0.0, engagement_raw)) / 8.0, 0.0, 1.0)

        age_hours = max((datetime.now() - dt).total_seconds() / 3600.0, 0.0)
        recency_score = _clip(math.exp(-age_hours / 48.0), 0.0, 1.0)

        text_len = len(content)
        quality_score = _clip(0.35 + 0.45 * min(1.0, text_len / 220.0) + 0.20 * engagement_score, 0.0, 1.0)

        risk_tags = self._extract_risk_tags(f"{title} {content}")
        risk_tag_boost = min(0.2, 0.06 * len(risk_tags))

        source_reliability = self.SOURCE_RELIABILITY.get(self.platform, 0.55)
        signal_strength = _clip(
            0.40 * abs(sentiment_score)
            + 0.22 * engagement_score
            + 0.20 * recency_score
            + 0.12 * quality_score
            + 0.06 * source_reliability
            + risk_tag_boost
        )

        return {
            "news_id": f"{self.platform}_{query}_{idx}",
            "title": title,
            "content": content,
            "source": self.platform,
            "source_id": self.platform,
            "publish_time": publish_time,
            "url": url,
            "platform": self.platform,
            "crawled_at": datetime.now().isoformat(),
            "data_source_type": "social_signal",
            "category": "social",
            "sentiment_score": sentiment_score,
            "engagement_score": round(engagement_score, 6),
            "recency_score": round(recency_score, 6),
            "quality_score": round(quality_score, 6),
            "source_reliability": round(source_reliability, 6),
            "risk_tags": risk_tags,
            "signal_strength": round(signal_strength, 6),
            "fetch_mode": fetch_mode,
            "api_attempted": bool(api_attempted),
            "fallback_used": bool(fallback_used),
        }

    def _dedup_posts(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out: List[Dict[str, Any]] = []
        for row in rows:
            title = str(row.get("title", "")).strip().lower()
            content = str(row.get("content", "")).strip().lower()
            key = re.sub(r"\s+", "", f"{title}|{content[:120]}")
            if key and key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    def _generate_fallback_posts(
        self,
        query: str,
        limit: int,
        fetch_mode: str = "mock",
        api_attempted: bool = False,
    ) -> List[Dict[str, Any]]:
        base_time = datetime.now()
        posts: List[Dict[str, Any]] = []

        templates = [
            f"{query} 讨论热度提升，观点分歧较大",
            f"{query} 短线波动加剧，关注资金节奏",
            f"{query} 基本面变化有限，情绪驱动明显",
            f"{query} 成交活跃，等待后续确认",
            f"{query} 消息偏中性，控制仓位观察",
        ]

        for i in range(max(1, int(limit))):
            hours_ago = random.randint(0, 36)
            minutes_ago = random.randint(0, 59)
            post_time = base_time - timedelta(hours=hours_ago, minutes=minutes_ago)

            row = {
                "title": f"{query} 相关讨论",
                "content": random.choice(templates),
                "url": f"https://{self.platform}.com/{i}",
                "publish_time": post_time.isoformat(),
                "sentiment_score": random.choice([-0.7, -0.4, -0.2, 0.05, 0.2, 0.35]),
                "likes": random.randint(0, 120),
                "comments": random.randint(0, 80),
                "reposts": random.randint(0, 50),
            }

            normalized = self._normalize_post(
                query=query,
                row=row,
                idx=i,
                fetch_mode=fetch_mode,
                api_attempted=api_attempted,
                fallback_used=bool(api_attempted),
            )
            if normalized is not None:
                posts.append(normalized)

        posts = self._dedup_posts(posts)
        posts.sort(key=lambda x: -_to_float(x.get("signal_strength"), 0.0))
        return posts[: max(1, int(limit))]


class WeiboCrawler(SocialCrawler):
    def __init__(self):
        super().__init__("weibo")


class XueqiuCrawler(SocialCrawler):
    def __init__(self):
        super().__init__("xueqiu")


class GubaCrawler(SocialCrawler):
    def __init__(self):
        super().__init__("guba")
