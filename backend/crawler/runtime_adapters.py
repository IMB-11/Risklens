"""Runtime adapters that normalize crawler interfaces for API usage."""

from __future__ import annotations

import asyncio
import hashlib
import os
from datetime import datetime
from typing import Any, Dict, List

from backend.crawler.news_crawler import NewsCrawler
from backend.crawler.social_crawler import WeiboCrawler, XueqiuCrawler, GubaCrawler


def _sentiment_from_text(text: str) -> str:
    content = (text or "").lower()
    negative_terms = (
        "下跌", "暴跌", "亏损", "风险", "处罚", "违约", "下调", "减持", "利空", "崩"
    )
    positive_terms = (
        "上涨", "增长", "盈利", "突破", "回购", "增持", "利好", "创新高", "超预期"
    )
    if any(token in content for token in negative_terms):
        return "negative"
    if any(token in content for token in positive_terms):
        return "positive"
    return "neutral"


def _safe_weight_from_source(
    reliability: Any,
    data_source_type: str,
    signal_type: str,
    category: str,
) -> float:
    try:
        base = float(reliability)
    except (TypeError, ValueError):
        base = 0.5
    base = max(0.01, min(1.0, base))

    source_mult = {
        "official_announcement": 1.12,
        "search_signal": 0.88,
        "search_news": 0.92,
        "news": 1.00,
    }.get(data_source_type, 1.00)

    signal_mult = {
        "official_disclosure": 1.10,
        "capital_flow": 1.03,
        "analyst_research": 0.95,
        "generic_signal": 0.90,
    }.get(signal_type, 1.00)

    category_mult = {
        "disclosure": 1.08,
        "capital_flow": 1.02,
        "research": 0.96,
        "regulatory": 1.06,
    }.get(category, 1.00)

    calibrated = base * source_mult * signal_mult * category_mult
    return max(0.02, min(0.98, calibrated))


def _normalize_news_item(item: Dict[str, Any], fallback_source: str) -> Dict[str, Any]:
    title = item.get("title", "") or ""
    content = item.get("content", "") or ""
    source = item.get("source", fallback_source) or fallback_source
    publish_time = item.get("publish_time", "") or ""
    url = item.get("url", "") or ""
    data_source_type = str(item.get("data_source_type", "news") or "news")
    signal_type = str(item.get("signal_type", "") or "")
    category = str(item.get("category", "general") or "general")

    weight = _safe_weight_from_source(
        reliability=item.get("reliability", 0.5),
        data_source_type=data_source_type,
        signal_type=signal_type,
        category=category,
    )

    sentiment_type = item.get("sentiment_type") or _sentiment_from_text(f"{title} {content}")
    digest = hashlib.md5(f"{source}|{title}|{url}".encode("utf-8")).hexdigest()

    return {
        "title": title,
        "content": content,
        "source": source,
        "source_id": item.get("source_id", source),
        "publish_time": publish_time,
        "url": url,
        "weight": weight,
        "sentiment_type": sentiment_type,
        "data_source_type": data_source_type,
        "category": category,
        "signal_type": signal_type,
        "weight_policy": "source_signal_calibrated_v1",
        "fetch_mode": item.get("fetch_mode", "unknown"),
        "api_attempted": bool(item.get("api_attempted", False)),
        "fallback_used": bool(item.get("fallback_used", False)),
        "content_hash": digest,
        "crawled_at": datetime.now().isoformat(),
    }


class AsyncNewsCrawlerAdapter:
    """Adapts the sync NewsCrawler into async interface used by API layer."""

    # Readme action: API-first + parallel fetching.
    # Core lane includes API-heavy/high-priority sources.
    CORE_SOURCE_IDS = [
        "akshare",
        "serpapi_signal",
        "serpapi",
        "tavily_signal",
        "tavily",
        "eastmoney",
        "sina",
        "sse",
        "szse",
    ]
    # Support lane contains slower/unstable crawl channels.
    FALLBACK_SOURCE_IDS = [
        "cnstock",
        "cs",
        "securities",
        "csrc",
        "pboc",
        "qq",
        "jrj",
        "163",
        "sohu",
    ]

    def __init__(self) -> None:
        self._core_crawler = NewsCrawler()
        self._fallback_crawler = NewsCrawler()
        self._query_count = 0
        self._last_queries: List[str] = []
        self._last_source_runtime: List[Dict[str, Any]] = []
        self._last_search_budget: Dict[str, Any] = {}
        self._parallel_enabled = True
        self._source_stage_timeout_sec = max(
            8,
            int(os.getenv("NEWS_SOURCE_STAGE_TIMEOUT_SEC", "18")),
        )
        self._last_stage_runtime: Dict[str, Any] = {}

    async def crawl(self, query: str, limit: int = 10, apply_final_limit: bool = True) -> List[Dict[str, Any]]:
        self._query_count += 1
        self._last_queries.append(query)
        if len(self._last_queries) > 20:
            self._last_queries = self._last_queries[-20:]

        limit_per_source = max(1, min(3, (max(limit, 1) + 5) // 6))
        core_task = asyncio.create_task(
            asyncio.to_thread(
                self._core_crawler.crawl_news,
                query,
                self.CORE_SOURCE_IDS,
                limit_per_source,
            )
        )
        fallback_task = asyncio.create_task(
            asyncio.to_thread(
                self._fallback_crawler.crawl_news,
                query,
                self.FALLBACK_SOURCE_IDS,
                limit_per_source,
            )
        )

        stage_timeout = float(self._source_stage_timeout_sec)
        done, pending = await asyncio.wait(
            {core_task, fallback_task},
            timeout=stage_timeout,
        )
        timeout_triggered = bool(pending)
        for task in pending:
            task.cancel()

        core_result: Any = TimeoutError("core_stage_timeout")
        fallback_result: Any = TimeoutError("fallback_stage_timeout")
        if core_task in done:
            try:
                core_result = core_task.result()
            except Exception as exc:
                core_result = exc
        if fallback_task in done:
            try:
                fallback_result = fallback_task.result()
            except Exception as exc:
                fallback_result = exc

        self._last_stage_runtime = {
            "parallel_enabled": self._parallel_enabled,
            "stage_timeout_seconds": stage_timeout,
            "timeout_triggered": timeout_triggered,
        }

        core_items = core_result.get("news_list", []) if isinstance(core_result, dict) else []
        fallback_items = fallback_result.get("news_list", []) if isinstance(fallback_result, dict) else []
        raw_items = list(core_items) + list(fallback_items)

        core_runtime = core_result.get("source_runtime", []) if isinstance(core_result, dict) else []
        fallback_runtime = fallback_result.get("source_runtime", []) if isinstance(fallback_result, dict) else []
        self._last_source_runtime = list(core_runtime) + list(fallback_runtime)

        search_budget: Dict[str, Any] = {}
        if isinstance(core_result, dict):
            search_budget.update(core_result.get("search_api_budget", {}) or {})
        if isinstance(fallback_result, dict):
            for key, val in (fallback_result.get("search_api_budget", {}) or {}).items():
                if key not in search_budget:
                    search_budget[key] = val
        self._last_search_budget = search_budget

        normalized = [_normalize_news_item(item, "news") for item in raw_items]
        dedup: Dict[str, Dict[str, Any]] = {}
        for item in normalized:
            key = str(item.get("content_hash") or "")
            if not key:
                key = hashlib.md5(
                    f"{item.get('title', '')}|{item.get('url', '')}|{item.get('source', '')}".encode("utf-8")
                ).hexdigest()
            prev = dedup.get(key)
            if prev is None or float(item.get("weight", 0.0)) > float(prev.get("weight", 0.0)):
                dedup[key] = item
        normalized = list(dedup.values())
        normalized.sort(key=lambda x: -x.get("weight", 0.5))
        if apply_final_limit:
            return normalized[: max(1, int(limit))]
        return normalized

    async def close(self) -> None:
        for crawler in (self._core_crawler, self._fallback_crawler):
            if getattr(crawler, "session", None):
                await asyncio.to_thread(crawler.session.close)

    def get_stats(self) -> Dict[str, Any]:
        mode_counter: Dict[str, int] = {}
        for row in self._last_source_runtime:
            mode = str((row or {}).get("mode", "unknown"))
            mode_counter[mode] = mode_counter.get(mode, 0) + 1

        return {
            "adapter": "AsyncNewsCrawlerAdapter",
            "query_count": self._query_count,
            "recent_queries_count": len(self._last_queries),
            "api_first_enabled": True,
            "parallel_enabled": self._parallel_enabled,
            "last_source_runtime_count": len(self._last_source_runtime),
            "last_source_mode_breakdown": mode_counter,
            "search_api_budget": dict(self._last_search_budget),
            "stage_runtime": dict(self._last_stage_runtime),
        }

    def get_recent_queries(self) -> List[str]:
        return list(self._last_queries)


class AsyncSocialMediaCrawlerAdapter:
    """Aggregates multiple social crawlers into normalized async interface."""

    SOCIAL_WEIGHT_FLOOR = 0.05
    SOCIAL_WEIGHT_CAP = 0.34
    FALLBACK_WEIGHT_DISCOUNT = 0.82

    def __init__(self) -> None:
        self._crawlers = [WeiboCrawler(), XueqiuCrawler(), GubaCrawler()]
        self._query_count = 0
        self._last_mode_breakdown: Dict[str, int] = {}

    async def crawl(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        self._query_count += 1
        per_crawler = max(1, (max(limit, 1) + len(self._crawlers) - 1) // len(self._crawlers))
        tasks = [crawler.crawl(query, limit=per_crawler) for crawler in self._crawlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        items: List[Dict[str, Any]] = []
        mode_breakdown: Dict[str, int] = {}
        for payload in results:
            if isinstance(payload, Exception):
                continue
            for raw in payload:
                score = raw.get("sentiment_score", 0.0)
                try:
                    score = float(score)
                except (TypeError, ValueError):
                    score = 0.0
                sentiment = "positive" if score > 0.25 else "negative" if score < -0.25 else "neutral"
                normalized = _normalize_news_item(raw, raw.get("platform", "social"))
                engagement = max(0.0, min(1.0, float(raw.get("engagement_score", 0.0) or 0.0)))
                reliability = max(0.0, min(1.0, float(raw.get("source_reliability", 0.55) or 0.55)))
                signal_strength = max(0.0, min(1.0, float(raw.get("signal_strength", 0.0) or 0.0)))

                weight = (
                    0.10
                    + 0.14 * min(1.0, abs(score))
                    + 0.06 * engagement
                    + 0.05 * reliability
                    + 0.04 * signal_strength
                )

                fetch_mode = str(normalized.get("fetch_mode", "unknown"))
                if fetch_mode != "official_api":
                    weight *= self.FALLBACK_WEIGHT_DISCOUNT

                weight = max(self.SOCIAL_WEIGHT_FLOOR, min(self.SOCIAL_WEIGHT_CAP, weight))
                normalized["weight"] = round(weight, 6)
                normalized["sentiment_type"] = sentiment
                normalized["weight_policy"] = "social_capped_low_v2"
                normalized["weight_cap"] = self.SOCIAL_WEIGHT_CAP
                mode = fetch_mode
                mode_breakdown[mode] = mode_breakdown.get(mode, 0) + 1
                items.append(normalized)

        self._last_mode_breakdown = mode_breakdown
        items.sort(key=lambda x: -x.get("weight", 0.5))
        return items[:limit]

    async def close(self) -> None:
        for crawler in self._crawlers:
            closer = getattr(crawler, "close", None)
            if callable(closer):
                result = closer()
                if asyncio.iscoroutine(result):
                    await result

    def get_stats(self) -> Dict[str, Any]:
        api_configured = 0
        for crawler in self._crawlers:
            if getattr(crawler, "api_url", ""):
                api_configured += 1
        return {
            "adapter": "AsyncSocialMediaCrawlerAdapter",
            "query_count": self._query_count,
            "crawler_count": len(self._crawlers),
            "api_configured_crawlers": api_configured,
            "social_weight_floor": self.SOCIAL_WEIGHT_FLOOR,
            "social_weight_cap": self.SOCIAL_WEIGHT_CAP,
            "last_mode_breakdown": dict(self._last_mode_breakdown),
        }


class RuntimeNewsAggregator:
    """Minimal aggregator to keep API contract stable."""

    def __init__(self) -> None:
        self._crawlers: List[Any] = []

    async def add_crawler(self, crawler: Any) -> None:
        self._crawlers.append(crawler)

    def get_buffer_stats(self) -> Dict[str, Any]:
        return {
            "registered_crawlers": len(self._crawlers),
            "buffered_events": 0,
        }
