"""China alternative-data hub: real-time collection + historical cache + risk-signal fusion.

This module intentionally avoids model training. It pulls accessible domestic sources,
converts them into normalized "news/event" records, and generates:
1) event_impacts (for inference_result.event_impacts)
2) macro_factors (for multi_model_result.macro_factors)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
from pathlib import Path
import random
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    for fmt in (
        None,
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y%m%d",
    ):
        try:
            dt = datetime.fromisoformat(text) if fmt is None else datetime.strptime(text, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def _clean_html_text(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<[^>]+>", " ", text)
    cleaned = re.sub(r"&nbsp;|&amp;|&quot;|&#39;", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_date_token(text: str) -> Optional[str]:
    if not text:
        return None
    hit = re.search(r"(20\d{2}[./-]\d{1,2}[./-]\d{1,2})", text)
    if not hit:
        return None
    token = hit.group(1).replace(".", "-").replace("/", "-")
    return token


def _try_parse_json_payload(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        pass

    # Handle JSONP wrappers like: callback({...})
    start = text.find("(")
    end = text.rfind(")")
    if start >= 0 and end > start:
        inner = text[start + 1:end].strip()
        try:
            payload = json.loads(inner)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
    return {}


def _normalize_stock_code(raw: str) -> Optional[str]:
    code = re.sub(r"\D", "", str(raw or ""))
    if len(code) != 6:
        return None
    return code


@dataclass(frozen=True)
class PlatformSpec:
    source_id: str
    source_name: str
    base_url: str
    category: str
    reliability: float
    supports_realtime: bool = True
    supports_history: bool = True
    requires_auth: bool = False
    notes: str = ""


class ChinaAltDataEngine:
    """Collect domestic alternative data and produce risk-ready signals."""

    PLATFORM_SPECS: Tuple[PlatformSpec, ...] = (
        PlatformSpec("cninfo", "巨潮资讯", "https://www.cninfo.com.cn/", "official_disclosure", 0.99),
        PlatformSpec("sse", "上海证券交易所", "https://www.sse.com.cn/disclosure/listedinfo/announcement/", "exchange_disclosure", 0.99),
        PlatformSpec("szse", "深圳证券交易所", "https://www.szse.cn/disclosure/listed/index.html", "exchange_disclosure", 0.99),
        PlatformSpec("pbc", "中国人民银行", "https://www.pbc.gov.cn/", "central_bank", 0.98),
        PlatformSpec("nbs", "国家统计局", "https://data.stats.gov.cn/", "macro_statistics", 0.98),
        PlatformSpec("safe", "国家外汇管理局", "https://www.safe.gov.cn/", "macro_statistics", 0.97),
        PlatformSpec("customs", "海关总署", "https://www.customs.gov.cn/", "macro_statistics", 0.97),
        PlatformSpec("cma", "中国气象局", "http://data.cma.cn/", "weather_climate", 0.95),
        PlatformSpec("cnemc", "中国环境监测总站", "https://www.cnemc.cn/", "environmental", 0.95),
        PlatformSpec("baidu_index", "百度指数", "http://index.baidu.com/v2/index.html", "search_heat", 0.88, requires_auth=True, notes="platform login/verification may be required"),
        PlatformSpec("weibo_index", "微博指数", "https://data.weibo.com/index", "social_heat", 0.86, requires_auth=True, notes="platform login/verification may be required"),
        PlatformSpec("chinamoney", "中国货币网", "https://www.chinamoney.com.cn/", "money_market", 0.96),
        PlatformSpec("shibor", "Shibor官网", "https://www.shibor.sh.cn/", "money_market", 0.97),
        PlatformSpec("baidu_hot", "百度热搜", "https://top.baidu.com/board?tab=realtime", "search_heat", 0.89, supports_history=False, notes="public hot-search board; low-frequency crawl"),
        PlatformSpec("eastmoney_hot_rank", "东方财富人气榜", "https://emappdata.eastmoney.com/stockrank/getAllCurrentList", "search_heat", 0.92, supports_history=False, notes="public popularity board; low-frequency crawl"),
        PlatformSpec("eastmoney_fund_flow", "东方财富资金流向", "https://push2his.eastmoney.com/", "capital_flow", 0.97, notes="main fund net inflow/outflow"),
        PlatformSpec("eastmoney_margin", "东方财富融资融券", "https://datacenter-web.eastmoney.com/", "financing", 0.96, notes="margin financing and securities lending"),
    )

    NEGATIVE_TERMS = (
        "处罚", "罚款", "立案", "调查", "违约", "破产", "亏损", "下滑", "暴跌", "减持",
        "监管趋严", "风险", "压力", "收紧", "违纪", "诉讼", "停牌", "退市",
        "default", "penalty", "probe", "distress", "bankruptcy", "downgrade", "selloff",
    )
    POSITIVE_TERMS = (
        "回购", "增持", "增长", "超预期", "盈利改善", "政策支持", "补贴", "修复",
        "降息", "降准", "利好", "突破", "中标", "签约", "盈利", "上调",
        "stimulus", "support", "beat", "growth", "upgrade", "buyback",
    )
    HIGH_SEVERITY_TERMS = (
        "破产", "退市", "刑事", "立案", "处罚", "违约", "暴雷", "流动性危机", "挤兑", "资产减值",
        "bankruptcy", "default", "fraud", "criminal", "insolvency",
    )

    EVENT_TYPE_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
        ("regulatory_penalty", ("处罚", "罚款", "立案", "调查", "监管", "问询", "停牌", "penalty", "probe")),
        ("financial_distress", ("违约", "暴雷", "破产", "退市", "亏损", "资产减值", "default", "bankruptcy")),
        ("macro_policy_tightening", ("加息", "收紧", "去杠杆", "监管趋严", "tightening", "rate hike")),
        ("policy_support", ("降息", "降准", "支持", "补贴", "回购", "stimulus", "easing")),
        ("supply_chain_shock", ("停产", "断供", "事故", "召回", "物流中断", "shutdown", "recall")),
        ("earnings_beat", ("超预期", "创新高", "扭亏", "增长", "beat", "record high")),
    )

    SOURCE_EVENT_TYPE = {
        "official_disclosure": "regulatory_penalty",
        "exchange_disclosure": "regulatory_penalty",
        "central_bank": "macro_policy_tightening",
        "macro_statistics": "macro_policy_tightening",
        "money_market": "macro_policy_tightening",
        "search_heat": "generic_news",
        "social_heat": "generic_news",
        "capital_flow": "financial_distress",
        "financing": "financial_distress",
        "environmental": "supply_chain_shock",
        "weather_climate": "supply_chain_shock",
    }

    USER_AGENT_POOL: Tuple[str, ...] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    )

    def __init__(
        self,
        cache_dir: str = "backend/data/alt_data_cache",
        timeout_sec: int = 5,
        max_retries: int = 3,
        request_concurrency: int = 3,
    ) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._source_cache_dir = self.cache_dir / "_source_cache"
        self._source_cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_sec = timeout_sec
        self.max_retries = max(1, int(max_retries))
        self.request_concurrency = max(1, int(request_concurrency))
        # Lower-frequency policy for sources that are authentication-heavy or unstable.
        self._source_refresh_seconds: Dict[str, int] = {
            "baidu_index": 6 * 3600,
            "weibo_index": 6 * 3600,
            "baidu_hot": 30 * 60,
            "eastmoney_hot_rank": 20 * 60,
            "eastmoney_fund_flow": 10 * 60,
            "eastmoney_margin": 30 * 60,
        }
        self._collector_semaphore = asyncio.Semaphore(self.request_concurrency)
        self.default_headers = {
            "User-Agent": self.USER_AGENT_POOL[0],
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
            "Connection": "keep-alive",
        }
        self._domain_state_lock = threading.Lock()
        self._domain_last_request_ts: Dict[str, float] = {}
        self._domain_penalty_sec: Dict[str, float] = {}
        self._base_interval_sec = 1.2
        self._max_penalty_sec = 8.0
        self._jitter_range = (0.2, 0.9)
        self._stats_lock = threading.Lock()
        self._request_stats: Dict[str, int] = {
            "total": 0,
            "success": 0,
            "retry": 0,
            "rate_limited": 0,
            "forbidden": 0,
            "failed": 0,
            "throttled_cache_hit": 0,
        }

    def _build_headers(self, extra: Optional[Dict[str, str]] = None, referer: Optional[str] = None) -> Dict[str, str]:
        headers = dict(self.default_headers)
        if extra:
            headers.update(extra)
        headers["User-Agent"] = random.choice(self.USER_AGENT_POOL)
        if referer:
            headers["Referer"] = referer
        return headers

    def _domain_from_url(self, url: str) -> str:
        host = urlparse(url).netloc.lower().strip()
        return host or "unknown-host"

    def _reserve_domain_slot(self, domain: str) -> float:
        now = time.monotonic()
        jitter = random.uniform(*self._jitter_range)
        with self._domain_state_lock:
            last_ts = self._domain_last_request_ts.get(domain, 0.0)
            penalty = self._domain_penalty_sec.get(domain, 0.0)
            min_gap = self._base_interval_sec + penalty + jitter
            wait_sec = max(0.0, min_gap - (now - last_ts))
            self._domain_last_request_ts[domain] = now + wait_sec
        return wait_sec

    def _mark_domain_outcome(self, domain: str, outcome: str) -> None:
        with self._domain_state_lock:
            current = self._domain_penalty_sec.get(domain, 0.0)
            if outcome == "ok":
                current = max(0.0, current - 0.30)
            elif outcome == "rate_limited":
                current = min(self._max_penalty_sec, current + 1.80)
            elif outcome == "forbidden":
                current = min(self._max_penalty_sec, current + 1.20)
            else:
                current = min(self._max_penalty_sec, current + 0.60)
            self._domain_penalty_sec[domain] = current

    def _stat_inc(self, key: str) -> None:
        with self._stats_lock:
            self._request_stats[key] = self._request_stats.get(key, 0) + 1

    def _retry_after_seconds(self, response: requests.Response) -> Optional[float]:
        raw = response.headers.get("Retry-After")
        if not raw:
            return None
        raw = raw.strip()
        if not raw:
            return None
        if raw.isdigit():
            return float(raw)
        try:
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return max(0.0, (dt - _now_utc()).total_seconds())
        except Exception:
            return None

    def _backoff_seconds(self, attempt: int, retry_after: Optional[float] = None) -> float:
        exp = min(12.0, 0.9 * (2 ** max(0, attempt)))
        jitter = random.uniform(0.15, 0.95)
        base = exp + jitter
        if retry_after is not None:
            base = max(base, min(25.0, retry_after))
        return base

    def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        referer: Optional[str] = None,
    ) -> requests.Response:
        domain = self._domain_from_url(url)
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            wait_sec = self._reserve_domain_slot(domain)
            if wait_sec > 0:
                time.sleep(wait_sec)
            req_headers = self._build_headers(extra=headers, referer=referer)
            try:
                response = requests.request(
                    method=method.upper(),
                    url=url,
                    params=params,
                    data=data,
                    headers=req_headers,
                    timeout=self.timeout_sec,
                )
                self._stat_inc("total")
                code = int(response.status_code)
                if 200 <= code < 300:
                    self._mark_domain_outcome(domain, "ok")
                    self._stat_inc("success")
                    return response

                if code == 429:
                    self._mark_domain_outcome(domain, "rate_limited")
                    self._stat_inc("rate_limited")
                    retry_after = self._retry_after_seconds(response)
                    last_error = RuntimeError(f"HTTP 429 from {domain}")
                elif code == 403:
                    self._mark_domain_outcome(domain, "forbidden")
                    self._stat_inc("forbidden")
                    retry_after = self._retry_after_seconds(response)
                    last_error = RuntimeError(f"HTTP 403 from {domain}")
                elif code in {408, 500, 502, 503, 504}:
                    self._mark_domain_outcome(domain, "transient")
                    retry_after = self._retry_after_seconds(response)
                    last_error = RuntimeError(f"HTTP {code} from {domain}")
                else:
                    self._mark_domain_outcome(domain, "hard_fail")
                    response.raise_for_status()
                    return response

                if attempt < self.max_retries - 1:
                    self._stat_inc("retry")
                    time.sleep(self._backoff_seconds(attempt, retry_after))
                    continue

            except requests.RequestException as exc:
                self._stat_inc("total")
                self._mark_domain_outcome(domain, "network_fail")
                last_error = exc
                if attempt < self.max_retries - 1:
                    self._stat_inc("retry")
                    time.sleep(self._backoff_seconds(attempt))
                    continue
                break

        self._stat_inc("failed")
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Request failed without error detail: {method} {url}")

    def _request_stats_snapshot(self) -> Dict[str, int]:
        with self._stats_lock:
            return dict(self._request_stats)

    def _domain_penalty_snapshot(self) -> Dict[str, float]:
        with self._domain_state_lock:
            return {k: round(v, 4) for k, v in self._domain_penalty_sec.items() if v > 0}

    def platform_catalog(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for spec in self.PLATFORM_SPECS:
            rows.append(
                {
                    "source_id": spec.source_id,
                    "source_name": spec.source_name,
                    "base_url": spec.base_url,
                    "category": spec.category,
                    "reliability": spec.reliability,
                    "supports_realtime": spec.supports_realtime,
                    "supports_history": spec.supports_history,
                    "requires_auth": spec.requires_auth,
                    "notes": spec.notes,
                }
            )
        return rows

    def runtime_stats(self) -> Dict[str, Any]:
        return {
            "request_stats": self._request_stats_snapshot(),
            "active_domain_penalty": self._domain_penalty_snapshot(),
            "max_retries": self.max_retries,
            "concurrency": self.request_concurrency,
            "source_refresh_seconds": dict(self._source_refresh_seconds),
        }

    def _source_cache_file(self, query: str, source_id: str) -> Path:
        query_hash = hashlib.md5((query or "").encode("utf-8")).hexdigest()[:16]
        return self._source_cache_dir / f"{query_hash}_{source_id}.json"

    def _load_source_cache(self, query: str, source_id: str, max_age_sec: int) -> Optional[List[Dict[str, Any]]]:
        if not query or max_age_sec <= 0:
            return None
        fp = self._source_cache_file(query, source_id)
        if not fp.exists():
            return None
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
            fetched_at = _parse_time(payload.get("fetched_at"))
            if fetched_at is None:
                return None
            age = (_now_utc() - fetched_at).total_seconds()
            if age > max_age_sec:
                return None
            records = payload.get("records", [])
            if isinstance(records, list):
                self._stat_inc("throttled_cache_hit")
                return records
        except Exception:
            return None
        return None

    def _persist_source_cache(self, query: str, source_id: str, records: List[Dict[str, Any]]) -> None:
        if not query:
            return
        fp = self._source_cache_file(query, source_id)
        payload = {
            "query": query,
            "source_id": source_id,
            "fetched_at": _now_utc().isoformat(),
            "records": records,
        }
        fp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _resolve_eastmoney_security(self, query: str) -> Optional[Dict[str, str]]:
        direct_code = _normalize_stock_code(query)
        if direct_code:
            sec_prefix = "1" if direct_code.startswith(("5", "6", "9")) else "0"
            return {
                "code": direct_code,
                "secid": f"{sec_prefix}.{direct_code}",
                "name": query.strip() or direct_code,
            }

        url = "https://searchapi.eastmoney.com/api/suggest/get"
        params = {
            "input": query,
            "type": "14",
            "token": "D43BF722C8E33BDC906FB84D85E326E8",
            "count": "10",
        }
        resp = self._request_with_retry(
            "GET",
            url,
            params=params,
            referer="https://quote.eastmoney.com/",
        )
        payload = _try_parse_json_payload(resp.text)
        table = (((payload.get("QuotationCodeTable") or {}).get("Data")) or payload.get("data") or [])
        if not isinstance(table, list):
            return None

        for row in table:
            if not isinstance(row, dict):
                continue
            code = _normalize_stock_code(row.get("Code") or row.get("SecurityCode") or row.get("scode"))
            if not code:
                continue
            market = str(row.get("MktNum") or row.get("Market") or row.get("market") or "").strip().upper()
            sec_prefix = "1" if market in {"1", "SH", "XSHG"} or code.startswith(("5", "6", "9")) else "0"
            name = str(row.get("Name") or row.get("ShortName") or query).strip() or query
            return {
                "code": code,
                "secid": f"{sec_prefix}.{code}",
                "name": name,
            }
        return None

    async def collect_signals(
        self,
        query: str,
        lookback_days: int = 30,
        limit_per_source: int = 8,
        include_realtime: bool = True,
        include_history: bool = True,
    ) -> Dict[str, Any]:
        safe_query = (query or "").strip()
        realtime_records: List[Dict[str, Any]] = []
        source_status: Dict[str, Dict[str, Any]] = {
            spec.source_id: {
                "source_name": spec.source_name,
                "category": spec.category,
                "requires_auth": spec.requires_auth,
                "collected_count": 0,
                "status": "idle",
                "error": None,
            }
            for spec in self.PLATFORM_SPECS
        }

        if include_realtime and safe_query:
            realtime_records, source_status = await self._collect_realtime_records(
                safe_query,
                lookback_days=max(1, lookback_days),
                limit_per_source=max(1, limit_per_source),
            )
            self._persist_cache_snapshot(safe_query, realtime_records)

        historical_records: List[Dict[str, Any]] = []
        if include_history and safe_query:
            historical_records = self._load_history_from_cache(safe_query, lookback_days=max(1, lookback_days))

        merged_records = self._merge_and_dedup_records([historical_records, realtime_records])
        event_impacts = self._build_event_impacts(merged_records)
        macro_factors = self._build_macro_factors(merged_records)

        return {
            "query": safe_query,
            "as_of": _now_utc().isoformat(),
            "lookback_days": max(1, lookback_days),
            "platform_catalog": self.platform_catalog(),
            "source_status": source_status,
            "news_items": merged_records,
            "event_impacts": event_impacts,
            "macro_factors": macro_factors,
            "summary": self._summary(merged_records, source_status, event_impacts, macro_factors),
            "anti_crawl": self.runtime_stats(),
        }

    async def _collect_realtime_records(
        self,
        query: str,
        lookback_days: int,
        limit_per_source: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        start_dt = (_now_utc() - timedelta(days=lookback_days)).date().isoformat()
        end_dt = _now_utc().date().isoformat()

        source_status: Dict[str, Dict[str, Any]] = {
            spec.source_id: {
                "source_name": spec.source_name,
                "category": spec.category,
                "requires_auth": spec.requires_auth,
                "collected_count": 0,
                "status": "idle",
                "error": None,
            }
            for spec in self.PLATFORM_SPECS
        }

        async def collect_one(spec: PlatformSpec) -> Tuple[str, List[Dict[str, Any]], str, Optional[str]]:
            refresh_sec = int(self._source_refresh_seconds.get(spec.source_id, 0))
            cached_rows = self._load_source_cache(query, spec.source_id, refresh_sec)
            if cached_rows is not None:
                return spec.source_id, cached_rows, "cached", None

            if spec.requires_auth:
                return spec.source_id, [], "auth_required", "requires_auth_or_captcha"
            try:
                async with self._collector_semaphore:
                    if spec.source_id == "cninfo":
                        rows = await asyncio.to_thread(
                            self._collect_cninfo_announcements,
                            spec,
                            query,
                            start_dt,
                            end_dt,
                            limit_per_source,
                        )
                    elif spec.source_id == "sse":
                        rows = await asyncio.to_thread(self._collect_sse_announcements, spec, query, limit_per_source)
                    elif spec.source_id == "szse":
                        rows = await asyncio.to_thread(self._collect_szse_announcements, spec, query, limit_per_source)
                    elif spec.source_id == "baidu_hot":
                        rows = await asyncio.to_thread(self._collect_baidu_hot_search, spec, query, limit_per_source)
                    elif spec.source_id == "eastmoney_hot_rank":
                        rows = await asyncio.to_thread(self._collect_eastmoney_hot_rank, spec, query, limit_per_source)
                    elif spec.source_id == "eastmoney_fund_flow":
                        rows = await asyncio.to_thread(self._collect_eastmoney_fund_flow, spec, query, limit_per_source)
                    elif spec.source_id == "eastmoney_margin":
                        rows = await asyncio.to_thread(self._collect_eastmoney_margin_financing, spec, query, limit_per_source)
                    else:
                        rows = await asyncio.to_thread(self._collect_generic_page_links, spec, query, limit_per_source)
                if rows:
                    self._persist_source_cache(query, spec.source_id, rows)
                return spec.source_id, rows, "ok", None
            except Exception as exc:
                # If live fetching fails, reuse stale cache (if any) to keep pipeline available.
                stale_rows = self._load_source_cache(
                    query,
                    spec.source_id,
                    max(6 * 3600, min(72 * 3600, lookback_days * 24 * 3600)),
                )
                if stale_rows:
                    return spec.source_id, stale_rows, "cached_stale", str(exc)[:300]
                return spec.source_id, [], "error", str(exc)[:300]

        tasks = [collect_one(spec) for spec in self.PLATFORM_SPECS]
        results = await asyncio.gather(*tasks)

        all_records: List[Dict[str, Any]] = []
        for source_id, rows, status, err in results:
            item = source_status[source_id]
            item["collected_count"] = len(rows)
            if status == "ok":
                item["status"] = "ok"
            elif status == "cached":
                item["status"] = "cached"
            elif status == "cached_stale":
                item["status"] = "cached_stale"
                item["error"] = f"live_fetch_failed: {err}" if err else "live_fetch_failed"
            elif status == "auth_required":
                item["status"] = "auth_required"
                item["error"] = "requires_auth_or_captcha"
            elif err:
                item["status"] = "error"
                item["error"] = err
            elif len(rows) == 0:
                item["status"] = "empty"
            all_records.extend(rows)

        return all_records, source_status

    def _collect_cninfo_announcements(
        self,
        spec: PlatformSpec,
        query: str,
        start_date: str,
        end_date: str,
        limit_per_source: int,
    ) -> List[Dict[str, Any]]:
        url = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
        payload = {
            "pageNum": 1,
            "pageSize": max(1, min(50, limit_per_source)),
            "column": "szse",
            "tabName": "fulltext",
            "plate": "",
            "stock": "",
            "searchkey": query,
            "secid": "",
            "category": "",
            "trade": "",
            "seDate": f"{start_date}~{end_date}",
            "sortName": "time",
            "sortType": "desc",
            "isHLtitle": "true",
        }
        headers = self.default_headers.copy()
        headers.update(
            {
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "X-Requested-With": "XMLHttpRequest",
            }
        )
        resp = self._request_with_retry(
            "POST",
            url,
            data=payload,
            headers=headers,
            referer="https://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
        )
        data = resp.json()
        announcements = data.get("announcements") or data.get("data") or []

        out: List[Dict[str, Any]] = []
        for row in announcements[:limit_per_source]:
            title = _clean_html_text(str(row.get("announcementTitle") or row.get("title") or ""))
            if not title:
                continue
            publish_ms = row.get("announcementTime")
            publish_time = ""
            if publish_ms is not None:
                try:
                    publish_time = datetime.fromtimestamp(int(publish_ms) / 1000, tz=timezone.utc).isoformat()
                except Exception:
                    publish_time = ""
            if not publish_time:
                publish_time = _now_utc().isoformat()

            raw_url = str(row.get("adjunctUrl") or row.get("url") or "")
            if raw_url and raw_url.startswith("http"):
                final_url = raw_url
            elif raw_url:
                final_url = urljoin("https://static.cninfo.com.cn/", raw_url.lstrip("/"))
            else:
                final_url = spec.base_url

            out.append(
                self._normalize_record(
                    spec=spec,
                    title=title,
                    content=f"{row.get('secName', '')} {title}",
                    publish_time=publish_time,
                    url=final_url,
                )
            )
        return out

    def _collect_sse_announcements(self, spec: PlatformSpec, query: str, limit_per_source: int) -> List[Dict[str, Any]]:
        params = {
            "jsonCallBack": "jsonpCallback",
            "isPagination": "true",
            "pageHelp.pageSize": max(1, min(30, limit_per_source)),
            "pageHelp.pageNo": 1,
            "pageHelp.beginPage": 1,
            "pageHelp.cacheSize": 1,
            "pageHelp.endPage": 5,
            "productId": "",
            "securityType": "0101,120100,020100,020200,120200",
            "reportType2": "DQBG",
            "reportType": "ALL",
            "beginDate": (_now_utc() - timedelta(days=30)).date().isoformat(),
            "endDate": _now_utc().date().isoformat(),
            "keyWord": query,
        }
        headers = self.default_headers.copy()
        headers.update(
            {
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "X-Requested-With": "XMLHttpRequest",
            }
        )
        url = "http://query.sse.com.cn/security/stock/queryCompanyBulletin.do"
        resp = self._request_with_retry(
            "GET",
            url,
            params=params,
            headers=headers,
            referer="https://www.sse.com.cn/disclosure/listedinfo/announcement/",
        )
        text = resp.text.strip()
        if text.startswith("jsonpCallback(") and text.endswith(")"):
            text = text[len("jsonpCallback("):-1]
        data = json.loads(text)
        page_data = (((data.get("pageHelp") or {}).get("data")) or [])[:limit_per_source]

        out: List[Dict[str, Any]] = []
        for row in page_data:
            title = _clean_html_text(str(row.get("TITLE") or row.get("title") or ""))
            if not title:
                continue
            raw_url = str(row.get("URL") or row.get("url") or "")
            final_url = urljoin("https://www.sse.com.cn/", raw_url.lstrip("/")) if raw_url else spec.base_url
            publish_time = str(row.get("SSEDATE") or row.get("publishTime") or _now_utc().date().isoformat())
            out.append(
                self._normalize_record(
                    spec=spec,
                    title=title,
                    content=title,
                    publish_time=publish_time,
                    url=final_url,
                )
            )
        return out

    def _collect_szse_announcements(self, spec: PlatformSpec, query: str, limit_per_source: int) -> List[Dict[str, Any]]:
        url = "http://www.szse.cn/api/report/ShowReport/data"
        params = {
            "SHOWTYPE": "JSON",
            "CATALOGID": "1803_szse",
            "TABKEY": "tab1",
            "PAGENO": 1,
            "PAGESIZE": max(1, min(30, limit_per_source)),
            "txtQueryKey": query,
            "random": f"0.{int(_now_utc().timestamp())}",
        }
        headers = self.default_headers.copy()
        resp = self._request_with_retry(
            "GET",
            url,
            params=params,
            headers=headers,
            referer="http://www.szse.cn/disclosure/listed/index.html",
        )
        data = resp.json()

        rows: List[Dict[str, Any]] = []
        if isinstance(data, list):
            for block in data:
                rows.extend(block.get("data", []) if isinstance(block, dict) else [])
        elif isinstance(data, dict):
            rows = data.get("data", []) or []
        rows = rows[:limit_per_source]

        out: List[Dict[str, Any]] = []
        for row in rows:
            title = _clean_html_text(str(row.get("title") or row.get("TITLE") or row.get("ssdq") or ""))
            if not title:
                candidate = _clean_html_text(str(row))
                title = candidate[:120]
            if not title:
                continue
            publish_time = str(row.get("fbrq") or row.get("FILLDATE") or _now_utc().date().isoformat())
            url_field = str(row.get("url") or row.get("URL") or "")
            final_url = urljoin("https://www.szse.cn/", url_field.lstrip("/")) if url_field else spec.base_url
            out.append(
                self._normalize_record(
                    spec=spec,
                    title=title,
                    content=title,
                    publish_time=publish_time,
                    url=final_url,
                )
            )
        return out

    def _collect_baidu_hot_search(self, spec: PlatformSpec, query: str, limit_per_source: int) -> List[Dict[str, Any]]:
        resp = self._request_with_retry("GET", spec.base_url, headers=self.default_headers, referer=spec.base_url)
        html = resp.text
        # Baidu hot board is mostly rendered client-side; keep parser permissive.
        titles = re.findall(r"c-single-text-ellipsis[^>]*>(.*?)<", html, flags=re.IGNORECASE)
        if not titles:
            titles = re.findall(r"<a[^>]*>(.*?)</a>", html, flags=re.IGNORECASE | re.DOTALL)
        cleaned = [_clean_html_text(t) for t in titles]
        cleaned = [t for t in cleaned if 2 <= len(t) <= 80]

        hits: List[Tuple[int, str]] = []
        for idx, title in enumerate(cleaned[:120], start=1):
            if query in title:
                hits.append((idx, title))
        if not hits:
            return []

        out: List[Dict[str, Any]] = []
        for rank, title in hits[:limit_per_source]:
            rank_score = _clip(1.0 - (rank - 1) / 120.0, 0.1, 1.0)
            row = self._normalize_record(
                spec=spec,
                title=f"{query} 热搜关注度提升 (Baidu rank={rank})",
                content=f"{title} | 热搜榜位置: {rank}",
                publish_time=_now_utc().isoformat(),
                url=spec.base_url,
            )
            row["metric_type"] = "search_heat"
            row["search_heat_score"] = round(rank_score, 6)
            row["search_heat_hits"] = len(hits)
            out.append(row)
        return out

    def _collect_eastmoney_hot_rank(self, spec: PlatformSpec, query: str, limit_per_source: int) -> List[Dict[str, Any]]:
        security = self._resolve_eastmoney_security(query)
        if not security:
            return []
        stock_code = security["code"]
        symbol = ("SH" if security["secid"].startswith("1.") else "SZ") + stock_code

        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://quote.eastmoney.com",
            "Referer": "https://quote.eastmoney.com/",
        }

        rows: List[Dict[str, Any]] = []
        # First fetch historical/current rank for the specific security.
        payload = {
            "appId": "appId01",
            "globalId": "786e4c21-70dc-435a-93bb-38",
            "marketType": "",
            "srcSecurityCode": symbol,
        }
        resp = self._request_with_retry(
            "POST",
            "https://emappdata.eastmoney.com/stockrank/getCurrentList",
            data=json.dumps(payload, ensure_ascii=False),
            headers=headers,
            referer="https://quote.eastmoney.com/",
        )
        data = _try_parse_json_payload(resp.text)
        rank_rows = data.get("data", [])
        if isinstance(rank_rows, list):
            rows = [row for row in rank_rows if isinstance(row, dict)]

        if not rows:
            # Fallback to top list and try to locate the same stock.
            payload = {
                "appId": "appId01",
                "globalId": "786e4c21-70dc-435a-93bb-38",
                "marketType": "",
                "pageNo": 1,
                "pageSize": 100,
            }
            resp = self._request_with_retry(
                "POST",
                spec.base_url,
                data=json.dumps(payload, ensure_ascii=False),
                headers=headers,
                referer="https://quote.eastmoney.com/",
            )
            data = _try_parse_json_payload(resp.text)
            all_rows = data.get("data", [])
            if isinstance(all_rows, list):
                for row in all_rows:
                    if not isinstance(row, dict):
                        continue
                    code = str(row.get("sc") or row.get("code") or row.get("srcSecurityCode") or "")
                    name = str(row.get("name") or row.get("securityName") or "")
                    if stock_code in code or query in name:
                        rows = [row]
                        break

        if not rows:
            return []

        out: List[Dict[str, Any]] = []
        for row in rows[:max(1, limit_per_source)]:
            rank_value = _to_float(row.get("rk") or row.get("rank") or row.get("currentRank"), 999.0)
            delta_value = _to_float(row.get("rkDiff") or row.get("rankDiff") or row.get("rankChange"), 0.0)
            rank_score = _clip(1.0 - (rank_value - 1.0) / 150.0, 0.1, 1.0)
            delta_score = _clip(abs(delta_value) / 60.0, 0.0, 1.0)
            heat_score = _clip(0.75 * rank_score + 0.25 * delta_score)
            title = f"{query} 东方财富人气榜 rank={int(rank_value) if rank_value < 900 else 'N/A'}"
            content = f"code={stock_code}, rank={rank_value}, rank_delta={delta_value}, heat_score={round(heat_score, 4)}"
            normalized = self._normalize_record(
                spec=spec,
                title=title,
                content=content,
                publish_time=_now_utc().isoformat(),
                url="https://guba.eastmoney.com/rank/",
            )
            normalized["metric_type"] = "search_heat"
            normalized["search_heat_score"] = round(heat_score, 6)
            normalized["search_heat_rank"] = rank_value
            normalized["search_heat_rank_delta"] = delta_value
            out.append(normalized)
        return out

    def _collect_eastmoney_fund_flow(self, spec: PlatformSpec, query: str, limit_per_source: int) -> List[Dict[str, Any]]:
        security = self._resolve_eastmoney_security(query)
        if not security:
            return []
        secid = security["secid"]
        stock_code = security["code"]

        params = {
            "lmt": max(3, min(30, limit_per_source)),
            "klt": "101",
            "secid": secid,
            "fields1": "f1,f2,f3,f7",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63",
            "ut": "b2884a393a59ad64002292a3e90d46a5",
            "_": str(int(time.time() * 1000)),
        }
        resp = self._request_with_retry(
            "GET",
            "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get",
            params=params,
            referer="https://quote.eastmoney.com/",
        )
        payload = _try_parse_json_payload(resp.text)
        klines = (((payload.get("data") or {}).get("klines")) or [])
        if not isinstance(klines, list) or not klines:
            return []

        out: List[Dict[str, Any]] = []
        for line in klines[:max(1, limit_per_source)]:
            cols = str(line).split(",")
            if len(cols) < 3:
                continue
            trade_date = cols[0]
            main_net_inflow = _to_float(cols[1], 0.0)
            net_signal = math.tanh(main_net_inflow / 5e8)
            inflow_yi = main_net_inflow / 1e8
            sentiment = "positive" if net_signal > 0.04 else ("negative" if net_signal < -0.04 else "neutral")
            sentiment_score = _clip(0.5 + min(0.45, abs(net_signal)) * (1 if sentiment != "neutral" else 0.4))

            row = self._normalize_record(
                spec=spec,
                title=f"{query} 资金流向: {'净流入' if main_net_inflow >= 0 else '净流出'} {inflow_yi:.2f} 亿",
                content=f"trade_date={trade_date}, code={stock_code}, main_net_inflow={main_net_inflow}",
                publish_time=trade_date,
                url=f"https://quote.eastmoney.com/{stock_code}.html",
            )
            row["sentiment_type"] = sentiment
            row["sentiment_score"] = round(sentiment_score, 6)
            row["event_type"] = "liquidity_flow"
            row["metric_type"] = "capital_flow"
            row["capital_flow_main_net_inflow"] = round(main_net_inflow, 3)
            row["capital_flow_signal"] = round(net_signal, 6)
            out.append(row)
        return out

    def _collect_eastmoney_margin_financing(self, spec: PlatformSpec, query: str, limit_per_source: int) -> List[Dict[str, Any]]:
        security = self._resolve_eastmoney_security(query)
        if not security:
            return []
        stock_code = security["code"]
        url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
        base_params = {
            "reportName": "RPTA_WEB_RZRQ_GGMX",
            "columns": "ALL",
            "source": "WEB",
            "client": "WEB",
            "pageNumber": 1,
            "pageSize": max(2, min(30, limit_per_source)),
            "sortColumns": "DATE",
            "sortTypes": "-1",
        }

        all_rows: List[Dict[str, Any]] = []
        for filt in (f"(SCODE='{stock_code}')", f"(SCODE={stock_code})", f"(SECUCODE='{stock_code}')"):
            params = dict(base_params)
            params["filter"] = filt
            resp = self._request_with_retry("GET", url, params=params, referer="https://data.eastmoney.com/rzrq/")
            payload = _try_parse_json_payload(resp.text)
            rows = (((payload.get("result") or {}).get("data")) or payload.get("data") or [])
            if isinstance(rows, list) and rows:
                all_rows = [row for row in rows if isinstance(row, dict)]
                break

        if not all_rows:
            return []

        out: List[Dict[str, Any]] = []
        prev_rzye: Optional[float] = None
        for row in all_rows[:max(1, limit_per_source)]:
            trade_date = str(row.get("DATE") or row.get("TRADE_DATE") or _now_utc().date().isoformat())
            rzye = _to_float(row.get("RZYE") or row.get("rzye"), 0.0)
            rzjme = _to_float(row.get("RZJME") or row.get("rzjme"), 0.0)
            rzmre = _to_float(row.get("RZMRE") or row.get("rzmre"), 0.0)
            rzche = _to_float(row.get("RZCHE") or row.get("rzche"), 0.0)
            rzye_change = rzye - prev_rzye if prev_rzye is not None else _to_float(row.get("RZRQYECZ"), 0.0)
            prev_rzye = rzye

            leverage_signal = math.tanh(rzye_change / 4e8)
            sentiment = "negative" if leverage_signal > 0.06 else ("positive" if leverage_signal < -0.06 else "neutral")
            sentiment_score = _clip(0.5 + min(0.42, abs(leverage_signal)) * (1 if sentiment != "neutral" else 0.35))
            rzye_yi = rzye / 1e8
            delta_yi = rzye_change / 1e8

            normalized = self._normalize_record(
                spec=spec,
                title=f"{query} 融资余额 {rzye_yi:.2f} 亿 (Δ {delta_yi:+.2f} 亿)",
                content=(
                    f"trade_date={trade_date}, code={stock_code}, rzye={rzye}, rzye_change={rzye_change}, "
                    f"rzmre={rzmre}, rzche={rzche}, rzjme={rzjme}"
                ),
                publish_time=trade_date,
                url=f"https://data.eastmoney.com/rzrq/detail/{stock_code}.html",
            )
            normalized["sentiment_type"] = sentiment
            normalized["sentiment_score"] = round(sentiment_score, 6)
            normalized["event_type"] = "financing_leverage"
            normalized["metric_type"] = "financing"
            normalized["financing_balance"] = round(rzye, 3)
            normalized["financing_change"] = round(rzye_change, 3)
            normalized["financing_net_buy"] = round(rzjme, 3)
            normalized["financing_signal"] = round(leverage_signal, 6)
            out.append(normalized)
        return out

    def _collect_generic_page_links(self, spec: PlatformSpec, query: str, limit_per_source: int) -> List[Dict[str, Any]]:
        resp = self._request_with_retry("GET", spec.base_url, headers=self.default_headers, referer=spec.base_url)
        html = resp.text
        anchors = re.findall(r"<a[^>]+href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", html, flags=re.IGNORECASE | re.DOTALL)

        out: List[Dict[str, Any]] = []
        for href, raw_title in anchors:
            title = _clean_html_text(raw_title)
            if len(title) < 8 or len(title) > 180:
                continue
            if query and query not in title and len(out) >= max(3, limit_per_source // 2):
                continue
            date_token = _extract_date_token(title) or _extract_date_token(_clean_html_text(href))
            publish_time = date_token or _now_utc().date().isoformat()
            final_url = href if href.startswith("http") else urljoin(spec.base_url, href)
            out.append(
                self._normalize_record(
                    spec=spec,
                    title=title,
                    content=title,
                    publish_time=publish_time,
                    url=final_url,
                )
            )
            if len(out) >= limit_per_source:
                break
        return out

    def _normalize_record(
        self,
        spec: PlatformSpec,
        title: str,
        content: str,
        publish_time: str,
        url: str,
    ) -> Dict[str, Any]:
        sentiment, sentiment_score = self._infer_sentiment(f"{title} {content}")
        event_type = self._infer_event_type(f"{title} {content}", spec.category)
        record = {
            "source_id": spec.source_id,
            "source": spec.source_name,
            "category": spec.category,
            "title": title[:220],
            "content": (content or title)[:1000],
            "publish_time": publish_time,
            "url": url or spec.base_url,
            "weight": round(_clip(spec.reliability, 0.05, 1.0), 4),
            "sentiment_type": sentiment,
            "sentiment_score": round(sentiment_score, 6),
            "event_type": event_type,
            "content_hash": hashlib.md5(f"{spec.source_id}|{title}|{publish_time}".encode("utf-8")).hexdigest(),
            "crawled_at": _now_utc().isoformat(),
        }
        return record

    def _infer_sentiment(self, text: str) -> Tuple[str, float]:
        content = (text or "").lower()
        neg_hits = sum(1 for token in self.NEGATIVE_TERMS if token and token in content)
        pos_hits = sum(1 for token in self.POSITIVE_TERMS if token and token in content)
        if neg_hits > pos_hits:
            score = _clip(0.5 + 0.08 * min(5, neg_hits) - 0.04 * min(3, pos_hits))
            return "negative", score
        if pos_hits > neg_hits:
            score = _clip(0.5 + 0.08 * min(5, pos_hits) - 0.04 * min(3, neg_hits))
            return "positive", score
        return "neutral", 0.5

    def _infer_event_type(self, text: str, category: str) -> str:
        content = (text or "").lower()
        for event_type, keywords in self.EVENT_TYPE_RULES:
            if any(token and token in content for token in keywords):
                return event_type
        return self.SOURCE_EVENT_TYPE.get(category, "generic_news")

    def _build_event_impacts(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scored: List[Dict[str, Any]] = []
        now = _now_utc()
        for row in records:
            sentiment = str(row.get("sentiment_type", "neutral"))
            if sentiment == "neutral":
                continue
            published = _parse_time(row.get("publish_time")) or now
            age_h = max(0.0, (now - published).total_seconds() / 3600.0)
            time_decay = max(0.10, math.exp(-age_h / 96.0))
            weight = _clip(_to_float(row.get("weight"), 0.5))
            base = 0.40 + 0.60 * weight

            text = f"{row.get('title', '')} {row.get('content', '')}".lower()
            severity_boost = 0.0
            severity_boost += 0.08 * sum(1 for token in self.HIGH_SEVERITY_TERMS if token in text)
            severity_boost = min(0.28, severity_boost)

            magnitude = _clip((base + severity_boost) * time_decay, 0.02, 1.0)
            impact_score = -magnitude if sentiment == "negative" else magnitude
            confidence = _clip(0.40 + 0.50 * weight + 0.10 * time_decay, 0.05, 1.0)
            scored.append(
                {
                    "event_type": row.get("event_type", "generic_news"),
                    "impact_score": round(impact_score, 6),
                    "confidence": round(confidence, 6),
                    "source": row.get("source", "unknown"),
                    "title": str(row.get("title", ""))[:140],
                }
            )

        scored.sort(key=lambda x: abs(_to_float(x.get("impact_score"), 0.0)), reverse=True)
        return scored[:32]

    def _build_macro_factors(self, records: List[Dict[str, Any]]) -> Dict[str, float]:
        if not records:
            return {
                "rate_stress": 0.5,
                "credit_stress": 0.5,
                "inflation_stress": 0.5,
                "fx_stress": 0.5,
                "vix_stress": 0.5,
                "liquidity_stress": 0.5,
                "policy_uncertainty": 0.5,
                "search_heat_stress": 0.5,
                "capital_flow_stress": 0.5,
                "financing_stress": 0.5,
                "macro_stress": 0.5,
            }

        total_weight = 0.0
        neg_weight = 0.0
        policy_weight = 0.0
        policy_neg = 0.0
        money_weight = 0.0
        money_neg = 0.0
        volatility_signals = 0.0
        credit_signals = 0.0
        inflation_signals = 0.0
        fx_signals = 0.0
        search_heat_weighted = 0.0
        search_heat_total_w = 0.0
        capital_flow_pressure = 0.0
        financing_pressure = 0.0

        for row in records:
            w = _clip(_to_float(row.get("weight"), 0.5), 0.01, 1.0)
            total_weight += w
            sentiment = str(row.get("sentiment_type", "neutral"))
            if sentiment == "negative":
                neg_weight += w

            category = str(row.get("category", ""))
            text = f"{row.get('title', '')} {row.get('content', '')}".lower()
            if category in {"central_bank", "macro_statistics", "official_disclosure", "exchange_disclosure"}:
                policy_weight += w
                if sentiment == "negative":
                    policy_neg += w
            if category in {"money_market", "central_bank"}:
                money_weight += w
                if sentiment == "negative":
                    money_neg += w

            if any(k in text for k in ("波动", "恐慌", "跳水", "震荡", "volatility", "selloff")):
                volatility_signals += w
            if any(k in text for k in ("违约", "信用", "债务", "违约风险", "default", "credit spread")):
                credit_signals += w
            if any(k in text for k in ("cpi", "ppi", "通胀", "物价", "inflation")):
                inflation_signals += w
            if any(k in text for k in ("汇率", "美元", "人民币", "外汇", "usd/cny", "fx")):
                fx_signals += w

            heat_score = _to_float(row.get("search_heat_score"), -1.0)
            if heat_score >= 0.0:
                search_heat_total_w += w
                search_heat_weighted += w * _clip(heat_score)

            if category == "capital_flow":
                flow_signal = _to_float(row.get("capital_flow_signal"), 0.0)
                capital_flow_pressure += w * _clip(-flow_signal, 0.0, 1.0)

            if category == "financing":
                financing_signal = _to_float(row.get("financing_signal"), 0.0)
                financing_pressure += w * _clip(financing_signal, 0.0, 1.0)

        total_weight = max(total_weight, 1e-6)
        neg_ratio = _clip(neg_weight / total_weight)
        policy_neg_ratio = _clip(policy_neg / max(policy_weight, 1e-6)) if policy_weight > 0 else 0.5
        money_neg_ratio = _clip(money_neg / max(money_weight, 1e-6)) if money_weight > 0 else 0.5
        search_heat_stress = _clip(search_heat_weighted / max(search_heat_total_w, 1e-6)) if search_heat_total_w > 0 else 0.5
        flow_stress = _clip(capital_flow_pressure / total_weight)
        financing_stress = _clip(financing_pressure / total_weight)

        rate_stress = _clip(0.35 + 0.45 * money_neg_ratio + 0.20 * neg_ratio + 0.12 * financing_stress)
        credit_stress = _clip(
            0.30 + 0.50 * (credit_signals / total_weight) + 0.25 * neg_ratio +
            0.25 * financing_stress + 0.18 * flow_stress
        )
        inflation_stress = _clip(0.25 + 0.60 * (inflation_signals / total_weight) + 0.15 * neg_ratio)
        fx_stress = _clip(0.25 + 0.60 * (fx_signals / total_weight) + 0.15 * neg_ratio)
        vix_stress = _clip(
            0.28 + 0.62 * (volatility_signals / total_weight) + 0.10 * neg_ratio +
            0.18 * search_heat_stress
        )
        liquidity_stress = _clip(
            0.30 + 0.55 * money_neg_ratio + 0.15 * (credit_signals / total_weight) +
            0.32 * flow_stress
        )
        policy_uncertainty = _clip(0.32 + 0.58 * policy_neg_ratio + 0.10 * neg_ratio + 0.10 * search_heat_stress)

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
            "search_heat_stress": round(search_heat_stress, 6),
            "capital_flow_stress": round(flow_stress, 6),
            "financing_stress": round(financing_stress, 6),
            "macro_stress": round(macro_stress, 6),
        }

    def _summary(
        self,
        records: List[Dict[str, Any]],
        source_status: Dict[str, Dict[str, Any]],
        event_impacts: List[Dict[str, Any]],
        macro_factors: Dict[str, float],
    ) -> Dict[str, Any]:
        by_source: Dict[str, int] = {}
        neg = 0
        pos = 0
        for row in records:
            source = str(row.get("source", "unknown"))
            by_source[source] = by_source.get(source, 0) + 1
            sentiment = row.get("sentiment_type")
            if sentiment == "negative":
                neg += 1
            elif sentiment == "positive":
                pos += 1

        available = sum(1 for v in source_status.values() if v.get("status") in {"ok", "cached", "cached_stale"})
        cached = sum(1 for v in source_status.values() if v.get("status") in {"cached", "cached_stale"})
        blocked = sum(1 for v in source_status.values() if v.get("status") == "error")
        auth_needed = sum(
            1 for v in source_status.values()
            if v.get("status") == "auth_required" or v.get("error") == "requires_auth_or_captcha"
        )

        return {
            "record_count": len(records),
            "source_count": len(by_source),
            "available_sources": available,
            "cached_sources": cached,
            "error_sources": blocked,
            "auth_required_sources": auth_needed,
            "negative_count": neg,
            "positive_count": pos,
            "top_sources": sorted(by_source.items(), key=lambda x: x[1], reverse=True)[:8],
            "event_impact_count": len(event_impacts),
            "macro_stress": round(_to_float(macro_factors.get("macro_stress"), 0.5), 6),
        }

    def _query_cache_dir(self, query: str) -> Path:
        query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()[:16]
        path = self.cache_dir / query_hash
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _persist_cache_snapshot(self, query: str, records: List[Dict[str, Any]]) -> None:
        if not query:
            return
        directory = self._query_cache_dir(query)
        ts = _now_utc().strftime("%Y%m%d_%H%M%S")
        payload = {
            "query": query,
            "created_at": _now_utc().isoformat(),
            "records": records,
        }
        (directory / f"{ts}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        # Keep latest 80 snapshots.
        snapshots = sorted(directory.glob("*.json"))
        if len(snapshots) > 80:
            for old in snapshots[:-80]:
                try:
                    old.unlink()
                except Exception:
                    pass

    def _load_history_from_cache(self, query: str, lookback_days: int) -> List[Dict[str, Any]]:
        directory = self._query_cache_dir(query)
        cutoff = _now_utc() - timedelta(days=max(1, lookback_days))
        out: List[Dict[str, Any]] = []
        for fp in sorted(directory.glob("*.json"), reverse=True):
            ts: Optional[datetime] = None
            try:
                ts = datetime.strptime(fp.stem, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
            except Exception:
                ts = _parse_time(fp.stem)
            if ts and ts < cutoff:
                continue
            try:
                payload = json.loads(fp.read_text(encoding="utf-8"))
                records = payload.get("records", [])
                if isinstance(records, list):
                    out.extend(records)
            except Exception:
                continue
        return out

    def _merge_and_dedup_records(self, groups: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        seen = set()
        merged: List[Dict[str, Any]] = []
        for rows in groups:
            for row in rows:
                source = str(row.get("source_id") or row.get("source") or "")
                title = str(row.get("title") or "")
                url = str(row.get("url") or "")
                key = hashlib.md5(f"{source}|{title}|{url}".encode("utf-8")).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                merged.append(row)
        merged.sort(
            key=lambda x: (
                _parse_time(x.get("publish_time")) or datetime.fromtimestamp(0, tz=timezone.utc),
                _to_float(x.get("weight"), 0.0),
            ),
            reverse=True,
        )
        return merged[:400]


china_alt_data_engine = ChinaAltDataEngine()
