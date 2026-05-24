from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio
import hashlib
import os
from dotenv import load_dotenv
load_dotenv()

from backend.engine.financial_action_trainer import AdvancedFinancialSentimentTrainer
from backend.engine.inference_engine import get_inference_and_prediction
from backend.engine.multi_model_coordinator import get_multi_model_prediction
from backend.engine.risk_output_enhancer import risk_output_enhancer
from collections import defaultdict
from backend.risk import risk_control_engine
from backend.risk.stress_testing import stress_testing_engine
from backend.risk.anomaly_detection import anomaly_detection_engine
from backend.risk.alternative_data import alternative_data_engine
from backend.alt_data import china_alt_data_engine
from backend.analysis.lexical_signal_linker import lexical_signal_linker
from backend.data.risk_data_provider import risk_data_provider, STOCK_CODE_MAPPING
from backend.data.stock_resolver import resolve_stock, search_stocks
from backend.crawler.runtime_adapters import (
    AsyncNewsCrawlerAdapter,
    AsyncSocialMediaCrawlerAdapter,
    RuntimeNewsAggregator,
)

app = FastAPI(title="Financial Risk Control Intelligence API", version="4.0.0")

# ==================== 数据模型 ====================

class StockQuery(BaseModel):
    stock_name: str
    risk_level: Optional[str] = "moderate"
    limit: Optional[int] = 10

class AnalysisRequest(BaseModel):
    texts: List[str]

class NewsQuery(BaseModel):
    query: str
    limit: Optional[int] = 10
    include_social: Optional[bool] = True


class RiskQuery(BaseModel):
    stock_name: str
    risk_level: Optional[str] = "moderate"
    limit: Optional[int] = 20
    include_social: Optional[bool] = True
    alt_lookback_days: Optional[int] = 30
    alt_limit_per_source: Optional[int] = 8
    enable_alt_data: Optional[bool] = True


class AltDataQuery(BaseModel):
    stock_name: str
    lookback_days: Optional[int] = 30
    limit_per_source: Optional[int] = 8
    include_realtime: Optional[bool] = True
    include_history: Optional[bool] = True


class PortfolioAsset(BaseModel):
    stock_name: str
    weight: float
    sector: Optional[str] = "unknown"
    style: Optional[str] = "unknown"
    leverage: Optional[float] = 1.0
    crowding: Optional[float] = 0.5
    returns: Optional[List[float]] = None
    predictions_7d: Optional[List[float]] = None
    expected_return: Optional[float] = 0.0
    volatility: Optional[float] = 0.02


class PortfolioRiskQuery(BaseModel):
    assets: List[PortfolioAsset]
    risk_level: Optional[str] = "moderate"
    covariance_matrix: Optional[List[List[float]]] = None
    gross_leverage: Optional[float] = None
    net_leverage: Optional[float] = None


class StressTestQuery(BaseModel):
    stock_name: Optional[str] = None
    portfolio_returns: Optional[List[float]] = None
    weights: Optional[List[float]] = None
    asset_sectors: Optional[List[str]] = None
    historical_scenarios: Optional[List[str]] = None
    shock_magnitude: Optional[float] = 0.20
    duration_days: Optional[int] = 10
    monte_carlo_sims: Optional[int] = 10000
    target_loss: Optional[float] = 0.20


class AnomalyDetectQuery(BaseModel):
    stock_name: str
    news_items: Optional[List[Dict[str, Any]]] = None
    market_data: Optional[Dict[str, Any]] = None
    limit: Optional[int] = 20


class AlternativeDataRiskQuery(BaseModel):
    stock_name: str
    data_types: Optional[List[str]] = None


class ApiKeyConfig(BaseModel):
    serpapi_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    qwen_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    llm_provider: Optional[str] = None

# ==================== 股票代码映射 ====================

stock_code_mapping = dict(STOCK_CODE_MAPPING)

# ==================== 全局组件初始化 ====================

# 情感分析训练器
try:
    trainer = AdvancedFinancialSentimentTrainer(use_ensemble=True)
    print("[OK] 高级金融情感分析训练器初始化成功")
    print(f"   已加载模型: {list(trainer.models.keys())}")
except Exception as e:
    print(f"[WARN] 训练器初始化失败: {e}")
    trainer = None

# 新闻爬虫和聚合器
news_crawler = AsyncNewsCrawlerAdapter()
social_crawler = AsyncSocialMediaCrawlerAdapter()
news_aggregator = RuntimeNewsAggregator()

# WebSocket连接管理
active_connections: List[WebSocket] = []
connection_lock = asyncio.Lock()

# 实时新闻推送任务（后台运行）
push_task = None
push_interval = max(5, int(os.getenv("RISK_PUSH_INTERVAL_SECONDS", "12")))  # 默认12秒，可通过环境变量调节
runtime_api_keys: Dict[str, str] = {}

# ==================== 辅助函数 ====================

async def fetch_multi_source_data(
    stock_name: str,
    limit: int = 10,
    include_social: bool = True,
    apply_limit: bool = True,
) -> List[Dict]:
    """
    从多个数据源抓取新闻数据
    
    参数:
        stock_name: 股票名称
        limit: 返回数量
        include_social: 是否包含社交媒体数据
    
    返回:
        新闻列表（已按权重排序）
    """
    external_crawl_enabled = (
        os.getenv("RISK_ENABLE_EXTERNAL_CRAWL", "false").lower() in {"1", "true", "yes", "on"}
        or _external_search_configured()
    )
    if not external_crawl_enabled:
        print("[INFO] 外部新闻抓取未启用，使用降级数据通道")
        return []

    all_news = []
    
    # 1. 抓取新闻数据
    try:
        news_results = await news_crawler.crawl(
            stock_name,
            limit,
            apply_final_limit=apply_limit,
        )
        all_news.extend(news_results)
        print(f"[NEWS] 新闻爬虫获取: {len(news_results)} 条")
    except Exception as e:
        print(f"[WARN] 新闻爬虫失败: {e}")
    
    # 2. 抓取社交媒体数据（可选）
    if include_social:
        try:
            social_limit = max(1, int(limit // 2))
            social_results = await social_crawler.crawl(stock_name, social_limit)
            all_news.extend(social_results)
            print(f"[SOCIAL] 社交媒体爬虫获取: {len(social_results)} 条")
        except Exception as e:
            print(f"[WARN] 社交媒体爬虫失败: {e}")
    
    # 3. 按权重排序，去重
    all_news.sort(key=lambda x: -x.get('weight', 0.5))
    
    # 去重（基于内容哈希）
    seen_hashes = set()
    unique_news = []
    for news in all_news:
        content_hash = news.get('content_hash', news.get('title', ''))
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_news.append(news)
    
    if apply_limit:
        return unique_news[: max(1, int(limit))]
    return unique_news


def _normalize_risk_profile(risk_profile: Optional[str]) -> str:
    candidate = (risk_profile or "moderate").lower().strip()
    return candidate if candidate in {"aggressive", "moderate", "conservative"} else "moderate"


def _resolve_include_social(include_social: Optional[bool]) -> bool:
    return True if include_social is None else include_social


async def _assess_with_validation(
    stock_name: str,
    risk_profile: Optional[str],
    limit: Optional[int],
    include_social: Optional[bool],
    alt_lookback_days: Optional[int] = 30,
    alt_limit_per_source: Optional[int] = 8,
    enable_alt_data: Optional[bool] = True,
) -> Dict[str, Any]:
    if not stock_name:
        raise HTTPException(status_code=400, detail="股票名称不能为空")
    return await _run_risk_assessment(
        stock_name=stock_name,
        risk_profile=risk_profile or "moderate",
        limit=limit or 20,
        include_social=_resolve_include_social(include_social),
        alt_lookback_days=max(1, int(alt_lookback_days or 30)),
        alt_limit_per_source=max(1, int(alt_limit_per_source or 8)),
        enable_alt_data=True if enable_alt_data is None else bool(enable_alt_data),
    )


def _legacy_action_from_risk_level(risk_level: str) -> Dict[str, str]:
    mapping = {
        "low": {"action": "低风险监控", "action_emoji": "[OK]"},
        "medium": {"action": "重点监控", "action_emoji": "[WARN]"},
        "high": {"action": "风险预警", "action_emoji": "[ALERT]"},
        "critical": {"action": "紧急处置", "action_emoji": "[STOP]"},
    }
    return mapping.get(risk_level, {"action": "风险预警", "action_emoji": "[ALERT]"})


def _has_configured_key(
    env_names: List[str],
    runtime_name: str,
    request: Optional[Request] = None,
    header_name: Optional[str] = None,
) -> bool:
    if runtime_api_keys.get(runtime_name):
        return True
    if any(os.getenv(name) for name in env_names):
        return True
    if request is not None and header_name:
        return bool(request.headers.get(header_name))
    return False


def _current_serpapi_key() -> str:
    return (
        runtime_api_keys.get("serpapi_api_key")
        or os.getenv("SERPAPI_API_KEY")
        or os.getenv("SERP_API_KEY")
        or ""
    ).strip()


def _current_tavily_key() -> str:
    return (
        runtime_api_keys.get("tavily_api_key")
        or os.getenv("TAVILY_API_KEY")
        or os.getenv("TRVILY_API_KEY")
        or os.getenv("TVLY_API_KEY")
        or ""
    ).strip()



def _normalize_llm_provider_name(raw: Optional[str]) -> str:
    provider = (raw or "auto").strip().lower().replace("-", "_")
    aliases = {
        "qwen": "qwen_api",
        "dashscope": "qwen_api",
        "aliyun_qwen": "qwen_api",
        "local": "local_qwen",
        "localqwen": "local_qwen",
        "deep_seek": "deepseek",
        "none": "rules",
        "fallback": "rules",
    }
    provider = aliases.get(provider, provider)
    return provider if provider in {"auto", "local_qwen", "qwen_api", "deepseek", "rules"} else "auto"


def _current_qwen_api_key() -> str:
    return (
        runtime_api_keys.get("qwen_api_key")
        or os.getenv("QWEN_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or ""
    ).strip()


def _current_deepseek_key() -> str:
    return (
        runtime_api_keys.get("deepseek_api_key")
        or os.getenv("DEEPSEEK_API_KEY")
        or ""
    ).strip()


def _current_llm_provider() -> str:
    return _normalize_llm_provider_name(runtime_api_keys.get("llm_provider") or os.getenv("LLM_PROVIDER") or "auto")
def _external_search_configured() -> bool:
    return bool(_current_serpapi_key() or _current_tavily_key())


def _clear_search_usage_cache(crawler: Any) -> None:
    cache = getattr(crawler, "_usage_cache", None)
    if isinstance(cache, dict):
        cache["serpapi"] = {"ts": 0.0, "payload": {}}
        cache["tavily"] = {"ts": 0.0, "payload": {}}


def _apply_runtime_api_keys(
    serpapi_api_key: Optional[str] = None,
    tavily_api_key: Optional[str] = None,
    qwen_api_key: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,
    llm_provider: Optional[str] = None,
) -> Dict[str, bool]:
    """Apply user-provided API keys to env and already-created runtime instances."""
    serpapi_key = (serpapi_api_key or "").strip()
    tavily_key = (tavily_api_key or "").strip()
    qwen_key = (qwen_api_key or "").strip()
    deepseek_key = (deepseek_api_key or "").strip()
    provider = _normalize_llm_provider_name(llm_provider) if llm_provider is not None else None

    if serpapi_key:
        runtime_api_keys["serpapi_api_key"] = serpapi_key
        os.environ["SERPAPI_API_KEY"] = serpapi_key
        os.environ["SERP_API_KEY"] = serpapi_key
    if tavily_key:
        runtime_api_keys["tavily_api_key"] = tavily_key
        os.environ["TAVILY_API_KEY"] = tavily_key
        os.environ["TRVILY_API_KEY"] = tavily_key
    if qwen_key:
        runtime_api_keys["qwen_api_key"] = qwen_key
        os.environ["QWEN_API_KEY"] = qwen_key
        os.environ["DASHSCOPE_API_KEY"] = qwen_key
    if deepseek_key:
        runtime_api_keys["deepseek_api_key"] = deepseek_key
        os.environ["DEEPSEEK_API_KEY"] = deepseek_key
    if provider:
        runtime_api_keys["llm_provider"] = provider
        os.environ["LLM_PROVIDER"] = provider

    active_serpapi_key = _current_serpapi_key()
    active_tavily_key = _current_tavily_key()

    # NewsCrawler reads keys at construction time, so update live instances too.
    for attr_name in ("_core_crawler", "_fallback_crawler"):
        crawler = getattr(news_crawler, attr_name, None)
        if crawler is None:
            continue
        changed = False
        if active_serpapi_key and getattr(crawler, "serpapi_api_key", "") != active_serpapi_key:
            setattr(crawler, "serpapi_api_key", active_serpapi_key)
            changed = True
        if active_tavily_key and getattr(crawler, "tavily_api_key", "") != active_tavily_key:
            setattr(crawler, "tavily_api_key", active_tavily_key)
            changed = True
        if changed:
            _clear_search_usage_cache(crawler)

    return {
        "serpapi_configured": bool(active_serpapi_key),
        "tavily_configured": bool(active_tavily_key),
        "qwen_api_configured": bool(_current_qwen_api_key()),
        "deepseek_configured": bool(_current_deepseek_key()),
        "llm_provider": _current_llm_provider(),
    }


@app.middleware("http")
async def runtime_api_key_middleware(request: Request, call_next):
    """Allow frontend-provided API keys to power backend search and LLM reports during this local session."""
    serpapi_key = (request.headers.get("X-SerpAPI-Key") or "").strip()
    tavily_key = (request.headers.get("X-Tavily-API-Key") or "").strip()
    qwen_key = (request.headers.get("X-Qwen-API-Key") or "").strip()
    deepseek_key = (request.headers.get("X-DeepSeek-API-Key") or "").strip()
    llm_provider = (request.headers.get("X-LLM-Provider") or "").strip() or None
    if serpapi_key or tavily_key or qwen_key or deepseek_key or llm_provider:
        _apply_runtime_api_keys(serpapi_key, tavily_key, qwen_key, deepseek_key, llm_provider)
    return await call_next(request)



def _narrative_meta_from_enhancement(enhancement: Dict[str, Any]) -> Dict[str, Any]:
    narrative = enhancement.get("narrative") or {}
    return {
        "provider": narrative.get("provider") or "rules",
        "used_llm": bool(narrative.get("used_llm") or narrative.get("used_qwen") or narrative.get("used_qwen_api") or narrative.get("used_deepseek")),
        "used_qwen": bool(narrative.get("used_qwen", False)),
        "used_qwen_api": bool(narrative.get("used_qwen_api", False)),
        "used_deepseek": bool(narrative.get("used_deepseek", False)),
        "llm_provider_requested": _current_llm_provider(),
        "error": narrative.get("error"),
    }
def _build_sentiment_snapshot(news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(news_items)
    if total == 0:
        return {
            "total_news": 0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "negative_ratio": 0.0,
            "sentiment_score": 0.0,
        }

    positive_count = sum(1 for n in news_items if n.get("sentiment_type") == "positive")
    negative_count = sum(1 for n in news_items if n.get("sentiment_type") == "negative")
    neutral_count = total - positive_count - negative_count
    sentiment_score = (positive_count - negative_count) / max(total, 1)
    negative_ratio = negative_count / max(total, 1)

    return {
        "total_news": total,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "negative_ratio": round(negative_ratio, 4),
        "sentiment_score": round(sentiment_score, 4),
    }


def _build_signal_snapshot(news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    source_type_count: Dict[str, int] = defaultdict(int)
    signal_type_count: Dict[str, int] = defaultdict(int)
    category_count: Dict[str, int] = defaultdict(int)

    for item in news_items or []:
        source_type = str(item.get("data_source_type", "news") or "news").strip()
        signal_type = str(item.get("signal_type", "") or "").strip()
        category = str(item.get("category", "general") or "general").strip()
        source_type_count[source_type] += 1
        category_count[category] += 1
        if signal_type:
            signal_type_count[signal_type] += 1

    return {
        "source_type_count": dict(source_type_count),
        "signal_type_count": dict(signal_type_count),
        "category_count": dict(category_count),
    }


def _merge_news_items(
    primary: List[Dict[str, Any]],
    secondary: List[Dict[str, Any]],
    limit: Optional[int] = 80,
) -> List[Dict[str, Any]]:
    def _safe_weight(row: Dict[str, Any]) -> float:
        try:
            return float(row.get("weight", 0.5))
        except (TypeError, ValueError):
            return 0.5

    merged: List[Dict[str, Any]] = []
    seen = set()
    for row in (primary or []) + (secondary or []):
        digest = row.get("content_hash")
        if not digest:
            raw = f"{row.get('source', '')}|{row.get('title', '')}|{row.get('url', '')}"
            digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        merged.append(row)
    merged.sort(
        key=lambda x: (
            _safe_weight(x),
            str(x.get("publish_time", "")),
        ),
        reverse=True,
    )
    if limit is None:
        return merged
    return merged[:max(1, int(limit))]


def _merge_event_impacts(base: Dict[str, Any], alt_event_impacts: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(base or {})
    events = list(out.get("event_impacts", []) or [])
    events.extend(alt_event_impacts or [])
    events.sort(
        key=lambda x: abs(float(x.get("impact_score", 0.0))) * max(0.01, float(x.get("confidence", 0.5))),
        reverse=True,
    )
    out["event_impacts"] = events[:64]
    summary = dict(out.get("summary", {}) or {})
    summary["alt_event_count"] = len(alt_event_impacts or [])
    summary["key_events_count"] = len(out["event_impacts"])
    out["summary"] = summary
    return out


def _merge_macro_factors(base: Dict[str, Any], alt_macro: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    if not alt_macro:
        return out
    existing_macro = dict(out.get("macro_factors", {}) or {})
    if not existing_macro:
        existing_macro = dict(out.get("market_environment", {}) or {})

    merged_macro: Dict[str, float] = {}
    keys = set(existing_macro.keys()) | set(alt_macro.keys())
    for key in keys:
        old_val = existing_macro.get(key)
        new_val = alt_macro.get(key)
        try:
            old_num = None if old_val is None else float(old_val)
        except (TypeError, ValueError):
            old_num = None
        try:
            new_num = None if new_val is None else float(new_val)
        except (TypeError, ValueError):
            new_num = None
        if old_num is None:
            if new_num is not None:
                merged_macro[key] = new_num
        elif new_num is None:
            merged_macro[key] = old_num
        else:
            merged_macro[key] = 0.35 * old_num + 0.65 * new_num

    out["macro_factors"] = merged_macro
    market_environment = dict(out.get("market_environment", {}) or {})
    market_environment.update(merged_macro)
    out["market_environment"] = market_environment
    return out


async def _fetch_real_market_data(stock_name: str, timeout_seconds: float = 5.0) -> Dict[str, Any]:
    """并行抓取真实行情/财务/市场数据，失败降级为空结构。"""

    async def _run() -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "kline": {},
            "financial": {},
            "market": {},
            "real_returns": [],
        }

        try:
            closes, highs, lows, volumes, returns, dates = await asyncio.to_thread(
                risk_data_provider.fetch_kline,
                stock_name,
                180,
            )
            result["kline"] = {
                "closes": closes,
                "highs": highs,
                "lows": lows,
                "volumes": volumes,
                "returns": returns,
                "dates": dates,
            }
            result["real_returns"] = returns or []
        except Exception as exc:
            result["kline_error"] = str(exc)

        try:
            result["financial"] = await asyncio.to_thread(
                risk_data_provider.fetch_financial_data,
                stock_name,
            ) or {}
        except Exception as exc:
            result["financial_error"] = str(exc)

        try:
            result["market"] = await asyncio.to_thread(
                risk_data_provider.fetch_market_data,
                stock_name,
                result.get("financial", {}),
            ) or {}
        except Exception as exc:
            result["market_error"] = str(exc)

        return result

    return await asyncio.wait_for(_run(), timeout=max(1.0, float(timeout_seconds)))


async def _run_risk_assessment(
    stock_name: str,
    risk_profile: str = "moderate",
    limit: int = 20,
    include_social: bool = True,
    alt_lookback_days: int = 30,
    alt_limit_per_source: int = 8,
    enable_alt_data: bool = True,
) -> Dict[str, Any]:
    profile = _normalize_risk_profile(risk_profile)
    resolution = resolve_stock(stock_name)
    resolved_name = str(resolution.get("name") or stock_name).strip() or stock_name
    resolved_code = str(resolution.get("code") or stock_name).strip() or stock_name
    if resolved_name and resolved_code:
        stock_code_mapping[resolved_name] = resolved_code

    lookup_name = resolved_name
    news_timeout_seconds = max(5.0, float(os.getenv("RISK_NEWS_TIMEOUT_SECONDS", "15")))
    real_data_timeout_seconds = max(2.0, float(os.getenv("RISK_REAL_DATA_TIMEOUT_SECONDS", "5")))

    real_data_task = asyncio.create_task(
        _fetch_real_market_data(lookup_name, timeout_seconds=real_data_timeout_seconds)
    )

    # 决策链路不做“固定条数截断”，仅依赖去重+排序+相关性筛选。
    news_task = asyncio.create_task(
        fetch_multi_source_data(
            lookup_name,
            limit,
            include_social,
            apply_limit=False,
        )
    )
    alt_task = None
    if enable_alt_data:
        alt_task = asyncio.create_task(
            china_alt_data_engine.collect_signals(
                query=lookup_name,
                lookback_days=max(1, int(alt_lookback_days or 30)),
                limit_per_source=max(1, int(alt_limit_per_source or 8)),
                include_realtime=True,
                include_history=True,
            )
        )

    news_timeout_error = None
    try:
        news_payload = await asyncio.wait_for(news_task, timeout=news_timeout_seconds)
    except Exception as exc:
        news_timeout_error = str(exc)
        news_payload = exc

    if alt_task is not None:
        alt_result_list = await asyncio.gather(alt_task, return_exceptions=True)
        alt_payload = alt_result_list[0] if alt_result_list else {}
    else:
        alt_payload = {}

    real_data_error = None
    try:
        real_data_payload = await real_data_task
    except Exception as exc:
        real_data_payload = {}
        real_data_error = str(exc)

    news_items: List[Dict[str, Any]] = [] if isinstance(news_payload, Exception) else list(news_payload or [])
    alt_bundle: Dict[str, Any] = {} if isinstance(alt_payload, Exception) else dict(alt_payload or {})
    alt_error = str(alt_payload) if isinstance(alt_payload, Exception) else None
    if news_timeout_error and not news_items:
        alt_error = alt_error or f"news_timeout_or_error={news_timeout_error}"

    alt_news = alt_bundle.get("news_items", []) if isinstance(alt_bundle, dict) else []
    if alt_news:
        news_items = _merge_news_items(news_items, alt_news, limit=None)

    # Algorithmic lexical linker to replace embedding retrieval dependency:
    # alias dictionary + BM25 + TF-IDF char ngram + SimHash dedup clustering.
    lexical_meta: Dict[str, Any] = {}
    if news_items:
        linked, lexical_meta = lexical_signal_linker.rank_records(
            lookup_name,
            news_items,
            top_k=max(1, len(news_items)),
            min_score=0.0,
        )
        if linked:
            news_items = linked

    if not news_items:
        assessment = risk_control_engine.build_empty_assessment(lookup_name, profile)
        assessment["input_stock_name"] = stock_name
        assessment["stock_code"] = resolved_code
        assessment["resolution"] = resolution
        assessment["diagnostics"] = {
            "inference_ready": False,
            "multi_model_ready": False,
            "inference_error": "No news data available",
            "multi_model_error": None,
            "news_snapshot": _build_sentiment_snapshot(news_items),
            "signal_snapshot": _build_signal_snapshot(news_items),
            "top_news": [],
            "lexical_linker": lexical_meta,
            "alt_data": {
                "enabled": enable_alt_data,
                "error": alt_error,
                "summary": (alt_bundle or {}).get("summary", {}),
                "anti_crawl": (alt_bundle or {}).get("anti_crawl", {}),
            },
            "real_data": {
                "enabled": True,
                "error": real_data_error,
                "kline_count": len((real_data_payload or {}).get("real_returns", []) or []),
                "financial_fields": len((real_data_payload or {}).get("financial", {}) or {}),
                "market_fields": len((real_data_payload or {}).get("market", {}) or {}),
            },
            "qwen_coordination": {
                "mode": "skipped_no_news",
                "algorithmic": {},
                "qwen": {"used_qwen": False, "error": "no_news"},
            },
            "news_timeout_seconds": news_timeout_seconds,
            "news_timeout_error": news_timeout_error,
        }
        enhancement = await risk_output_enhancer.enhance(lookup_name, assessment, news_items)
        assessment["risk_score_fused"] = enhancement.get("risk_score_fused", assessment.get("risk_score"))
        assessment["risk_level_fused"] = enhancement.get("risk_level_fused", assessment.get("risk_level"))
        assessment["model_fusion"] = {
            "risk_sklearn": enhancement.get("risk_sklearn", {}),
            "risk_sklearn_applied": enhancement.get("risk_sklearn_applied", False),
            "risk_sklearn_override_enabled": enhancement.get("risk_sklearn_override_enabled", False),
            "risk_score_base": enhancement.get("risk_score_base", assessment.get("risk_score")),
            "risk_score_fused": enhancement.get("risk_score_fused", assessment.get("risk_score")),
            "risk_score_shadow": enhancement.get("risk_score_shadow", assessment.get("risk_score")),
            "risk_level_shadow": enhancement.get("risk_level_shadow", assessment.get("risk_level")),
        }
        assessment["narrative"] = (enhancement.get("narrative") or {}).get("text", "")
        assessment["narrative_meta"] = _narrative_meta_from_enhancement(enhancement)

        return assessment

    inference_task = asyncio.create_task(get_inference_and_prediction(lookup_name, news_items, profile))
    multi_model_task = asyncio.create_task(get_multi_model_prediction(lookup_name))

    inference_result: Dict[str, Any] = {}
    multi_model_result: Dict[str, Any] = {}
    inference_error = None
    multi_model_error = None

    inference_payload, multi_model_payload = await asyncio.gather(
        inference_task,
        multi_model_task,
        return_exceptions=True,
    )

    if isinstance(inference_payload, Exception):
        inference_error = str(inference_payload)
    else:
        inference_result = inference_payload

    if isinstance(multi_model_payload, Exception):
        multi_model_error = str(multi_model_payload)
    else:
        multi_model_result = multi_model_payload

    if real_data_payload:
        if real_data_payload.get("real_returns"):
            multi_model_result["real_returns"] = list(real_data_payload.get("real_returns") or [])
        if real_data_payload.get("financial"):
            multi_model_result["financial_data"] = dict(real_data_payload.get("financial") or {})
        if real_data_payload.get("market"):
            market_data = dict(real_data_payload.get("market") or {})
            if "stock_returns" not in market_data and real_data_payload.get("real_returns"):
                market_data["stock_returns"] = list(real_data_payload.get("real_returns") or [])
            multi_model_result["market_data"] = market_data

    if alt_bundle:
        inference_result = _merge_event_impacts(inference_result, alt_bundle.get("event_impacts", []) or [])
        multi_model_result = _merge_macro_factors(multi_model_result, alt_bundle.get("macro_factors", {}) or {})

    coordination_bundle = await risk_output_enhancer.coordinate_chain(
        stock_name=lookup_name,
        news_items=news_items,
        inference_result=inference_result,
        multi_model_result=multi_model_result,
    )
    inference_result = coordination_bundle.get("inference_result", inference_result) or {}
    multi_model_result = coordination_bundle.get("multi_model_result", multi_model_result) or {}
    coordination_meta = coordination_bundle.get("coordination_meta", {}) or {}

    assessment = risk_control_engine.assess(
        stock_name=lookup_name,
        risk_profile=profile,
        news_items=news_items,
        inference_result=inference_result,
        multi_model_result=multi_model_result,
    )
    assessment["input_stock_name"] = stock_name
    assessment["stock_code"] = resolved_code
    assessment["resolution"] = resolution

    assessment["diagnostics"] = {
        "inference_ready": bool(inference_result),
        "multi_model_ready": bool(multi_model_result),
        "inference_error": inference_error,
        "multi_model_error": multi_model_error,
        "news_snapshot": _build_sentiment_snapshot(news_items),
        "signal_snapshot": _build_signal_snapshot(news_items),
        "top_news": [{
            "title": n.get("title", ""),
            "source": n.get("source", ""),
            "weight": round(n.get("weight", 0.0), 3),
            "sentiment": n.get("sentiment_type", "neutral"),
            "lexical_link_score": round(float(n.get("lexical_link_score", 0.0) or 0.0), 4),
        } for n in news_items[:5]],
        "lexical_linker": lexical_meta,
        "alt_data": {
            "enabled": enable_alt_data,
            "error": alt_error,
            "summary": (alt_bundle or {}).get("summary", {}),
            "source_status": (alt_bundle or {}).get("source_status", {}),
            "anti_crawl": (alt_bundle or {}).get("anti_crawl", {}),
        },
        "real_data": {
            "enabled": True,
            "error": real_data_error,
            "kline_count": len((real_data_payload or {}).get("real_returns", []) or []),
            "financial_fields": len((real_data_payload or {}).get("financial", {}) or {}),
            "market_fields": len((real_data_payload or {}).get("market", {}) or {}),
        },
        "qwen_coordination": coordination_meta,
        "news_timeout_seconds": news_timeout_seconds,
        "news_timeout_error": news_timeout_error,
    }
    assessment["coordination_meta"] = coordination_meta
    enhancement = await risk_output_enhancer.enhance(lookup_name, assessment, news_items)
    assessment["risk_score_fused"] = enhancement.get("risk_score_fused", assessment.get("risk_score"))
    assessment["risk_level_fused"] = enhancement.get("risk_level_fused", assessment.get("risk_level"))
    assessment["model_fusion"] = {
        "risk_sklearn": enhancement.get("risk_sklearn", {}),
        "risk_sklearn_applied": enhancement.get("risk_sklearn_applied", False),
        "risk_sklearn_override_enabled": enhancement.get("risk_sklearn_override_enabled", False),
        "risk_score_base": enhancement.get("risk_score_base", assessment.get("risk_score")),
        "risk_score_fused": enhancement.get("risk_score_fused", assessment.get("risk_score")),
        "risk_score_shadow": enhancement.get("risk_score_shadow", assessment.get("risk_score")),
        "risk_level_shadow": enhancement.get("risk_level_shadow", assessment.get("risk_level")),
    }
    assessment["narrative"] = (enhancement.get("narrative") or {}).get("text", "")
    assessment["narrative_meta"] = {
        "used_qwen": bool((enhancement.get("narrative") or {}).get("used_qwen", False)),
        "error": (enhancement.get("narrative") or {}).get("error"),
    }
    return assessment

# ==================== 定时推送任务 ====================

async def real_time_push_task():
    """实时新闻推送任务"""
    monitored_stocks = list(stock_code_mapping.keys())[:5]  # 监控前5只股票
    last_top_hash = defaultdict(str)
    
    while True:
        try:
            # 检查每只监控股票的新闻更新
            for stock in monitored_stocks:
                news_items = await news_crawler.crawl(stock, limit=5)
                if not news_items:
                    continue

                latest_hash = news_items[0].get("content_hash", "")
                previous_hash = last_top_hash.get(stock, "")
                if latest_hash and previous_hash and latest_hash != previous_hash:
                    await broadcast_news(stock, news_items[:2])
                if latest_hash:
                    last_top_hash[stock] = latest_hash
            
            await asyncio.sleep(push_interval)
        except Exception as e:
            print(f"[WARN] 推送任务异常: {e}")
            await asyncio.sleep(60)

async def broadcast_news(stock_name: str, news_items: List[Dict]):
    """广播新闻给所有连接的客户端"""
    async with connection_lock:
        message = {
            "type": "news_alert",
            "stock_name": stock_name,
            "count": len(news_items),
            "news": news_items,
            "timestamp": datetime.now().isoformat()
        }

        dead_connections = []
        for connection in active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)

        for dead in dead_connections:
            if dead in active_connections:
                active_connections.remove(dead)

# ==================== API路由 ====================

@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    # 添加爬虫到聚合器
    await news_aggregator.add_crawler(news_crawler)
    await news_aggregator.add_crawler(social_crawler)
    
    # 启动实时推送任务
    global push_task
    push_task = asyncio.create_task(real_time_push_task())
    print("[START] 实时推送任务已启动")

@app.on_event("shutdown")
async def shutdown_event():
    """关闭时清理"""
    # 停止推送任务
    if push_task:
        push_task.cancel()
        try:
            await push_task
        except asyncio.CancelledError:
            pass
    
    # 关闭爬虫资源
    await news_crawler.close()
    await social_crawler.close()
    print("[OK] 资源已清理")

@app.post("/api/action")
async def get_action_recommendation(query: StockQuery):
    """
    获取风控动作建议 - 兼容原有接口
    
    参数:
        query: StockQuery对象
            - stock_name: 股票名称
            - risk_level: 风险偏好 (aggressive/moderate/conservative)
            - limit: 新闻数量限制
    
    返回:
        包含风险评分、风险等级、行动建议与关键告警
    """
    try:
        stock_name = query.stock_name
        assessment = await _assess_with_validation(
            stock_name=stock_name,
            risk_profile=query.risk_level,
            limit=query.limit,
            include_social=True,
        )

        display_score = assessment.get("risk_score_fused", assessment.get("risk_score", 50.0))
        display_level = assessment.get("risk_level_fused", assessment.get("risk_level", "high"))
        legacy_action = _legacy_action_from_risk_level(display_level)
        diagnostics = assessment.get("diagnostics", {})
        sentiment_snapshot = diagnostics.get("news_snapshot", {})
        factor_breakdown = assessment.get("factor_breakdown", {})
        trend_factor = factor_breakdown.get("trend", {})
        volatility_factor = factor_breakdown.get("volatility", {})
        uncertainty_factor = factor_breakdown.get("uncertainty", {})

        return {
            "stock_name": stock_name,
            "stock_code": stock_code_mapping.get(stock_name, "unknown"),
            "action": legacy_action["action"],
            "action_emoji": legacy_action["action_emoji"],
            "risk_score": assessment.get("risk_score", 50.0),
            "risk_score_fused": display_score,
            "risk_level": display_level,
            "risk_level_fused": display_level,
            "risk_label": assessment.get("risk_label", "HIGH"),
            "confidence_percent": round((100.0 - float(display_score)), 2),
            "sentiment_summary": f"新闻{sentiment_snapshot.get('total_news', 0)}条，负面占比{sentiment_snapshot.get('negative_ratio', 0.0):.2%}",
            "why": f"主要风险驱动: {', '.join(assessment.get('top_risk_drivers', [])[:3]) or 'n/a'}",
            "next_steps": assessment.get("control_actions", {}).get("next_steps", []),
            "data_summary": sentiment_snapshot,
            "signal_summary": diagnostics.get("signal_snapshot", {}),
            "top_news": diagnostics.get("top_news", []),
            "risk_controls": assessment.get("control_actions", {}),
            "risk_factors": factor_breakdown,
            "alerts": assessment.get("alerts", []),
            "quantile_risk": assessment.get("quantile_risk", {}),
            "dynamic_thresholds": assessment.get("dynamic_thresholds", {}),
            "escalation_strategy": assessment.get("escalation_strategy", {}),
            "market_state_engine": assessment.get("market_state_engine", {}),
            "model_fusion": assessment.get("model_fusion", {}),
            "narrative": assessment.get("narrative", ""),
            "narrative_meta": assessment.get("narrative_meta", {}),
            "trend_risk_score": trend_factor.get("score", 50.0),
            "volatility_risk_score": volatility_factor.get("score", 50.0),
            "uncertainty_risk_score": uncertainty_factor.get("score", 50.0),
            "diagnostics": {
                "inference_ready": diagnostics.get("inference_ready", False),
                "multi_model_ready": diagnostics.get("multi_model_ready", False),
                "inference_error": diagnostics.get("inference_error"),
                "multi_model_error": diagnostics.get("multi_model_error"),
                "signal_snapshot": diagnostics.get("signal_snapshot", {}),
                "alt_data": diagnostics.get("alt_data", {}),
            },
            "timestamp": datetime.now().isoformat(),
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/risk/assess")
async def risk_assess(query: RiskQuery):
    """核心风控评估接口。"""
    try:
        assessment = await _assess_with_validation(
            stock_name=query.stock_name,
            risk_profile=query.risk_level,
            limit=query.limit,
            include_social=query.include_social,
            alt_lookback_days=query.alt_lookback_days,
            alt_limit_per_source=query.alt_limit_per_source,
            enable_alt_data=query.enable_alt_data,
        )
        return assessment
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/risk/portfolio")
async def risk_portfolio_assess(query: PortfolioRiskQuery):
    """组合风控评估接口（多标的、协方差与暴露维度）。"""
    try:
        if not query.assets:
            raise HTTPException(status_code=400, detail="资产列表不能为空")

        assessment = risk_control_engine.assess_portfolio(
            assets=[asset.dict() for asset in query.assets],
            risk_profile=query.risk_level or "moderate",
            covariance_matrix=query.covariance_matrix,
            gross_leverage=query.gross_leverage,
            net_leverage=query.net_leverage,
        )
        return assessment
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/risk/alerts")
async def risk_alerts(query: RiskQuery):
    """返回当前风险告警视图。"""
    try:
        assessment = await _assess_with_validation(
            stock_name=query.stock_name,
            risk_profile=query.risk_level,
            limit=query.limit,
            include_social=query.include_social,
            alt_lookback_days=query.alt_lookback_days,
            alt_limit_per_source=query.alt_limit_per_source,
            enable_alt_data=query.enable_alt_data,
        )

        return {
            "stock_name": assessment.get("stock_name"),
            "risk_score": assessment.get("risk_score"),
            "risk_score_fused": assessment.get("risk_score_fused", assessment.get("risk_score")),
            "risk_level": assessment.get("risk_level"),
            "risk_level_fused": assessment.get("risk_level_fused", assessment.get("risk_level")),
            "alert_count": len(assessment.get("alerts", [])),
            "alerts": assessment.get("alerts", []),
            "escalation_strategy": assessment.get("escalation_strategy", {}),
            "quantile_risk": assessment.get("quantile_risk", {}),
            "dynamic_thresholds": assessment.get("dynamic_thresholds", {}),
            "top_risk_drivers": assessment.get("top_risk_drivers", []),
            "model_fusion": assessment.get("model_fusion", {}),
            "narrative": assessment.get("narrative", ""),
            "narrative_meta": assessment.get("narrative_meta", {}),
            "signal_snapshot": (assessment.get("diagnostics", {}) or {}).get("signal_snapshot", {}),
            "alt_data": (assessment.get("diagnostics", {}) or {}).get("alt_data", {}),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/risk/report/{stock_name}")
async def risk_report(stock_name: str, risk_level: Optional[str] = "moderate", limit: Optional[int] = 20):
    """返回结构化风险报告。"""
    try:
        assessment = await _assess_with_validation(
            stock_name=stock_name,
            risk_profile=risk_level,
            limit=limit,
            include_social=True,
        )

        diagnostics = assessment.get("diagnostics", {})
        return {
            "stock_name": stock_name,
            "report_type": "risk_control_report",
            "risk_profile": assessment.get("risk_profile"),
            "risk_score": assessment.get("risk_score"),
            "risk_score_fused": assessment.get("risk_score_fused", assessment.get("risk_score")),
            "risk_level": assessment.get("risk_level"),
            "risk_level_fused": assessment.get("risk_level_fused", assessment.get("risk_level")),
            "risk_label": assessment.get("risk_label"),
            "risk_method": assessment.get("risk_method"),
            "market_state_engine": assessment.get("market_state_engine", {}),
            "quantile_risk": assessment.get("quantile_risk", {}),
            "dynamic_thresholds": assessment.get("dynamic_thresholds", {}),
            "factor_breakdown": assessment.get("factor_breakdown", {}),
            "top_risk_drivers": assessment.get("top_risk_drivers", []),
            "alerts": assessment.get("alerts", []),
            "escalation_strategy": assessment.get("escalation_strategy", {}),
            "control_actions": assessment.get("control_actions", {}),
            "data_quality": assessment.get("data_quality", {}),
            "model_fusion": assessment.get("model_fusion", {}),
            "narrative": assessment.get("narrative", ""),
            "narrative_meta": assessment.get("narrative_meta", {}),
            "news_snapshot": diagnostics.get("news_snapshot", {}),
            "signal_snapshot": diagnostics.get("signal_snapshot", {}),
            "top_news": diagnostics.get("top_news", []),
            "alt_data": diagnostics.get("alt_data", {}),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/risk/stress-test")
async def risk_stress_test(query: StressTestQuery):
    """压力测试接口：支持直接传收益率，或基于股票名拉取历史收益。"""
    try:
        returns: List[float] = []
        source = "portfolio_returns"
        stock_resolution: Dict[str, Any] = {}

        if query.portfolio_returns:
            returns = [float(r) for r in query.portfolio_returns if r is not None]
        elif query.stock_name:
            stock_resolution = resolve_stock(query.stock_name)
            lookup_name = str(stock_resolution.get("name") or query.stock_name).strip() or query.stock_name
            _, _, _, _, real_returns, _ = await asyncio.to_thread(
                risk_data_provider.fetch_kline,
                lookup_name,
                240,
            )
            returns = [float(r) for r in (real_returns or []) if r is not None]
            source = "real_kline"
            if len(returns) < 10:
                raise HTTPException(
                    status_code=404,
                    detail=f"无法获取 {query.stock_name} 的历史行情数据",
                )
        else:
            raise HTTPException(status_code=400, detail="请提供 portfolio_returns 或 stock_name")

        shock = abs(float(query.shock_magnitude or 0.20))
        duration_days = max(1, int(query.duration_days or 10))
        hypothetical = [{
            "name": "custom_user_shock",
            "market_shock": -shock,
            "vol_multiplier": max(1.2, 1.0 + shock * 3.0),
            "correlation_shift": min(0.85, 0.10 + shock),
            "liquidity_shrink": min(0.90, 0.15 + shock),
            "duration_days": duration_days,
        }]

        reverse_targets = [-abs(float(query.target_loss))] if query.target_loss is not None else None
        result = stress_testing_engine.run_full_stress_test(
            portfolio_returns=returns,
            weights=query.weights,
            asset_sectors=query.asset_sectors,
            historical_scenarios=query.historical_scenarios,
            hypothetical_scenarios=hypothetical,
            monte_carlo_sims=max(1000, int(query.monte_carlo_sims or 10000)),
            horizon_days=max(7, duration_days),
            reverse_targets=reverse_targets,
        )

        return {
            "stock_name": query.stock_name,
            "source": source,
            "resolution": stock_resolution,
            "input_summary": {
                "return_count": len(returns),
                "weights_count": len(query.weights or []),
                "asset_sectors_count": len(query.asset_sectors or []),
            },
            "stress_test": result,
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/risk/scenarios")
async def get_stress_scenarios():
    """获取可用历史压力测试情景列表。"""
    try:
        scenarios = stress_testing_engine.get_available_scenarios()
        return {
            "scenarios": scenarios,
            "count": len(scenarios),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/risk/anomaly-detect")
async def anomaly_detect(query: AnomalyDetectQuery):
    """异常交易与合规风险检测。"""
    try:
        resolution = resolve_stock(query.stock_name)
        lookup_name = str(resolution.get("name") or query.stock_name).strip() or query.stock_name

        news_items = query.news_items or []
        if not news_items:
            news_items = await fetch_multi_source_data(
                lookup_name,
                query.limit or 20,
                include_social=True,
            )

        news_text_parts: List[str] = []
        for row in news_items:
            title = str(row.get("title", "") or "").strip()
            content = str(row.get("content", "") or "").strip()
            if title or content:
                news_text_parts.append(f"{title} {content}".strip())
        news_text = "\n".join(news_text_parts)

        market_data = query.market_data or {}
        volume_data = market_data.get("volumes") or market_data.get("volume_data")
        price_data = market_data.get("prices") or market_data.get("price_data") or market_data.get("closes")

        if not volume_data or not price_data:
            closes, _, _, volumes, _, _ = await asyncio.to_thread(
                risk_data_provider.fetch_kline,
                lookup_name,
                90,
            )
            if not price_data:
                price_data = closes
            if not volume_data:
                volume_data = volumes

        result = anomaly_detection_engine.detect(
            stock_name=lookup_name,
            news_text=news_text,
            volume_data=volume_data,
            price_data=price_data,
        )
        result["stock_name"] = lookup_name
        result["input_stock_name"] = query.stock_name
        result["resolution"] = resolution
        result["news_items_analyzed"] = len(news_items)
        result["timestamp"] = datetime.now().isoformat()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/risk/alternative-data")
async def risk_alternative_data(query: AlternativeDataRiskQuery):
    """另类数据接口：搜索热度/龙虎榜/融资融券/情绪过热。"""
    try:
        resolution = resolve_stock(query.stock_name)
        stock_code = str(resolution.get("code") or query.stock_name).strip() or query.stock_name
        data_types = [str(x).strip().lower() for x in (query.data_types or ["hotness", "dragon_tiger", "margin"])]

        payload: Dict[str, Any] = {
            "stock_name": query.stock_name,
            "stock_code": stock_code,
            "resolution": resolution,
            "data": {},
        }

        if "hotness" in data_types or "search_hotness" in data_types:
            payload["data"]["search_hotness"] = alternative_data_engine.fetch_search_hotness(stock_code)
        if "dragon_tiger" in data_types or "lhb" in data_types:
            payload["data"]["dragon_tiger"] = alternative_data_engine.fetch_dragon_tiger(stock_code)
        if "margin" in data_types or "margin_trading" in data_types:
            payload["data"]["margin_trading"] = alternative_data_engine.fetch_margin_trading(stock_code)

        payload["data"]["sentiment_overheating"] = alternative_data_engine.analyze_sentiment_overheating(stock_code)
        payload["timestamp"] = datetime.now().isoformat()
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze")
async def analyze_sentiment(request: AnalysisRequest):
    """
    批量分析中文情感
    
    参数:
        request: AnalysisRequest对象
            - texts: 文本列表
    
    返回:
        情感分析结果列表
    """
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="文本列表不能为空")
        
        if not trainer:
            raise HTTPException(status_code=503, detail="情感分析器未初始化")
        
        results = []
        for text in request.texts:
            analysis = trainer.analyze_sentiment(text)
            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "sentiment": analysis.get("sentiment", "中性"),
                "confidence": analysis.get("confidence", 0.5),
                "label": analysis.get("label", 1)
            })
        
        return {
            "results": results,
            "count": len(results),
            "positive_count": sum(1 for r in results if r["sentiment"] == "正面"),
            "negative_count": sum(1 for r in results if r["sentiment"] == "负面"),
            "neutral_count": sum(1 for r in results if r["sentiment"] == "中性"),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/news")
async def get_news(query: NewsQuery):
    """
    获取实时新闻数据
    
    参数:
        query: NewsQuery对象
            - query: 搜索关键词
            - limit: 返回数量
            - include_social: 是否包含社交媒体
    
    返回:
        新闻列表（已按权重排序）
    """
    try:
        news_items = await fetch_multi_source_data(
            query.query, 
            query.limit or 10,
            _resolve_include_social(query.include_social)
        )
        
        # 构建响应
        return {
            "query": query.query,
            "count": len(news_items),
            "news": news_items,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/alt-data/collect")
async def collect_alt_data(query: AltDataQuery):
    """
    获取国内另类数据（实时 + 历史窗口）并输出风险融合信号。
    """
    try:
        stock_name = (query.stock_name or "").strip()
        if not stock_name:
            raise HTTPException(status_code=400, detail="股票名称不能为空")

        payload = await china_alt_data_engine.collect_signals(
            query=stock_name,
            lookback_days=max(1, int(query.lookback_days or 30)),
            limit_per_source=max(1, int(query.limit_per_source or 8)),
            include_realtime=bool(query.include_realtime if query.include_realtime is not None else True),
            include_history=bool(query.include_history if query.include_history is not None else True),
        )
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/news/summary/{stock_name}")
async def get_news_summary(stock_name: str):
    """
    获取股票新闻摘要
    
    参数:
        stock_name: 股票名称
    
    返回:
        新闻摘要统计信息
    """
    try:
        news_items = await fetch_multi_source_data(stock_name, limit=20)

        positive_count = sum(1 for n in news_items if n.get('sentiment_type') == 'positive')
        negative_count = sum(1 for n in news_items if n.get('sentiment_type') == 'negative')
        neutral_count = sum(1 for n in news_items if n.get('sentiment_type') == 'neutral')
        if positive_count > negative_count:
            sentiment_trend = "positive"
        elif negative_count > positive_count:
            sentiment_trend = "negative"
        else:
            sentiment_trend = "neutral"

        # 统计分析
        stats = {
            "total_count": len(news_items),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "avg_weight": sum(n.get('weight', 0) for n in news_items) / max(len(news_items), 1),
            "sources": list(set(n.get('source', '') for n in news_items)),
            "sentiment_trend": sentiment_trend,
        }
        
        return {
            "stock_name": stock_name,
            "summary": stats,
            "top_news": [{
                "title": n.get('title', ''),
                "source": n.get('source', ''),
                "publish_time": n.get('publish_time', ''),
                "weight": round(n.get('weight', 0), 3),
                "sentiment": n.get('sentiment_type', 'neutral')
            } for n in news_items[:5]],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stocks")
async def get_stock_list():
    """
    获取支持的股票列表
    
    返回:
        股票代码和名称列表
    """
    stocks = [
        {"code": code, "name": name}
        for name, code in stock_code_mapping.items()
    ]
    return {"stocks": stocks}


@app.get("/api/stocks/search")
async def search_stock(q: str, limit: int = 10):
    """股票模糊搜索。"""
    try:
        if not (q or "").strip():
            return {"query": q, "count": 0, "results": [], "timestamp": datetime.now().isoformat()}
        rows = search_stocks(q, max(1, int(limit)))
        return {
            "query": q,
            "count": len(rows),
            "results": rows,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/resolve/{user_input}")
async def resolve_stock_input(user_input: str):
    """股票名称/代码解析。"""
    try:
        resolution = resolve_stock(user_input)
        return {
            "input": user_input,
            "resolution": resolution,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/stats")
async def get_model_stats():
    """
    获取模型统计信息
    
    返回:
        模型配置和状态信息
    """
    if not trainer:
        return {
            "status": "not_initialized",
            "message": "情感分析器未初始化"
        }
    
    return {
        "status": "initialized",
        "model_count": len(trainer.models),
        "models": list(trainer.models.keys()),
        "device": trainer.device,
        "use_ensemble": trainer.use_ensemble,
        "adaptive_threshold": trainer.adaptive_threshold,
        "label_mapping": trainer.label_mapping,
        "action_mapping": trainer.action_mapping,
        "risk_adjustments": trainer.risk_adjustments
    }

@app.get("/api/crawler/stats")
async def get_crawler_stats():
    """
    获取爬虫统计信息
    
    返回:
        爬虫状态和统计信息
    """
    news_stats = news_crawler.get_stats()
    social_stats = social_crawler.get_stats()
    
    return {
        "news_crawler": news_stats,
        "social_crawler": social_stats,
        "alt_data_platforms": china_alt_data_engine.platform_catalog(),
        "alt_data_runtime": china_alt_data_engine.runtime_stats(),
        "aggregator_buffer": news_aggregator.get_buffer_stats(),
        "recent_queries": news_crawler.get_recent_queries(),
        "push_interval_seconds": push_interval,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/health")
async def health_check():
    """
    健康检查
    
    返回:
        系统健康状态
    """
    model_ready = trainer is not None and len(trainer.models) > 0
    crawler_ready = news_crawler is not None
    
    return {
        "status": "healthy" if model_ready and crawler_ready else "degraded",
        "components": {
            "model_analyzer": "ready" if model_ready else "not_ready",
            "news_crawler": "ready" if crawler_ready else "not_ready",
            "social_crawler": "ready" if social_crawler else "not_ready",
            "news_aggregator": "ready" if news_aggregator else "not_ready"
        },
        "active_connections": len(active_connections),
        "timestamp": datetime.now().isoformat()
    }

@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    """
    WebSocket实时推送端点
    
    客户端连接后将接收实时新闻推送
    """
    await websocket.accept()
    async with connection_lock:
        active_connections.append(websocket)
        print(f"[WS] 新连接: {len(active_connections)} 活跃连接")
    
    try:
        while True:
            # 保持连接活跃
            await websocket.receive_text()
    except WebSocketDisconnect:
        async with connection_lock:
            active_connections.remove(websocket)
            print(f"[WS] 连接断开: {len(active_connections)} 活跃连接")
    except Exception as e:
        print(f"[WARN] WebSocket异常: {e}")
        async with connection_lock:
            if websocket in active_connections:
                active_connections.remove(websocket)

@app.post("/api/multi-model/predict")
async def multi_model_prediction(query: StockQuery):
    """
    多模型综合预测 - 整合 FinanceLM 和 NP_LSTM 进行预测
    
    参数:
        query: StockQuery对象
            - stock_name: 股票名称
            - risk_level: 风险偏好
            - limit: 新闻数量限制
    
    返回:
        综合预测结果，包含当下建议和未来预测
    """
    try:
        stock_name = query.stock_name
        
        if not stock_name:
            raise HTTPException(status_code=400, detail="股票名称不能为空")
        
        print(f"[INFO] 多模型预测: {stock_name}")
        
        # 调用多模型协调器
        result = await get_multi_model_prediction(stock_name)
        
        # 构建响应
        response = {
            "stock_name": stock_name,
            "stock_code": stock_code_mapping.get(stock_name, 'unknown'),
            "analysis_time": result.get('analysis_time', datetime.now().isoformat()),
            "market_state": result.get('market_state', {}).get('state_label', '未知'),
            "confidence": round(result.get('confidence', 0.5) * 100),
            "weights_used": {
                model: round(weight, 3) 
                for model, weight in result.get('weights_used', {}).items()
            },
            # ==================== 当下建议 ====================
            "current_suggestion": {
                "action": result.get('current_suggestion', {}).get('action', '观望'),
                "reason": result.get('current_suggestion', {}).get('reason', ''),
                "confidence": round(result.get('current_suggestion', {}).get('confidence', 0.5) * 100),
                "score": round(result.get('current_suggestion', {}).get('score', 0.5), 3),
            },
            # ==================== 未来预测 ====================
            "future_forecast": {
                "trend_summary": result.get('future_forecast', {}).get('trend_summary', ''),
                "predictions_7d": result.get('future_forecast', {}).get('predictions_7d', []),
                "expected_movement": result.get('future_forecast', {}).get('expected_movement', {}),
                "key_levels": result.get('future_forecast', {}).get('key_levels', {}),
            },
            # ==================== 模型预测详情 ====================
            "model_predictions": {
                model: {
                    "success": pred.get('success', False),
                    "trend": pred.get('trend', ''),
                    "confidence": round(pred.get('confidence', 0) * 100),
                    "predictions": pred.get('predictions', [])[:7]
                }
                for model, pred in result.get('model_predictions', {}).items()
            },
            # ==================== 舆情分析 ====================
            "sentiment_analysis": {
                "total_news": result.get('sentiment_analysis', {}).get('total_news', 0),
                "positive_count": result.get('sentiment_analysis', {}).get('positive_count', 0),
                "negative_count": result.get('sentiment_analysis', {}).get('negative_count', 0),
                "sentiment_score": round(result.get('sentiment_analysis', {}).get('sentiment_score', 0), 3),
                "confidence": round(result.get('sentiment_analysis', {}).get('confidence', 0.5) * 100),
            }
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/config/status")
async def config_status(request: Request):
    """返回外部检索服务配置状态，不暴露任何 API Key。"""
    _apply_runtime_api_keys(
        request.headers.get("X-SerpAPI-Key"),
        request.headers.get("X-Tavily-API-Key"),
        request.headers.get("X-Qwen-API-Key"),
        request.headers.get("X-DeepSeek-API-Key"),
        request.headers.get("X-LLM-Provider"),
    )
    serpapi_configured = _has_configured_key(
        ["SERPAPI_API_KEY", "SERP_API_KEY"],
        "serpapi_api_key",
        request,
        "X-SerpAPI-Key",
    )
    tavily_configured = _has_configured_key(
        ["TAVILY_API_KEY", "TRVILY_API_KEY"],
        "tavily_api_key",
        request,
        "X-Tavily-API-Key",
    )
    qwen_api_configured = _has_configured_key(
        ["QWEN_API_KEY", "DASHSCOPE_API_KEY"],
        "qwen_api_key",
        request,
        "X-Qwen-API-Key",
    )
    deepseek_configured = _has_configured_key(
        ["DEEPSEEK_API_KEY"],
        "deepseek_api_key",
        request,
        "X-DeepSeek-API-Key",
    )
    return {
        "serpapi_configured": serpapi_configured,
        "tavily_configured": tavily_configured,
        "qwen_api_configured": qwen_api_configured,
        "deepseek_configured": deepseek_configured,
        "llm_provider": _current_llm_provider(),
        "demo_mode_available": True,
        "external_search_enabled": serpapi_configured or tavily_configured,
        "external_crawl_enabled": (
            os.getenv("RISK_ENABLE_EXTERNAL_CRAWL", "false").lower() in {"1", "true", "yes", "on"}
            or serpapi_configured
            or tavily_configured
        ),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/config/keys")
async def save_config_keys(payload: ApiKeyConfig):
    """黑客松本地运行使用：只保存配置状态到内存，重启后丢失。"""
    status = _apply_runtime_api_keys(payload.serpapi_api_key, payload.tavily_api_key, payload.qwen_api_key, payload.deepseek_api_key, payload.llm_provider)
    return {
        "serpapi_configured": status["serpapi_configured"],
        "tavily_configured": status["tavily_configured"],
        "qwen_api_configured": status["qwen_api_configured"],
        "deepseek_configured": status["deepseek_configured"],
        "llm_provider": status["llm_provider"],
        "demo_mode_available": True,
        "external_search_enabled": status["serpapi_configured"] or status["tavily_configured"],
        "external_crawl_enabled": status["serpapi_configured"] or status["tavily_configured"],
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/")
async def root():
    """
    欢迎页面
    
    返回:
        API文档和端点信息
    """
    return {
        "message": "金融风控智能分析系统 v4.0",
        "version": "4.0.0",
        "description": "基于多模型推理与舆情数据的风险控制与预警系统",
        "endpoints": {
            "POST /api/risk/assess": "核心风控评估",
            "POST /api/risk/portfolio": "组合风控评估（多标的）",
            "POST /api/risk/alerts": "风险告警视图",
            "GET /api/risk/report/{stock_name}": "结构化风险报告",
            "POST /api/risk/stress-test": "压力测试（收益率输入或按股票拉取历史收益）",
            "GET /api/risk/scenarios": "可用历史压力情景列表",
            "POST /api/risk/anomaly-detect": "异常交易与合规风险检测",
            "POST /api/risk/alternative-data": "风险用另类数据（热度/龙虎榜/融资融券）",
            "POST /api/action": "兼容接口：返回风控动作建议",
            "POST /api/multi-model/predict": "多模型综合预测",
            "POST /api/analyze": "批量分析中文情感",
            "POST /api/news": "获取实时新闻数据",
            "POST /api/alt-data/collect": "国内另类数据采集（实时+历史）",
            "GET /api/news/summary/{stock_name}": "获取股票新闻摘要",
            "GET /api/stocks": "股票列表",
            "GET /api/stocks/search?q=xxx": "股票模糊搜索",
            "GET /api/stocks/resolve/{input}": "股票代码/名称解析",
            "GET /api/model/stats": "模型统计信息",
            "GET /api/crawler/stats": "爬虫统计信息",
            "GET /api/health": "健康检查",
            "WebSocket /ws/realtime": "实时新闻推送"
        },
        "features": [
            "[OK] 多源舆情与行情融合风险建模",
            "[OK] 市场状态引擎 2.0（HMM/Regime-Switching + 宏观因子驱动）",
            "[OK] 事件风险大模型层（事件类型/严重度/时效衰减/可逆性/传导链评分）",
            "[OK] VaR/CVaR分位数风险评分 + 动态阈值校准",
            "[OK] 分层升级告警策略（L1-L4）",
            "[OK] 组合风控层（协方差、拥挤度、集中度、行业/风格/杠杆暴露）",
            "[OK] 六维因子风险分解（舆情/趋势/波动/事件/不确定性/市场状态）",
            "[OK] 预训练语义风险模型增强（本地模型优先，失败自动回退）",
            "[OK] risk_sklearn 小模型低权重融合（默认8%，上限20%）",
            "[OK] Qwen 风控叙述增强（失败自动回退模板化输出）",
            "[OK] 模型栈可裁剪（默认关闭次要重模型）",
            "[OK] 风险分级与动作编排",
            "[OK] 多模型推理与预测集成",
            "[OK] 实时新闻抓取与推送",
            "[OK] 国内另类数据层（巨潮/上交所/深交所/央行/统计局/外汇局/海关/气象/环保/百度指数/微博指数/货币网/Shibor）",
            "[OK] 另类数据历史窗口缓存与实时增量融合",
            "[OK] 另类数据反爬安全层（限速/UA轮换/退避重试/403-429惩罚/并发闸门）",
            "[OK] 风险告警标准化输出",
            "[OK] 数据质量约束与降级容错"
        ],
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)




