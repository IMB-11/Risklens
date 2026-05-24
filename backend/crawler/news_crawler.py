"""
新闻舆情爬虫 - 优化版
核心策略：
1. 每个网站只爬取少量新闻（3-5条）
2. 智能请求间隔控制
3. 多源数据融合
4. 自动降级机制
5. 严格的反爬规避
"""

import requests
import json
import time
import random
import re
import os
import calendar
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Tuple
from bs4 import BeautifulSoup
from collections import defaultdict

try:
    import akshare as ak
except Exception:
    ak = None

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if callable(load_dotenv):
    # Ensure backend/crawler/.env is loaded regardless of process cwd.
    load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"), override=False)

# ==================== 反爬配置 ====================
class CrawlerConfig:
    """爬虫配置"""
    
    # User-Agent 轮换池
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    ]
    
    # 每个网站爬取数量
    ITEMS_PER_SOURCE = 3  # 关键：每个网站只爬3条
    
    # 请求模式配置（秒级）
    # 目标：在金融实时性与反爬稳定性之间取得平衡
    REQUEST_MODE = {
        "official_api": {
            "base_delay": 0.22,
            "jitter": (0.04, 0.16),
            "max_rpm": 60,
            "burst_reset_seconds": 1.2,
        },
        "search_api": {
            "base_delay": 0.30,
            "jitter": (0.06, 0.22),
            "max_rpm": 48,
            "burst_reset_seconds": 1.4,
        },
        "web_crawl": {
            "base_delay": 0.85,
            "jitter": (0.16, 0.45),
            "max_rpm": 24,
            "burst_reset_seconds": 2.0,
        },
        "web_crawl_fallback": {
            "base_delay": 1.10,
            "jitter": (0.20, 0.60),
            "max_rpm": 18,
            "burst_reset_seconds": 2.5,
        },
        "akshare_api": {
            "base_delay": 0.70,
            "jitter": (0.10, 0.35),
            "max_rpm": 28,
            "burst_reset_seconds": 1.8,
        },
    }

    # 按域名微调倍率（越大越保守）
    DOMAIN_DELAY_MULTIPLIER = {
        "sse": 1.35,
        "szse": 1.35,
        "csrc": 1.45,
        "pboc": 1.45,
        "securities": 1.20,
        "cnstock": 1.20,
        "cs": 1.20,
        "eastmoney": 1.10,
        "sina": 1.00,
        "qq": 1.15,
        "jrj": 1.15,
        "163": 1.20,
        "sohu": 1.20,
        "akshare": 1.00,
        "serpapi": 1.00,
        "tavily": 1.00,
    }

    DOMAIN_RPM_MULTIPLIER = {
        "sse": 0.80,
        "szse": 0.80,
        "csrc": 0.75,
        "pboc": 0.75,
        "akshare": 0.90,
        "serpapi": 0.90,
        "tavily": 0.90,
    }
    
    # 最大重试次数
    MAX_RETRIES = 2
    
    # 请求超时时间（秒）
    TIMEOUT = 6
    
    # 单域名每分钟最大请求数下限保护
    MIN_REQUESTS_PER_MINUTE = 6

    # AKShare 反爬与保护配置
    AKSHARE_MAX_RETRIES = 3
    AKSHARE_BASE_DELAY_SECONDS = 0.70
    AKSHARE_JITTER_RANGE = (0.10, 0.35)
    AKSHARE_QUERY_CACHE_SECONDS = 180
    AKSHARE_SYMBOL_CACHE_SECONDS = 6 * 3600

    SERPAPI_ENDPOINT = "https://serpapi.com/search.json"
    SERPAPI_ACCOUNT_ENDPOINT = "https://serpapi.com/account.json"
    TAVILY_ENDPOINT = "https://api.tavily.com/search"
    TAVILY_USAGE_ENDPOINT = "https://api.tavily.com/usage"

    # 搜索API预算与额度控制（默认按免费层保护）
    SEARCH_USAGE_CACHE_SECONDS = 30 * 60
    SERPAPI_FREE_MONTHLY_SEARCHES = 250
    TAVILY_FREE_MONTHLY_CREDITS = 1000
    SERPAPI_RESERVE_SEARCHES = 25
    TAVILY_RESERVE_CREDITS = 120
    SERPAPI_FREE_MAX_CALLS_PER_RUN = 2
    TAVILY_FREE_MAX_CALLS_PER_RUN = 2
    SERPAPI_PAID_MAX_CALLS_PER_RUN = 8
    TAVILY_PAID_MAX_CALLS_PER_RUN = 8
    SERPAPI_FREE_MAX_RESULTS_PER_CALL = 3
    TAVILY_FREE_MAX_RESULTS_PER_CALL = 4

# ==================== 请求频率限制器 ====================
class RateLimiter:
    def __init__(self):
        self.timestamps = defaultdict(list)
    
    def _effective_profile(self, domain: str, mode: str) -> Dict[str, Any]:
        profile = CrawlerConfig.REQUEST_MODE.get(mode, CrawlerConfig.REQUEST_MODE["web_crawl"])
        delay_mult = CrawlerConfig.DOMAIN_DELAY_MULTIPLIER.get(domain, 1.0)
        rpm_mult = CrawlerConfig.DOMAIN_RPM_MULTIPLIER.get(domain, 1.0)
        max_rpm = max(
            CrawlerConfig.MIN_REQUESTS_PER_MINUTE,
            int(profile["max_rpm"] * rpm_mult),
        )
        return {
            "base_delay": max(0.03, float(profile["base_delay"]) * delay_mult),
            "jitter": profile["jitter"],
            "max_rpm": max_rpm,
            "burst_reset_seconds": float(profile["burst_reset_seconds"]),
        }
    
    def wait(self, domain: str, mode: str = "web_crawl"):
        """根据域名 + 通道模式进行限流"""
        pf = self._effective_profile(domain, mode)
        now = time.time()
        window = 60  # 60秒窗口
        
        # 清理过期时间戳
        self.timestamps[domain] = [
            t for t in self.timestamps[domain]
            if now - t < window
        ]
        
        # 如果超过限制，等待
        while len(self.timestamps[domain]) >= pf["max_rpm"]:
            oldest = self.timestamps[domain][0]
            wait_time = max(0, oldest + window - now)
            time.sleep(wait_time)
            now = time.time()
            self.timestamps[domain] = [
                t for t in self.timestamps[domain]
                if now - t < window
            ]
        
        # 自适应间隔:
        # - 该域名长时间未请求，允许小抖动快速首包
        # - 连续请求时按基线间隔 + 抖动限速
        last_ts = self.timestamps[domain][-1] if self.timestamps[domain] else None
        if last_ts is None or (now - last_ts) > pf["burst_reset_seconds"]:
            delay = random.uniform(0.02, 0.08)
        else:
            delay = pf["base_delay"] + random.uniform(*pf["jitter"])
        if delay > 0:
            time.sleep(delay)
        
        # 记录时间戳
        self.timestamps[domain].append(time.time())

# 全局限流器
rate_limiter = RateLimiter()

# ==================== 新闻爬虫类 ====================
class NewsCrawler:
    """
    新闻舆情爬虫 - 多源融合
    
    支持的数据源：
    1. 新浪财经 - 可信度高
    2. 东方财富 - 股票相关
    3. 网易财经 - 综合财经
    4. 搜狐财经 - 全面覆盖
    5. 腾讯财经 - 实时更新
    6. 金融界 - 专业财经
    """
    
    def __init__(self):
        self.session = self._create_session()
        self.retry_count = defaultdict(int)
        self.last_source_runtime: List[Dict[str, Any]] = []
        self.search_api_budget_mode = (os.getenv("SEARCH_API_BUDGET_MODE", "auto") or "auto").strip().lower()
        if self.search_api_budget_mode not in {"auto", "free", "paid"}:
            self.search_api_budget_mode = "auto"
        self.serpapi_api_key = (os.getenv("SERPAPI_API_KEY", "") or "").strip()
        self.tavily_api_key = (
            os.getenv("TAVILY_API_KEY")
            or os.getenv("TRVILY_API_KEY")
            or os.getenv("TVLY_API_KEY")
            or ""
        ).strip()
        self._usage_cache: Dict[str, Dict[str, Any]] = {
            "serpapi": {"ts": 0.0, "payload": {}},
            "tavily": {"ts": 0.0, "payload": {}},
        }
        self._provider_calls_this_run: Dict[str, int] = defaultdict(int)
        self._last_search_budget: Dict[str, Any] = {}
        self._akshare_last_call_ts = 0.0
        self._akshare_query_cache: Dict[str, Dict[str, Any]] = {}
        self._akshare_symbol_cache: Dict[str, Any] = {
            "updated_at": 0.0,
            "mapping": {},
        }
        print("[OK] 新闻舆情爬虫初始化完成")
        print("   [OK] 策略: 每个网站爬取3条新闻")
        print("   [OK] 已启用: API优先、请求限流、UA轮换、失败重试")
        print(f"   [OK] SerpApi配置: {'ON' if self.serpapi_api_key else 'OFF'}")
        print(f"   [OK] Tavily配置: {'ON' if self.tavily_api_key else 'OFF'}")
        print(f"   [OK] Search预算模式: {self.search_api_budget_mode}")
    
    def _create_session(self) -> requests.Session:
        """创建带反爬配置的Session"""
        session = requests.Session()
        
        # 默认headers
        session.headers.update({
            'User-Agent': random.choice(CrawlerConfig.USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Cache-Control': 'max-age=0',
        })
        
        session.timeout = CrawlerConfig.TIMEOUT
        return session
    
    def _rotate_headers(self, referer: str = None):
        """轮换请求头"""
        self.session.headers['User-Agent'] = random.choice(CrawlerConfig.USER_AGENTS)
        if referer:
            self.session.headers['Referer'] = referer
    
    def _retry_request(
        self,
        url: str,
        params: dict = None,
        method: str = 'GET',
        mode: str = "web_crawl",
    ) -> Optional[requests.Response]:
        """带重试的请求"""
        domain = self._extract_domain(url)
        
        for attempt in range(CrawlerConfig.MAX_RETRIES):
            try:
                # 限流
                rate_limiter.wait(domain, mode=mode)
                
                # 轮换headers
                self._rotate_headers(referer='https://www.baidu.com/')
                
                # 发送请求
                if method == 'GET':
                    response = self.session.get(url, params=params)
                else:
                    response = self.session.post(url, data=params)
                
                # 检查状态码
                if response.status_code == 200:
                    self.retry_count[domain] = 0
                    return response
                elif response.status_code == 403:
                    wait_sec = 1.3 * (attempt + 1) + random.uniform(0.3, 0.8)
                    print(f"   [WARN] {domain} 返回403，等待{wait_sec:.1f}s后重试...")
                    time.sleep(wait_sec)
                elif response.status_code == 429:
                    wait_sec = min(6.0, 1.8 * (attempt + 1) + random.uniform(0.5, 1.2))
                    print(f"   [WARN] {domain} 返回429，等待{wait_sec:.1f}s后重试...")
                    time.sleep(wait_sec)
                    
            except requests.exceptions.RequestException as e:
                print(f"   [WARN] 请求失败 ({attempt+1}/{CrawlerConfig.MAX_RETRIES}): {str(e)[:40]}")
                if attempt < CrawlerConfig.MAX_RETRIES - 1:
                    time.sleep(0.7 * (attempt + 1) + random.uniform(0.2, 0.6))
        
        self.retry_count[domain] += 1
        return None
    
    def _extract_domain(self, url: str) -> str:
        """提取域名"""
        match = re.match(r'https?://([^/]+)', url)
        if match:
            host = match.group(1).split(":")[0].lower()
            parts = host.split(".")
            if len(parts) >= 3 and parts[-2] in {"co", "com", "net", "org"}:
                return parts[-3]
            if len(parts) >= 2:
                return parts[-2]
            return parts[0]
        return 'unknown'

    # ========== AKShare: API优先 + 反爬保护 ==========
    def _akshare_wait(self) -> None:
        """
        AKShare调用节流:
        - 固定基线间隔
        - 随机抖动
        - 避免短时间重复访问同类端点
        """
        rate_limiter.wait("akshare", mode="akshare_api")
        jitter = random.uniform(*CrawlerConfig.AKSHARE_JITTER_RANGE)
        min_gap = CrawlerConfig.AKSHARE_BASE_DELAY_SECONDS + jitter
        elapsed = time.time() - self._akshare_last_call_ts
        if elapsed < min_gap:
            time.sleep(min_gap - elapsed)
        self._akshare_last_call_ts = time.time()

    def _normalize_code(self, code: Any) -> str:
        raw = str(code or "").strip()
        digits = re.sub(r"\D", "", raw)
        return digits[-6:] if len(digits) >= 6 else raw

    def _infer_symbol_from_query(self, query: str) -> str:
        # 1) query直接就是6位代码
        code = self._normalize_code(query)
        if re.fullmatch(r"\d{6}", code or ""):
            return code

        # 2) 从缓存的 股票名->代码 映射中匹配
        mapping = self._load_akshare_symbol_mapping()
        if not mapping:
            return ""
        q = (query or "").strip().lower()

        # 先精确
        if q in mapping:
            return mapping[q]

        # 再包含匹配
        for name_key, sym in mapping.items():
            if q and (q in name_key or name_key in q):
                return sym
        return ""

    def _load_akshare_symbol_mapping(self) -> Dict[str, str]:
        now = time.time()
        cache_updated = float(self._akshare_symbol_cache.get("updated_at", 0.0) or 0.0)
        if now - cache_updated < CrawlerConfig.AKSHARE_SYMBOL_CACHE_SECONDS:
            mapping = self._akshare_symbol_cache.get("mapping", {})
            if isinstance(mapping, dict):
                return mapping

        if ak is None:
            return {}

        mapping: Dict[str, str] = {}
        last_err = None
        for attempt in range(CrawlerConfig.AKSHARE_MAX_RETRIES):
            try:
                self._akshare_wait()
                code_name_df = ak.stock_info_a_code_name()
                if code_name_df is None or getattr(code_name_df, "empty", True):
                    break

                col_code = None
                col_name = None
                for candidate in ("code", "代码", "symbol"):
                    if candidate in code_name_df.columns:
                        col_code = candidate
                        break
                for candidate in ("name", "名称", "股票简称"):
                    if candidate in code_name_df.columns:
                        col_name = candidate
                        break
                if not col_code or not col_name:
                    break

                for _, row in code_name_df[[col_code, col_name]].dropna().iterrows():
                    code = self._normalize_code(row[col_code])
                    name = str(row[col_name]).strip()
                    if re.fullmatch(r"\d{6}", code) and name:
                        mapping[name.lower()] = code
                if mapping:
                    break
            except Exception as exc:
                last_err = exc
                backoff = 1.2 * (attempt + 1) + random.uniform(0.2, 0.8)
                time.sleep(backoff)

        if not mapping and last_err:
            print(f"   [WARN] AKShare 代码映射加载失败: {str(last_err)[:80]}")

        # 失败时缩短缓存，避免长时间“空映射”冻结
        cache_ts = now if mapping else (now - CrawlerConfig.AKSHARE_SYMBOL_CACHE_SECONDS + 90)
        self._akshare_symbol_cache = {
            "updated_at": cache_ts,
            "mapping": mapping,
        }
        return mapping

    def _akshare_get_cached(self, query: str) -> Optional[List[Dict[str, Any]]]:
        key = (query or "").strip().lower()
        row = self._akshare_query_cache.get(key)
        if not row:
            return None
        ts = float(row.get("ts", 0.0) or 0.0)
        if time.time() - ts > CrawlerConfig.AKSHARE_QUERY_CACHE_SECONDS:
            return None
        data = row.get("data")
        return list(data) if isinstance(data, list) else None

    def _akshare_set_cached(self, query: str, data: List[Dict[str, Any]]) -> None:
        key = (query or "").strip().lower()
        self._akshare_query_cache[key] = {
            "ts": time.time(),
            "data": list(data),
        }

    def _apply_fetch_meta(
        self,
        items: List[Dict[str, Any]],
        source_id: str,
        fetch_mode: str,
        api_attempted: bool,
        fallback_used: bool,
    ) -> List[Dict[str, Any]]:
        """为每条记录附加抓取通道元数据，便于后续观测 API 命中率。"""
        for row in items:
            if not isinstance(row, dict):
                continue
            row.setdefault("source_id", source_id)
            row["fetch_mode"] = fetch_mode
            row["api_attempted"] = bool(api_attempted)
            row["fallback_used"] = bool(fallback_used)
        return items

    def _fetch_source_api_first(
        self,
        source_id: str,
        source_name: str,
        keyword: str,
        limit: int,
        api_fetcher: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None,
        crawl_fetcher: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        API-first 抓取策略：
        1) 有 API 就先 API
        2) API 失败或为空再走爬取 fallback
        3) 无 API 则直接爬取
        """
        runtime: Dict[str, Any] = {
            "source_id": source_id,
            "source_name": source_name,
            "requested_limit": max(1, int(limit)),
            "api_attempted": False,
            "fallback_used": False,
            "mode": "none",
            "count": 0,
            "status": "failed",
        }

        if callable(api_fetcher):
            runtime["api_attempted"] = True
            api_items = api_fetcher(keyword, limit) or []
            if api_items:
                api_items = self._apply_fetch_meta(
                    api_items,
                    source_id=source_id,
                    fetch_mode="official_api",
                    api_attempted=True,
                    fallback_used=False,
                )
                runtime.update({
                    "mode": "official_api",
                    "count": len(api_items),
                    "status": "ok",
                })
                return api_items, runtime

        if callable(crawl_fetcher):
            crawl_items = crawl_fetcher(keyword, limit) or []
            if crawl_items:
                fallback_used = bool(runtime["api_attempted"])
                mode = "web_crawl_fallback" if fallback_used else "web_crawl"
                crawl_items = self._apply_fetch_meta(
                    crawl_items,
                    source_id=source_id,
                    fetch_mode=mode,
                    api_attempted=bool(runtime["api_attempted"]),
                    fallback_used=fallback_used,
                )
                runtime.update({
                    "mode": mode,
                    "count": len(crawl_items),
                    "status": "ok",
                    "fallback_used": fallback_used,
                })
                return crawl_items, runtime

        return [], runtime

    def _resolve_source_limit(self, source_id: str, base_limit: int) -> int:
        """
        按数据源特性定制抓取条数：
        - SerpApi / Tavily：优先保护免费额度，单次结果数收敛。
        - 其他源：沿用基础条数。
        """
        base = max(1, int(base_limit))
        if source_id == "serpapi":
            return min(3, max(2, base))
        if source_id == "tavily":
            return min(4, max(2, base))
        if source_id == "serpapi_signal":
            return min(2, max(1, base // 2))
        if source_id == "tavily_signal":
            return min(2, max(1, base // 2))
        return base
    
    # ========== 各网站爬虫方法 ==========
    def crawl_sina(self, keyword: str, limit: int = 3) -> List[Dict]:
        """
        新浪财经 - 可信度高
        """
        news_list = []
        try:
            url = f"https://interface.sina.cn/wap_api/news_search.do?keyword={keyword}&size={limit}"
            response = self._retry_request(url, mode="official_api")
            
            if response:
                data = response.json()
                if 'result' in data and 'list' in data['result']:
                    for item in data['result']['list'][:limit]:
                        news_list.append({
                            'title': item.get('title', ''),
                            'url': item.get('url', ''),
                            'source': '新浪财经',
                            'publish_time': item.get('ctime', ''),
                            'data_source_type': 'news',
                            'reliability': 0.85,
                            'category': self._classify_news(item.get('title', ''))
                        })
                print(f"   [OK] 新浪财经: {len(news_list)} 条")
        except Exception as e:
            print(f"   [WARN] 新浪财经爬取失败: {str(e)[:40]}")
        return news_list
    
    def crawl_eastmoney(self, keyword: str, limit: int = 3) -> List[Dict]:
        """
        东方财富 - 股票相关
        """
        news_list = []
        try:
            url = f"https://push2.eastmoney.com/api/qt/keytword/get?keyword={keyword}&type=1&count={limit}"
            response = self._retry_request(url, mode="official_api")
            
            if response:
                data = response.json()
                if 'data' in data and data['data']:
                    for item in data['data'][:limit]:
                        news_list.append({
                            'title': item.get('Title', item.get('title', '')),
                            'url': item.get('Url', item.get('url', '')),
                            'source': '东方财富',
                            'publish_time': item.get('UpdateTime', ''),
                            'data_source_type': 'news',
                            'reliability': 0.88,
                            'category': self._classify_news(item.get('Title', ''))
                        })
                print(f"   [OK] 东方财富: {len(news_list)} 条")
        except Exception as e:
            print(f"   [WARN] 东方财富爬取失败: {str(e)[:40]}")
        return news_list

    def crawl_akshare_news(self, keyword: str, limit: int = 3) -> List[Dict]:
        """
        AKShare 数据源（API封装）:
        - 优先使用 AKShare 的个股新闻接口
        - 内置节流 + 抖动 + 重试 + 短期缓存，降低触发限频概率
        """
        news_list: List[Dict] = []
        if ak is None:
            return news_list

        cached = self._akshare_get_cached(keyword)
        if cached is not None:
            return cached[: max(1, int(limit))]

        symbol = self._infer_symbol_from_query(keyword)
        if not symbol:
            # 无法映射到代码时直接返回空，不强行重载端点
            return news_list

        last_err = None
        for attempt in range(CrawlerConfig.AKSHARE_MAX_RETRIES):
            try:
                self._akshare_wait()
                df = ak.stock_news_em(symbol=symbol)
                if df is None or getattr(df, "empty", True):
                    break

                # 常见列名兼容
                title_col = None
                content_col = None
                time_col = None
                source_col = None
                url_col = None
                for c in ("新闻标题", "标题", "title"):
                    if c in df.columns:
                        title_col = c
                        break
                for c in ("新闻内容", "内容", "摘要", "content"):
                    if c in df.columns:
                        content_col = c
                        break
                for c in ("发布时间", "时间", "日期", "publish_time"):
                    if c in df.columns:
                        time_col = c
                        break
                for c in ("文章来源", "来源", "source"):
                    if c in df.columns:
                        source_col = c
                        break
                for c in ("新闻链接", "链接", "url"):
                    if c in df.columns:
                        url_col = c
                        break

                # 即使列名不全也尽量产出
                rows = df.head(max(1, int(limit) * 2))
                for _, row in rows.iterrows():
                    title = str(row[title_col]).strip() if title_col else ""
                    content = str(row[content_col]).strip() if content_col else ""
                    publish_time = str(row[time_col]).strip() if time_col else ""
                    src = str(row[source_col]).strip() if source_col else "AKShare"
                    link = str(row[url_col]).strip() if url_col else ""

                    if not title and not content:
                        continue

                    # 轻过滤：至少标题或内容命中关键词
                    text = f"{title} {content}".lower()
                    if keyword and keyword.lower() not in text:
                        continue

                    news_list.append({
                        "title": title or content[:80],
                        "content": content,
                        "url": link,
                        "source": f"AKShare-{src}" if src else "AKShare",
                        "publish_time": publish_time,
                        "data_source_type": "news",
                        "reliability": 0.90,
                        "category": self._classify_news(title or content),
                    })

                    if len(news_list) >= max(1, int(limit)):
                        break

                if news_list:
                    self._akshare_set_cached(keyword, news_list)
                break
            except Exception as exc:
                last_err = exc
                backoff = 1.0 * (attempt + 1) + random.uniform(0.4, 1.4)
                time.sleep(backoff)

        if not news_list and last_err is not None:
            print(f"   [WARN] AKShare 获取失败: {str(last_err)[:60]}")
        else:
            print(f"   [OK] AKShare: {len(news_list)} 条")
        return news_list

    def _retry_json_post(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        mode: str = "search_api",
    ) -> Optional[requests.Response]:
        """带重试的 JSON POST 请求。"""
        domain = self._extract_domain(url)
        merged_headers = {"Content-Type": "application/json"}
        if headers:
            merged_headers.update(headers)

        for attempt in range(CrawlerConfig.MAX_RETRIES):
            try:
                rate_limiter.wait(domain, mode=mode)
                self._rotate_headers(referer="https://www.baidu.com/")
                response = self.session.post(
                    url,
                    json=payload,
                    headers=merged_headers,
                    timeout=CrawlerConfig.TIMEOUT,
                )
                if response.status_code == 200:
                    return response
                if response.status_code in (401, 403):
                    break
                if response.status_code == 429:
                    wait_sec = min(6.0, 1.8 * (attempt + 1) + random.uniform(0.5, 1.2))
                    time.sleep(wait_sec)
                    continue
            except requests.exceptions.RequestException:
                if attempt < CrawlerConfig.MAX_RETRIES - 1:
                    time.sleep(0.7 * (attempt + 1) + random.uniform(0.2, 0.6))
        return None

    def _retry_json_get_with_headers(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        mode: str = "search_api",
    ) -> Optional[requests.Response]:
        """带重试的 JSON GET（支持自定义headers，如Bearer鉴权）。"""
        domain = self._extract_domain(url)
        merged_headers = {"Accept": "application/json"}
        if headers:
            merged_headers.update(headers)

        for attempt in range(CrawlerConfig.MAX_RETRIES):
            try:
                rate_limiter.wait(domain, mode=mode)
                self._rotate_headers(referer="https://www.baidu.com/")
                response = self.session.get(
                    url,
                    params=params or {},
                    headers=merged_headers,
                    timeout=CrawlerConfig.TIMEOUT,
                )
                if response.status_code == 200:
                    return response
                if response.status_code in (401, 403):
                    break
                if response.status_code == 429:
                    wait_sec = min(6.0, 1.8 * (attempt + 1) + random.uniform(0.5, 1.2))
                    time.sleep(wait_sec)
                    continue
            except requests.exceptions.RequestException:
                if attempt < CrawlerConfig.MAX_RETRIES - 1:
                    time.sleep(0.7 * (attempt + 1) + random.uniform(0.2, 0.6))
        return None

    def _days_left_in_month(self) -> int:
        now = datetime.now()
        month_days = calendar.monthrange(now.year, now.month)[1]
        return max(1, month_days - now.day + 1)

    def _is_free_mode(self, provider: str, usage_payload: Dict[str, Any]) -> bool:
        mode = self.search_api_budget_mode
        if mode == "free":
            return True
        if mode == "paid":
            return False

        if provider == "serpapi":
            try:
                monthly_price = float(usage_payload.get("plan_monthly_price", 0.0) or 0.0)
            except (TypeError, ValueError):
                monthly_price = 0.0
            monthly_limit = int(usage_payload.get("searches_per_month") or CrawlerConfig.SERPAPI_FREE_MONTHLY_SEARCHES)
            return monthly_price <= 0.0 or monthly_limit <= CrawlerConfig.SERPAPI_FREE_MONTHLY_SEARCHES

        if provider == "tavily":
            account = usage_payload.get("account", {}) if isinstance(usage_payload, dict) else {}
            key_info = usage_payload.get("key", {}) if isinstance(usage_payload, dict) else {}
            plan_name = str(account.get("current_plan", "") or "").strip().lower()
            monthly_limit = int(key_info.get("limit") or CrawlerConfig.TAVILY_FREE_MONTHLY_CREDITS)
            return plan_name in {"researcher", "free"} or monthly_limit <= CrawlerConfig.TAVILY_FREE_MONTHLY_CREDITS

        return True

    def _get_cached_usage(self, provider: str) -> Optional[Dict[str, Any]]:
        row = self._usage_cache.get(provider, {})
        ts = float(row.get("ts", 0.0) or 0.0)
        if time.time() - ts > CrawlerConfig.SEARCH_USAGE_CACHE_SECONDS:
            return None
        payload = row.get("payload")
        return payload if isinstance(payload, dict) else None

    def _set_cached_usage(self, provider: str, payload: Dict[str, Any]) -> None:
        self._usage_cache[provider] = {
            "ts": time.time(),
            "payload": dict(payload or {}),
        }

    def _fetch_serpapi_usage(self) -> Dict[str, Any]:
        if not self.serpapi_api_key:
            return {}
        cached = self._get_cached_usage("serpapi")
        if cached is not None:
            return cached

        response = self._retry_request(
            CrawlerConfig.SERPAPI_ACCOUNT_ENDPOINT,
            params={"api_key": self.serpapi_api_key},
            method="GET",
            mode="search_api",
        )
        payload: Dict[str, Any] = {}
        if response is not None:
            try:
                payload = response.json() or {}
            except Exception:
                payload = {}
        self._set_cached_usage("serpapi", payload)
        return payload

    def _fetch_tavily_usage(self) -> Dict[str, Any]:
        if not self.tavily_api_key:
            return {}
        cached = self._get_cached_usage("tavily")
        if cached is not None:
            return cached

        response = self._retry_json_get_with_headers(
            CrawlerConfig.TAVILY_USAGE_ENDPOINT,
            headers={"Authorization": f"Bearer {self.tavily_api_key}"},
            mode="search_api",
        )
        payload: Dict[str, Any] = {}
        if response is not None:
            try:
                payload = response.json() or {}
            except Exception:
                payload = {}
        self._set_cached_usage("tavily", payload)
        return payload

    def _build_provider_budget(self, provider: str) -> Dict[str, Any]:
        days_left = self._days_left_in_month()
        if provider == "serpapi":
            usage = self._fetch_serpapi_usage()
            monthly_limit = int(usage.get("searches_per_month") or CrawlerConfig.SERPAPI_FREE_MONTHLY_SEARCHES)
            monthly_left = int(
                usage.get("total_searches_left")
                or usage.get("plan_searches_left")
                or monthly_limit
            )
            hourly_limit = int(usage.get("account_rate_limit_per_hour") or 50)
            last_hour_used = int(usage.get("last_hour_searches") or 0)
            reserve = min(CrawlerConfig.SERPAPI_RESERVE_SEARCHES, max(0, monthly_limit - 1))
            free_mode = self._is_free_mode("serpapi", usage)
            hard_cap = (
                CrawlerConfig.SERPAPI_FREE_MAX_CALLS_PER_RUN
                if free_mode else CrawlerConfig.SERPAPI_PAID_MAX_CALLS_PER_RUN
            )
            result_cap = (
                CrawlerConfig.SERPAPI_FREE_MAX_RESULTS_PER_CALL
                if free_mode else max(6, CrawlerConfig.SERPAPI_FREE_MAX_RESULTS_PER_CALL * 2)
            )
        else:
            usage = self._fetch_tavily_usage()
            key_info = usage.get("key", {}) if isinstance(usage, dict) else {}
            monthly_limit = int(key_info.get("limit") or CrawlerConfig.TAVILY_FREE_MONTHLY_CREDITS)
            monthly_left = int(key_info.get("remaining") or monthly_limit)
            reserve = min(CrawlerConfig.TAVILY_RESERVE_CREDITS, max(0, monthly_limit - 1))
            free_mode = self._is_free_mode("tavily", usage)
            hard_cap = (
                CrawlerConfig.TAVILY_FREE_MAX_CALLS_PER_RUN
                if free_mode else CrawlerConfig.TAVILY_PAID_MAX_CALLS_PER_RUN
            )
            result_cap = (
                CrawlerConfig.TAVILY_FREE_MAX_RESULTS_PER_CALL
                if free_mode else max(8, CrawlerConfig.TAVILY_FREE_MAX_RESULTS_PER_CALL * 2)
            )

        available_after_reserve = max(0, monthly_left - reserve)
        daily_budget = available_after_reserve / max(days_left, 1)
        if available_after_reserve <= 0:
            allowed_calls = 0
        elif daily_budget < 1.0:
            allowed_calls = 1
        elif daily_budget < 3.0:
            allowed_calls = min(2, hard_cap)
        else:
            allowed_calls = min(3, hard_cap)

        # 若当月余额非常充足，不必过度保守
        if available_after_reserve > (monthly_limit * 0.40):
            allowed_calls = min(max(allowed_calls, 2), hard_cap)

        if provider == "serpapi":
            if last_hour_used >= max(0, hourly_limit - 1):
                allowed_calls = 0
            elif last_hour_used >= max(0, hourly_limit - 3):
                allowed_calls = min(allowed_calls, 1)

        return {
            "provider": provider,
            "free_mode": bool(free_mode),
            "monthly_limit": int(monthly_limit),
            "monthly_left": int(monthly_left),
            "reserve": int(reserve),
            "days_left": int(days_left),
            "available_after_reserve": int(available_after_reserve),
            "allowed_calls_per_run": int(max(0, allowed_calls)),
            "max_results_per_call": int(max(1, result_cap)),
            "hourly_limit": int(hourly_limit) if provider == "serpapi" else None,
            "last_hour_used": int(last_hour_used) if provider == "serpapi" else None,
        }

    def _reserve_provider_call(self, provider: str) -> Dict[str, Any]:
        budget = self._build_provider_budget(provider)
        used = int(self._provider_calls_this_run.get(provider, 0))
        allowed = int(budget.get("allowed_calls_per_run", 0))
        remaining_calls = max(0, allowed - used)
        budget["used_calls_this_run"] = used
        budget["remaining_calls_this_run"] = remaining_calls
        if remaining_calls <= 0:
            budget["allow"] = False
            self._last_search_budget[provider] = dict(budget)
            return budget
        self._provider_calls_this_run[provider] = used + 1
        budget["allow"] = True
        budget["used_calls_this_run"] = used + 1
        budget["remaining_calls_this_run"] = max(0, allowed - (used + 1))
        self._last_search_budget[provider] = dict(budget)
        return budget

    def _allocate_budgets(self, total: int, weights: List[float], min_each: int = 1) -> List[int]:
        n = len(weights)
        if n == 0:
            return []
        total_safe = max(n * min_each, int(total))
        budgets = [min_each for _ in range(n)]
        remain = total_safe - n * min_each
        if remain <= 0:
            return budgets

        w_sum = sum(max(0.0, float(w)) for w in weights) or float(n)
        scaled = [remain * (max(0.0, float(w)) / w_sum) for w in weights]
        for i, val in enumerate(scaled):
            add = int(val)
            budgets[i] += add
            remain -= add
            scaled[i] = val - add

        # 把剩余名额分配给小数部分最大的桶
        order = sorted(range(n), key=lambda i: scaled[i], reverse=True)
        idx = 0
        while remain > 0:
            budgets[order[idx % n]] += 1
            idx += 1
            remain -= 1
        return budgets

    def _serp_source_name(self, raw_source: Any) -> str:
        if isinstance(raw_source, dict):
            return str(raw_source.get("name") or raw_source.get("title") or "SerpApi").strip()
        return str(raw_source or "SerpApi").strip()

    def _infer_signal_type(self, text: str) -> Tuple[str, str, float]:
        content = (text or "").lower()
        disclosure_kw = ("公告", "信披", "问询函", "停牌", "复牌", "减持", "增持", "监管", "处罚")
        research_kw = ("研报", "评级", "目标价", "券商", "机构观点")
        flow_kw = ("资金流向", "融资融券", "北向资金", "龙虎榜", "主力资金", "换手率", "成交额")

        if any(k in content for k in disclosure_kw):
            return "official_disclosure", "disclosure", 0.93
        if any(k in content for k in research_kw):
            return "analyst_research", "research", 0.85
        if any(k in content for k in flow_kw):
            return "capital_flow", "capital_flow", 0.83
        return "generic_signal", self._classify_news(content), 0.80

    def crawl_serpapi_news(self, keyword: str, limit: int = 3) -> List[Dict]:
        """
        SerpApi 新闻检索（Google News）:
        - 用于补充多站新闻覆盖
        - API通道，非网页硬爬
        """
        news_list: List[Dict] = []
        if not self.serpapi_api_key:
            return news_list

        try:
            budget = self._reserve_provider_call("serpapi")
            if not budget.get("allow", False):
                print("   [SKIP] SerpApi: 免费额度保护触发，跳过本轮调用")
                return news_list
            effective_limit = min(
                max(1, int(limit)),
                int(budget.get("max_results_per_call", CrawlerConfig.SERPAPI_FREE_MAX_RESULTS_PER_CALL)),
            )

            params = {
                "engine": "google_news",
                "q": keyword,
                "hl": "zh-cn",
                "gl": "cn",
                "num": effective_limit,
                "api_key": self.serpapi_api_key,
            }
            response = self._retry_request(
                CrawlerConfig.SERPAPI_ENDPOINT,
                params=params,
                method="GET",
                mode="search_api",
            )
            if not response:
                return news_list

            data = response.json() or {}
            rows = data.get("news_results", [])
            if not isinstance(rows, list):
                rows = []

            for item in rows[:effective_limit]:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "")).strip()
                snippet = str(item.get("snippet", "")).strip()
                if not title and not snippet:
                    continue
                source_name = self._serp_source_name(item.get("source", "SerpApi"))
                news_list.append({
                    "title": title or snippet[:80],
                    "content": snippet,
                    "url": str(item.get("link", "")).strip(),
                    "source": f"SerpApi-{source_name}" if source_name else "SerpApi",
                    "publish_time": str(item.get("date", "")).strip(),
                    "data_source_type": "search_news",
                    "reliability": 0.86,
                    "category": self._classify_news(f"{title} {snippet}"),
                })
            print(
                f"   [OK] SerpApi: {len(news_list)} 条 "
                f"(run {budget.get('used_calls_this_run')}/{budget.get('allowed_calls_per_run')}, "
                f"month_left={budget.get('monthly_left')})"
            )
        except Exception as exc:
            print(f"   [WARN] SerpApi 获取失败: {str(exc)[:60]}")
        return news_list

    def crawl_serpapi_signals(self, keyword: str, limit: int = 4) -> List[Dict]:
        """
        SerpApi 多类型信号检索（非纯新闻）:
        - 信披/监管（公告、问询、停复牌、减持增持）
        - 研报/评级（目标价、券商观点）
        - 资金/交易（融资融券、北向、龙虎榜、主力资金）
        """
        signals: List[Dict] = []
        if not self.serpapi_api_key:
            return signals

        try:
            budget = self._reserve_provider_call("serpapi")
            if not budget.get("allow", False):
                print("   [SKIP] SerpApi信号: 免费额度保护触发，跳过本轮调用")
                return signals
            effective_limit = min(
                max(1, int(limit)),
                int(budget.get("max_results_per_call", CrawlerConfig.SERPAPI_FREE_MAX_RESULTS_PER_CALL)),
            )

            query = (
                f"{keyword} 公告 信披 问询函 停复牌 减持 增持 "
                f"研报 评级 目标价 券商观点 资金流向 融资融券 北向资金 龙虎榜 主力资金"
            )
            params = {
                "engine": "google",
                "q": query,
                "hl": "zh-cn",
                "gl": "cn",
                "num": max(3, min(10, effective_limit * 3)),
                "api_key": self.serpapi_api_key,
            }
            response = self._retry_request(
                CrawlerConfig.SERPAPI_ENDPOINT,
                params=params,
                method="GET",
                mode="search_api",
            )
            if not response:
                return signals

            data = response.json() or {}
            rows = data.get("organic_results", [])
            if not isinstance(rows, list):
                rows = []

            seen = set()
            for item in rows:
                if len(signals) >= effective_limit:
                    break
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "")).strip()
                snippet = str(item.get("snippet", "")).strip()
                url = str(item.get("link", "")).strip()
                if not title and not snippet:
                    continue

                dedup_key = re.sub(r"\s+", "", f"{title}|{url}".lower())
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                source_name = self._serp_source_name(item.get("source", "SerpApi"))
                text = f"{title} {snippet}".strip()
                signal_type, category, reliability = self._infer_signal_type(text)
                signals.append({
                    "title": title or snippet[:80],
                    "content": snippet,
                    "url": url,
                    "source": f"SerpApi-Signal-{source_name}",
                    "publish_time": str(item.get("date", "")).strip(),
                    "data_source_type": "search_signal",
                    "signal_type": signal_type,
                    "reliability": reliability,
                    "category": category,
                })

            print(
                f"   [OK] SerpApi信号: {len(signals)} 条 "
                f"(run {budget.get('used_calls_this_run')}/{budget.get('allowed_calls_per_run')}, "
                f"month_left={budget.get('monthly_left')})"
            )
        except Exception as exc:
            print(f"   [WARN] SerpApi信号获取失败: {str(exc)[:60]}")
        return signals

    def crawl_tavily_news(self, keyword: str, limit: int = 3) -> List[Dict]:
        """
        Tavily 新闻检索:
        - topic=news，优先拿结构化新闻结果
        - 仅走 API，不进行页面爬取
        """
        news_list: List[Dict] = []
        if not self.tavily_api_key:
            return news_list

        try:
            budget = self._reserve_provider_call("tavily")
            if not budget.get("allow", False):
                print("   [SKIP] Tavily: 免费额度保护触发，跳过本轮调用")
                return news_list
            effective_limit = min(
                max(1, int(limit)),
                int(budget.get("max_results_per_call", CrawlerConfig.TAVILY_FREE_MAX_RESULTS_PER_CALL)),
            )

            payload = {
                "api_key": self.tavily_api_key,
                "query": f"{keyword} 股票 风险 舆情",
                "topic": "news",
                "search_depth": "basic",
                "max_results": effective_limit,
                "include_raw_content": False,
                "include_usage": True,
            }
            response = self._retry_json_post(
                CrawlerConfig.TAVILY_ENDPOINT,
                payload=payload,
                mode="search_api",
            )
            if not response:
                return news_list

            data = response.json() or {}
            rows = data.get("results", [])
            if not isinstance(rows, list):
                rows = []

            for item in rows[:effective_limit]:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "")).strip()
                content = str(item.get("content", "")).strip()
                if not title and not content:
                    continue
                score = item.get("score", 0.0)
                try:
                    score = float(score)
                except (TypeError, ValueError):
                    score = 0.0
                reliability = 0.78 + max(0.0, min(0.18, 0.18 * score))
                text = f"{title} {content}".strip()
                news_list.append({
                    "title": title or content[:80],
                    "content": content,
                    "url": str(item.get("url", "")).strip(),
                    "source": "Tavily-News",
                    "publish_time": str(item.get("published_date", "")).strip(),
                    "data_source_type": "search_news",
                    "reliability": round(reliability, 6),
                    "category": self._classify_news(text),
                })
            print(
                f"   [OK] Tavily: {len(news_list)} 条 "
                f"(run {budget.get('used_calls_this_run')}/{budget.get('allowed_calls_per_run')}, "
                f"month_left={budget.get('monthly_left')})"
            )
        except Exception as exc:
            print(f"   [WARN] Tavily 获取失败: {str(exc)[:60]}")
        return news_list

    def crawl_tavily_signals(self, keyword: str, limit: int = 4) -> List[Dict]:
        """
        Tavily 多类型信号检索（topic=finance）:
        - 面向公告、研报、资金风向的结构化检索
        """
        signals: List[Dict] = []
        if not self.tavily_api_key:
            return signals

        try:
            budget = self._reserve_provider_call("tavily")
            if not budget.get("allow", False):
                print("   [SKIP] Tavily信号: 免费额度保护触发，跳过本轮调用")
                return signals
            effective_limit = min(
                max(1, int(limit)),
                int(budget.get("max_results_per_call", CrawlerConfig.TAVILY_FREE_MAX_RESULTS_PER_CALL)),
            )

            payload = {
                "api_key": self.tavily_api_key,
                "query": (
                    f"{keyword} 公告 信披 问询函 停复牌 减持 增持 "
                    f"研报 评级 目标价 机构观点 资金流向 融资融券 北向资金 龙虎榜 主力资金"
                ),
                "topic": "finance",
                "search_depth": "basic",
                "max_results": max(3, min(12, effective_limit * 3)),
                "include_raw_content": False,
                "include_domains": [
                    "cninfo.com.cn",
                    "sse.com.cn",
                    "szse.cn",
                    "eastmoney.com",
                    "10jqka.com.cn",
                    "jrj.com.cn",
                    "cs.com.cn",
                ],
                "include_usage": True,
            }
            response = self._retry_json_post(
                CrawlerConfig.TAVILY_ENDPOINT,
                payload=payload,
                mode="search_api",
            )
            if not response:
                return signals

            data = response.json() or {}
            rows = data.get("results", [])
            if not isinstance(rows, list):
                rows = []

            seen = set()
            for item in rows:
                if len(signals) >= effective_limit:
                    break
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "")).strip()
                content = str(item.get("content", "")).strip()
                url = str(item.get("url", "")).strip()
                if not title and not content:
                    continue

                dedup_key = re.sub(r"\s+", "", f"{title}|{url}".lower())
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                score = item.get("score", 0.0)
                try:
                    score = float(score)
                except (TypeError, ValueError):
                    score = 0.0
                text = f"{title} {content}".strip()
                signal_type, category, base_rel = self._infer_signal_type(text)
                reliability = base_rel + max(0.0, min(0.10, 0.10 * score))
                signals.append({
                    "title": title or content[:80],
                    "content": content,
                    "url": url,
                    "source": "Tavily-Signal",
                    "publish_time": str(item.get("published_date", "")).strip(),
                    "data_source_type": "search_signal",
                    "signal_type": signal_type,
                    "reliability": round(reliability, 6),
                    "category": category,
                })

            print(
                f"   [OK] Tavily信号: {len(signals)} 条 "
                f"(run {budget.get('used_calls_this_run')}/{budget.get('allowed_calls_per_run')}, "
                f"month_left={budget.get('monthly_left')})"
            )
        except Exception as exc:
            print(f"   [WARN] Tavily信号获取失败: {str(exc)[:60]}")
        return signals
    
    def crawl_163(self, keyword: str, limit: int = 3) -> List[Dict]:
        """
        网易财经 - 综合财经
        """
        news_list = []
        try:
            url = f"https://news.163.com/special/0001386F/rank_keyword_new?keyword={keyword}"
            response = self._retry_request(url, mode="web_crawl")
            
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                items = soup.find_all('div', class_='news-item', limit=limit)
                for item in items:
                    title_tag = item.find('a')
                    if title_tag:
                        news_list.append({
                            'title': title_tag.get_text(strip=True),
                            'url': title_tag.get('href', ''),
                            'source': '网易财经',
                            'publish_time': '',
                            'data_source_type': 'news',
                            'reliability': 0.82,
                            'category': self._classify_news(title_tag.get_text(strip=True))
                        })
                print(f"   [OK] 网易财经: {len(news_list)} 条")
        except Exception as e:
            print(f"   [WARN] 网易财经爬取失败: {str(e)[:40]}")
        return news_list
    
    def crawl_sohu(self, keyword: str, limit: int = 3) -> List[Dict]:
        """
        搜狐财经 - 全面覆盖
        """
        news_list = []
        try:
            url = f"https://search.sohu.com/?query={keyword}&type=news"
            response = self._retry_request(url, mode="web_crawl")
            
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                items = soup.find_all('div', class_='news-item', limit=limit)
                for item in items:
                    title_tag = item.find('h3').find('a') if item.find('h3') else None
                    if title_tag:
                        news_list.append({
                            'title': title_tag.get_text(strip=True),
                            'url': title_tag.get('href', ''),
                            'source': '搜狐财经',
                            'publish_time': '',
                            'data_source_type': 'news',
                            'reliability': 0.80,
                            'category': self._classify_news(title_tag.get_text(strip=True))
                        })
                print(f"   [OK] 搜狐财经: {len(news_list)} 条")
        except Exception as e:
            print(f"   [WARN] 搜狐财经爬取失败: {str(e)[:40]}")
        return news_list
    
    def crawl_qq(self, keyword: str, limit: int = 3) -> List[Dict]:
        """
        腾讯财经 - 实时更新
        """
        news_list = []
        try:
            url = f"https://news.qq.com/search?query={keyword}&page=1"
            response = self._retry_request(url, mode="web_crawl")
            
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                items = soup.find_all('div', class_='news-item', limit=limit)
                for item in items:
                    title_tag = item.find('a')
                    if title_tag:
                        news_list.append({
                            'title': title_tag.get_text(strip=True),
                            'url': title_tag.get('href', ''),
                            'source': '腾讯财经',
                            'publish_time': '',
                            'data_source_type': 'news',
                            'reliability': 0.83,
                            'category': self._classify_news(title_tag.get_text(strip=True))
                        })
                print(f"   [OK] 腾讯财经: {len(news_list)} 条")
        except Exception as e:
            print(f"   [WARN] 腾讯财经爬取失败: {str(e)[:40]}")
        return news_list
    
    def crawl_jrj(self, keyword: str, limit: int = 3) -> List[Dict]:
        """
        金融界 - 专业财经
        """
        news_list = []
        try:
            url = f"https://so.jrj.com.cn/s?q={keyword}"
            response = self._retry_request(url, mode="web_crawl")
            
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                items = soup.find_all('div', class_='news-item', limit=limit)
                for item in items:
                    title_tag = item.find('a')
                    if title_tag:
                        news_list.append({
                            'title': title_tag.get_text(strip=True),
                            'url': title_tag.get('href', ''),
                            'source': '金融界',
                            'publish_time': '',
                            'data_source_type': 'news',
                            'reliability': 0.85,
                            'category': self._classify_news(title_tag.get_text(strip=True))
                        })
                print(f"   [OK] 金融界: {len(news_list)} 条")
        except Exception as e:
            print(f"   [WARN] 金融界爬取失败: {str(e)[:40]}")
        return news_list
    
    # ========== 官方机构数据源 ==========
    
    def crawl_sse(self, keyword: str, limit: int = 3) -> List[Dict]:
        """
        上海证券交易所 - 官方权威
        """
        news_list = []
        try:
            # 上交所官网公告API
            url = f"http://query.sse.com.cn/infodisplay/queryCommonInfo.do?isPagination=true&pageSize={limit}&pageNo=1&keyWord={keyword}"
            headers = {
                'Referer': 'http://www.sse.com.cn/',
                'Accept': 'application/json, text/javascript, */*; q=0.01'
            }
            response = self._retry_request(url, mode="official_api")
            
            if response:
                data = response.json()
                if data and 'result' in data:
                    for item in data['result'][:limit]:
                        news_list.append({
                            'title': item.get('title', item.get('noticeTitle', '')),
                            'url': f"http://www.sse.com.cn{item.get('url', '')}" if item.get('url') else '',
                            'source': '上海证券交易所',
                            'publish_time': item.get('publishTime', ''),
                            'data_source_type': 'official_announcement',
                            'reliability': 0.99,
                            'category': self._classify_news(item.get('title', ''))
                        })
                print(f"   [OK] 上交所: {len(news_list)} 条")
        except Exception as e:
            print(f"   [WARN] 上交所爬取失败: {str(e)[:40]}")
        return news_list
    
    def crawl_szse(self, keyword: str, limit: int = 3) -> List[Dict]:
        """
        深圳证券交易所 - 官方权威
        """
        news_list = []
        try:
            url = f"https://www.szse.cn/api/report/ShowReport?SHOWTYPE=JSON&CATALOGID=main_notice&TABKEY=tab1&PAGENO=1&PAGECOUNT={limit}&txtKeyword={keyword}"
            response = self._retry_request(url, mode="official_api")
            
            if response:
                data = response.json()
                if data and isinstance(data, list):
                    for item in data[:limit]:
                        news_list.append({
                            'title': item.get('TITLE', item.get('title', '')),
                            'url': f"https://www.szse.cn{item.get('URL', '')}" if item.get('URL') else '',
                            'source': '深圳证券交易所',
                            'publish_time': item.get('FILLDATE', ''),
                            'data_source_type': 'official_announcement',
                            'reliability': 0.99,
                            'category': self._classify_news(item.get('TITLE', ''))
                        })
                print(f"   [OK] 深交所: {len(news_list)} 条")
        except Exception as e:
            print(f"   [WARN] 深交所爬取失败: {str(e)[:40]}")
        return news_list
    
    def crawl_csrc(self, keyword: str, limit: int = 3) -> List[Dict]:
        """
        中国证券监督管理委员会 - 官方权威
        """
        news_list = []
        try:
            url = f"http://www.csrc.gov.cn/pub/newsite/xxpl/index.html"
            response = self._retry_request(url, mode="web_crawl")
            
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                items = soup.find_all('div', class_='news-item', limit=limit*2)
                for item in items[:limit]:
                    title_tag = item.find('a')
                    if title_tag:
                        title = title_tag.get_text(strip=True)
                        if keyword in title:
                            news_list.append({
                                'title': title,
                                'url': f"http://www.csrc.gov.cn{title_tag.get('href', '')}",
                                'source': '中国证监会',
                                'publish_time': '',
                                'data_source_type': 'official_announcement',
                                'reliability': 0.99,
                                'category': 'regulatory'
                            })
                print(f"   [OK] 证监会: {len(news_list)} 条")
        except Exception as e:
            print(f"   [WARN] 证监会爬取失败: {str(e)[:40]}")
        return news_list
    
    def crawl_pboc(self, keyword: str, limit: int = 3) -> List[Dict]:
        """
        中国人民银行 - 官方权威
        """
        news_list = []
        try:
            url = f"http://www.pbc.gov.cn/rmyh/105142/index.html"
            response = self._retry_request(url, mode="web_crawl")
            
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                items = soup.find_all('li', class_='news-item', limit=limit*2)
                for item in items[:limit]:
                    title_tag = item.find('a')
                    if title_tag:
                        title = title_tag.get_text(strip=True)
                        news_list.append({
                            'title': title,
                            'url': f"http://www.pbc.gov.cn{title_tag.get('href', '')}",
                            'source': '中国人民银行',
                            'publish_time': item.find('span', class_='time').get_text() if item.find('span', class_='time') else '',
                            'data_source_type': 'official_announcement',
                            'reliability': 0.98,
                            'category': 'regulatory'
                        })
                print(f"   [OK] 央行: {len(news_list)} 条")
        except Exception as e:
            print(f"   [WARN] 央行爬取失败: {str(e)[:40]}")
        return news_list
    
    def crawl_securities(self, keyword: str, limit: int = 3) -> List[Dict]:
        """
        证券时报 - 权威财经媒体
        """
        news_list = []
        try:
            url = f"https://www.securities.com.cn/api/search?keyword={keyword}&limit={limit}"
            response = self._retry_request(url, mode="official_api")
            
            if response:
                data = response.json()
                if data and 'list' in data:
                    for item in data['list'][:limit]:
                        news_list.append({
                            'title': item.get('title', ''),
                            'url': item.get('url', ''),
                            'source': '证券时报',
                            'publish_time': item.get('publish_time', ''),
                            'data_source_type': 'news',
                            'reliability': 0.92,
                            'category': self._classify_news(item.get('title', ''))
                        })
                print(f"   [OK] 证券时报: {len(news_list)} 条")
        except Exception as e:
            print(f"   [WARN] 证券时报爬取失败: {str(e)[:40]}")
        return news_list
    
    def crawl_cnstock(self, keyword: str, limit: int = 3) -> List[Dict]:
        """
        上海证券报 - 权威财经媒体
        """
        news_list = []
        try:
            url = f"https://www.cnstock.com/api/search?query={keyword}&count={limit}"
            response = self._retry_request(url, mode="official_api")
            
            if response:
                data = response.json()
                if data and 'items' in data:
                    for item in data['items'][:limit]:
                        news_list.append({
                            'title': item.get('title', ''),
                            'url': item.get('link', ''),
                            'source': '上海证券报',
                            'publish_time': item.get('time', ''),
                            'data_source_type': 'news',
                            'reliability': 0.93,
                            'category': self._classify_news(item.get('title', ''))
                        })
                print(f"   [OK] 上海证券报: {len(news_list)} 条")
        except Exception as e:
            print(f"   [WARN] 上海证券报爬取失败: {str(e)[:40]}")
        return news_list
    
    def crawl_cs(self, keyword: str, limit: int = 3) -> List[Dict]:
        """
        中国证券报 - 权威财经媒体
        """
        news_list = []
        try:
            url = f"https://www.cs.com.cn/api/search?keyword={keyword}&num={limit}"
            response = self._retry_request(url, mode="official_api")
            
            if response:
                data = response.json()
                if data and 'data' in data:
                    for item in data['data'][:limit]:
                        news_list.append({
                            'title': item.get('title', ''),
                            'url': item.get('url', ''),
                            'source': '中国证券报',
                            'publish_time': item.get('pubdate', ''),
                            'data_source_type': 'news',
                            'reliability': 0.93,
                            'category': self._classify_news(item.get('title', ''))
                        })
                print(f"   [OK] 中国证券报: {len(news_list)} 条")
        except Exception as e:
            print(f"   [WARN] 中国证券报爬取失败: {str(e)[:40]}")
        return news_list
    
    def _classify_news(self, title: str) -> str:
        """简单新闻分类"""
        title_lower = title.lower()

        # 信披/监管类
        if any(word in title_lower for word in ['公告', '信披', '问询函', '停牌', '复牌', '减持', '增持']):
            return 'disclosure'
        # 研报/评级类
        elif any(word in title_lower for word in ['研报', '评级', '目标价', '券商观点', '机构观点']):
            return 'research'
        # 资金流向类
        elif any(word in title_lower for word in ['资金流向', '融资融券', '北向资金', '龙虎榜', '主力资金']):
            return 'capital_flow'
        
        # 财务相关
        elif any(word in title_lower for word in ['净利润', '营收', '业绩', '财报', '盈利']):
            return 'financial'
        # 市场动态
        elif any(word in title_lower for word in ['股市', '大盘', '指数', '行情']):
            return 'market'
        # 公司新闻
        elif any(word in title_lower for word in ['公司', '集团', '企业', '公告']):
            return 'company'
        # 政策监管
        elif any(word in title_lower for word in ['政策', '监管', '央行', '证监会']):
            return 'regulatory'
        # 行业新闻
        elif any(word in title_lower for word in ['行业', '板块', '概念']):
            return 'industry'
        else:
            return 'general'
    
    # ========== 主爬取方法 ==========
    def crawl_news(self, keyword: str, sources: List[str] = None, limit_per_source: int = 3) -> Dict[str, Any]:
        """
        爬取新闻舆情 - 多源融合
        
        参数:
            keyword: 搜索关键词（如股票名称）
            sources: 指定数据源列表，None表示全部
            limit_per_source: 每个数据源爬取数量（默认3条）
        
        返回:
            包含所有新闻的字典
        """
        print(f"\n[INFO] 正在爬取新闻舆情: {keyword}")
        self._provider_calls_this_run = defaultdict(int)
        self._last_search_budget = {}
        
        # 默认数据源优先级
        # 规则：能走官方/公开API的优先走API；无API或API失败，再走爬取fallback。
        all_sources = [
            # ========== 官方机构 ==========
            {"id": "sse", "name": "上海证券交易所", "api_fetcher": self.crawl_sse, "crawl_fetcher": None},
            {"id": "szse", "name": "深圳证券交易所", "api_fetcher": self.crawl_szse, "crawl_fetcher": None},
            {"id": "csrc", "name": "中国证监会", "api_fetcher": None, "crawl_fetcher": self.crawl_csrc},
            {"id": "pboc", "name": "中国人民银行", "api_fetcher": None, "crawl_fetcher": self.crawl_pboc},

            # ========== 权威财经媒体 ==========
            {"id": "cnstock", "name": "上海证券报", "api_fetcher": self.crawl_cnstock, "crawl_fetcher": None},
            {"id": "cs", "name": "中国证券报", "api_fetcher": self.crawl_cs, "crawl_fetcher": None},
            {"id": "securities", "name": "证券时报", "api_fetcher": self.crawl_securities, "crawl_fetcher": None},
            {"id": "akshare", "name": "AKShare个股新闻", "api_fetcher": self.crawl_akshare_news, "crawl_fetcher": None},
            {"id": "serpapi_signal", "name": "SerpApi多类型信号", "api_fetcher": self.crawl_serpapi_signals, "crawl_fetcher": None},
            {"id": "serpapi", "name": "SerpApi新闻检索", "api_fetcher": self.crawl_serpapi_news, "crawl_fetcher": None},
            {"id": "tavily_signal", "name": "Tavily多类型信号", "api_fetcher": self.crawl_tavily_signals, "crawl_fetcher": None},
            {"id": "tavily", "name": "Tavily新闻检索", "api_fetcher": self.crawl_tavily_news, "crawl_fetcher": None},

            # ========== 商业媒体 ==========
            {"id": "eastmoney", "name": "东方财富", "api_fetcher": self.crawl_eastmoney, "crawl_fetcher": None},
            {"id": "sina", "name": "新浪财经", "api_fetcher": self.crawl_sina, "crawl_fetcher": None},
            {"id": "qq", "name": "腾讯财经", "api_fetcher": None, "crawl_fetcher": self.crawl_qq},
            {"id": "jrj", "name": "金融界", "api_fetcher": None, "crawl_fetcher": self.crawl_jrj},
            {"id": "163", "name": "网易财经", "api_fetcher": None, "crawl_fetcher": self.crawl_163},
            {"id": "sohu", "name": "搜狐财经", "api_fetcher": None, "crawl_fetcher": self.crawl_sohu},
        ]
        
        # 筛选指定数据源
        if sources:
            all_sources = [s for s in all_sources if s["id"] in sources]
        
        # 爬取各数据源
        all_news = []
        success_count = 0
        fail_count = 0
        source_runtime: List[Dict[str, Any]] = []
        
        for source in all_sources:
            source_id = source["id"]
            source_name = source["name"]
            api_fetcher = source.get("api_fetcher")
            crawl_fetcher = source.get("crawl_fetcher")
            source_limit = self._resolve_source_limit(source_id, limit_per_source)

            print(f"   [INFO]  正在抓取 {source_name} (API优先, limit={source_limit})...")
            news, runtime = self._fetch_source_api_first(
                source_id=source_id,
                source_name=source_name,
                keyword=keyword,
                limit=source_limit,
                api_fetcher=api_fetcher,
                crawl_fetcher=crawl_fetcher,
            )
            source_runtime.append(runtime)
            
            if news:
                all_news.extend(news)
                success_count += 1
                print(f"   [OK] {source_name} 命中: mode={runtime.get('mode')} count={runtime.get('count')}")
            else:
                fail_count += 1
                print(f"   [ERR] {source_name} 失败: mode={runtime.get('mode')}")
        
        # 数据清洗和去重
        cleaned_news = self._clean_and_dedup(all_news)
        
        # 综合结果
        result = {
            'keyword': keyword,
            'timestamp': datetime.now().isoformat(),
            'api_first': True,
            'total_news': len(cleaned_news),
            'sources_count': len(all_sources),
            'success_sources': success_count,
            'failed_sources': fail_count,
            'source_runtime': source_runtime,
            'search_api_budget': dict(self._last_search_budget),
            'news_list': cleaned_news,
            'top_news': self._get_top_news(cleaned_news, 10)
        }

        self.last_source_runtime = source_runtime
        
        print(f"\n[OK] 爬取完成！共获取 {len(cleaned_news)} 条新闻（去重后）")
        print(f"   [OK] 成功: {success_count} 个数据源")
        print(f"   [ERR] 失败: {fail_count} 个数据源")
        
        return result

    def get_last_source_runtime(self) -> List[Dict[str, Any]]:
        """返回最近一次 crawl_news 的分源运行信息。"""
        return list(self.last_source_runtime)

    def get_last_search_budget(self) -> Dict[str, Any]:
        """返回最近一次 crawl_news 的搜索API预算快照。"""
        return dict(self._last_search_budget)
    
    def _clean_and_dedup(self, news_list: List[Dict]) -> List[Dict]:
        """数据清洗和去重"""
        if not news_list:
            return []
        
        # 移除标题为空的
        cleaned = [n for n in news_list if n.get('title', '').strip()]
        
        # 去重：
        # 1) 有URL优先按URL去重（保留不同信号类型的同标题记录）
        # 2) 无URL时按 标题+源类型 去重
        seen_keys = set()
        unique_news = []
        
        for news in cleaned:
            title = news['title'].strip()
            # 去除标题中的特殊字符用于比较
            normalized_title = re.sub(r'[【】「」《》\s]+', '', title)

            url = str(news.get("url", "") or "").strip().lower()
            source_type = str(news.get("data_source_type", "news") or "news").strip().lower()
            if url:
                dedup_key = f"url::{url}"
            else:
                dedup_key = f"title::{normalized_title}::{source_type}"

            if normalized_title and dedup_key not in seen_keys:
                seen_keys.add(dedup_key)
                unique_news.append(news)
        
        # 按可信度排序
        unique_news.sort(key=lambda x: x.get('reliability', 0.5), reverse=True)
        
        return unique_news
    
    def _get_top_news(self, news_list: List[Dict], limit: int = 10) -> List[Dict]:
        """获取最重要的新闻"""
        if not news_list:
            return []
        
        # 优先选择高可信度 + 相关分类
        prioritized = sorted(
            news_list,
            key=lambda x: (
                x.get('reliability', 0.5),
                1 if x.get('category') in [
                    'financial',
                    'market',
                    'company',
                    'disclosure',
                    'research',
                    'capital_flow',
                ] else 0
            ),
            reverse=True
        )
        
        return prioritized[:limit]

# ==================== 便捷函数 ====================
def crawl_stock_news(stock_name: str, limit_per_source: int = 3) -> Dict:
    """
    便捷函数：爬取股票新闻舆情
    
    参数:
        stock_name: 股票名称
        limit_per_source: 每个网站爬取数量（默认3条）
    
    返回:
        舆情数据字典
    """
    crawler = NewsCrawler()
    return crawler.crawl_news(stock_name, limit_per_source=limit_per_source)

# ==================== 测试 ====================
if __name__ == "__main__":
    print("="*60)
    print("[START] 新闻爬虫优化版 - 测试运行")
    print("="*60)
    
    crawler = NewsCrawler()
    
    # 测试爬取贵州茅台新闻
    print("\n[INFO] 测试爬取: 贵州茅台")
    result = crawler.crawl_news("贵州茅台", limit_per_source=3)
    
    print(f"\n[DATA] 爬取结果:")
    print(f"   关键词: {result['keyword']}")
    print(f"   新闻总数: {result['total_news']}")
    print(f"   数据源: {result['success_sources']}/{result['sources_count']}")
    
    print("\n[NEWS] 头条新闻:")
    for i, news in enumerate(result['top_news'][:5], 1):
        print(f"   {i}. [{news['source']}] {news['title']}")
        print(f"      分类: {news['category']} | 可信度: {news['reliability']:.2f}")
    
    print("\n[OK] 测试完成！")


