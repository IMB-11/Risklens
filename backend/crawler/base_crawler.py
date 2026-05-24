import asyncio
import aiohttp
import time
import hashlib
import random
import re
import json
import zlib
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

# ==================== 常量定义 ====================
DEFAULT_MAX_REQUESTS_PER_MINUTE = 30
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRY = 3
CACHE_CLEANUP_INTERVAL = 3600  # 每小时清理一次缓存
MAX_CACHE_SIZE = 50000  # 最大缓存条目数

# ==================== 全局状态管理 ====================
_global_url_cache = set()
_global_url_cache_lock = asyncio.Lock()
_global_content_fingerprints = {}  # SimHash指纹缓存

# ==================== 枚举类型 ====================
class CrawlStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    BLOCKED = "blocked"
    RETRY = "retry"

class SourceType(Enum):
    DISCLOSURE = "disclosure"      # 官方披露
    EXCHANGE = "exchange"          # 交易所
    REGULATOR = "regulator"        # 监管机构
    CENTRAL_BANK = "central_bank"  # 央行
    STATISTICS = "statistics"      # 统计数据
    NEWS = "news"                  # 新闻媒体
    REUTERS = "reuters"            # 路透社
    BLOOMBERG = "bloomberg"        # 彭博
    SOCIAL = "social"              # 社交媒体
    UNKNOWN = "unknown"            # 未知

# ==================== 数据类 ====================
@dataclass
class CrawlResult:
    """爬取结果封装"""
    status: CrawlStatus
    content: Optional[str] = None
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    status_code: Optional[int] = None
    response_time: float = 0.0
    retry_count: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProxyInfo:
    """代理信息"""
    ip: str
    port: int
    protocol: str = "http"
    username: Optional[str] = None
    password: Optional[str] = None
    country: Optional[str] = None
    response_time: float = 0.0
    success_rate: float = 1.0
    last_used: Optional[datetime] = None
    is_healthy: bool = True
    
    @property
    def proxy_url(self) -> str:
        if self.username and self.password:
            return f"{self.protocol}://{self.username}:{self.password}@{self.ip}:{self.port}"
        return f"{self.protocol}://{self.ip}:{self.port}"

@dataclass
class CrawlMetrics:
    """爬虫性能指标"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    bytes_downloaded: int = 0
    cache_hits: int = 0
    duplicates_found: int = 0
    rate_limit_triggers: int = 0
    circuit_breaker_triggers: int = 0
    proxy_switches: int = 0
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def avg_response_time(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_response_time / self.successful_requests
    
    def record_success(self, response_time: float, bytes_downloaded: int):
        self.total_requests += 1
        self.successful_requests += 1
        self.total_response_time += response_time
        self.bytes_downloaded += bytes_downloaded
    
    def record_failure(self):
        self.total_requests += 1
        self.failed_requests += 1
    
    def reset(self):
        self.__init__()

# ==================== 智能限流控制器 ====================
class SmartRateLimiter:
    """智能限流控制器：基于令牌桶算法，支持动态调整速率"""
    
    def __init__(self, max_requests_per_minute: int = 30):
        self.max_requests = max_requests_per_minute
        self.tokens = max_requests_per_minute
        self.last_refill_time = datetime.now()
        self.lock = asyncio.Lock()
        self.adjustment_factor = 1.0  # 动态调整因子
        self.min_rate = 5  # 最小速率
        self.max_rate = 100  # 最大速率
    
    async def acquire(self, count: int = 1) -> bool:
        """获取请求许可"""
        async with self.lock:
            now = datetime.now()
            self._refill_tokens(now)
            
            effective_rate = int(self.max_requests * self.adjustment_factor)
            
            if self.tokens >= count:
                self.tokens -= count
                return True
            
            # 计算等待时间
            wait_time = (count - self.tokens) * 60 / effective_rate + 1
            await asyncio.sleep(wait_time)
            self.tokens = max(0, self.tokens - count)
            return True
    
    def _refill_tokens(self, now: datetime):
        """补充令牌"""
        elapsed = (now - self.last_refill_time).total_seconds()
        if elapsed >= 60:
            effective_rate = int(self.max_requests * self.adjustment_factor)
            self.tokens = min(effective_rate, self.tokens + effective_rate)
            self.last_refill_time = now
    
    def adjust_rate(self, factor: float):
        """动态调整速率"""
        self.adjustment_factor = max(
            self.min_rate / self.max_requests,
            min(self.max_rate / self.max_requests, factor)
        )
    
    @property
    def current_rate(self) -> float:
        """当前请求速率（每分钟）"""
        return self.max_requests * self.adjustment_factor

# ==================== 智能熔断机制 ====================
class SmartCircuitBreaker:
    """智能熔断机制：支持多种熔断策略和自动恢复"""
    
    class BreakerState(Enum):
        CLOSED = "closed"      # 正常运行
        OPEN = "open"          # 熔断中
        HALF_OPEN = "half_open"  # 试探恢复
    
    def __init__(self, failure_threshold: float = 0.5, 
                 sample_size: int = 20, cool_down_minutes: int = 5,
                 recovery_samples: int = 5):
        self.failure_threshold = failure_threshold
        self.sample_size = sample_size
        self.cool_down_minutes = cool_down_minutes
        self.recovery_samples = recovery_samples
        
        self.success_count = 0
        self.failure_count = 0
        self.consecutive_failures = 0
        self.last_failure_time = None
        self.state = self.BreakerState.CLOSED
        self.recovery_successes = 0
        self.lock = asyncio.Lock()
    
    async def record_success(self):
        """记录成功请求"""
        async with self.lock:
            self.consecutive_failures = 0
            
            if self.state == self.BreakerState.HALF_OPEN:
                self.recovery_successes += 1
                if self.recovery_successes >= self.recovery_samples:
                    self._reset()
                    print("🟢 熔断已完全恢复")
            
            if self.state == self.BreakerState.CLOSED:
                self.success_count += 1
                self._trim_history()
    
    async def record_failure(self):
        """记录失败请求"""
        async with self.lock:
            self.consecutive_failures += 1
            self.last_failure_time = datetime.now()
            
            if self.state == self.BreakerState.CLOSED:
                self.failure_count += 1
                self._trim_history()
                self._check_threshold()
    
    def _trim_history(self):
        """保持样本大小"""
        total = self.success_count + self.failure_count
        if total > self.sample_size:
            ratio = self.sample_size / total
            self.success_count = int(self.success_count * ratio)
            self.failure_count = int(self.failure_count * ratio)
    
    def _check_threshold(self):
        """检查是否需要熔断"""
        total = self.success_count + self.failure_count
        if total >= self.sample_size:
            failure_rate = self.failure_count / total
            if failure_rate >= self.failure_threshold or self.consecutive_failures >= 5:
                self.state = self.BreakerState.OPEN
                print(f"🔴 熔断触发！失败率: {failure_rate:.2%}, 连续失败: {self.consecutive_failures}")
    
    def _attempt_reset(self) -> bool:
        """尝试进入试探状态"""
        if self.state == self.BreakerState.OPEN:
            if self.last_failure_time and (datetime.now() - self.last_failure_time).total_seconds() >= self.cool_down_minutes * 60:
                self.state = self.BreakerState.HALF_OPEN
                self.recovery_successes = 0
                print("🟡 进入熔断试探状态")
                return True
        return False
    
    def _reset(self):
        """完全重置熔断"""
        self.state = self.BreakerState.CLOSED
        self.success_count = 0
        self.failure_count = 0
        self.consecutive_failures = 0
        self.recovery_successes = 0
    
    def should_allow(self) -> bool:
        """检查是否允许请求"""
        if self.state == self.BreakerState.CLOSED:
            return True
        elif self.state == self.BreakerState.HALF_OPEN:
            return True  # 允许试探请求
        else:
            self._attempt_reset()
            return self.state == self.BreakerState.HALF_OPEN
    
    @property
    def failure_rate(self) -> float:
        """当前失败率"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.failure_count / total

# ==================== 智能代理池 ====================
class SmartProxyPool:
    """智能代理池：自动健康检测和负载均衡"""
    
    def __init__(self, auto_discover: bool = True):
        self.proxies: List[ProxyInfo] = []
        self.current_index = 0
        self.lock = asyncio.Lock()
        self.health_check_interval = 300  # 每5分钟检查一次健康
        self._health_task = None
        self._discovery_task = None
        self.auto_discover = auto_discover
        self.proxy_sources = [
            'https://free-proxy-list.net/',
            'https://www.sslproxies.org/',
            'https://www.us-proxy.org/',
            'https://www.socks-proxy.net/'
        ]
        self.discovery_interval = 3600  # 每小时发现新代理
    
    def add_proxy(self, proxy: ProxyInfo):
        """添加代理"""
        # 避免重复
        if not any(p.ip == proxy.ip and p.port == proxy.port for p in self.proxies):
            self.proxies.append(proxy)
    
    def add_proxies(self, proxies: List[ProxyInfo]):
        """批量添加代理"""
        for proxy in proxies:
            self.add_proxy(proxy)
    
    async def get_proxy(self, strategy: str = "intelligent") -> Optional[ProxyInfo]:
        """获取可用代理"""
        async with self.lock:
            healthy_proxies = [p for p in self.proxies if p.is_healthy]
            
            if not healthy_proxies:
                return None
            
            if strategy == "round_robin":
                proxy = self._get_round_robin(healthy_proxies)
            elif strategy == "least_used":
                proxy = self._get_least_used(healthy_proxies)
            elif strategy == "fastest":
                proxy = self._get_fastest(healthy_proxies)
            elif strategy == "intelligent":
                proxy = self._get_intelligent(healthy_proxies)
            else:
                proxy = random.choice(healthy_proxies)
            
            proxy.last_used = datetime.now()
            return proxy
    
    def _get_round_robin(self, proxies: List[ProxyInfo]) -> ProxyInfo:
        """轮询选择"""
        proxy = proxies[self.current_index % len(proxies)]
        self.current_index += 1
        return proxy
    
    def _get_least_used(self, proxies: List[ProxyInfo]) -> ProxyInfo:
        """选择最少使用的"""
        return min(proxies, key=lambda p: p.last_used or datetime.min)
    
    def _get_fastest(self, proxies: List[ProxyInfo]) -> ProxyInfo:
        """选择响应最快的"""
        return min(proxies, key=lambda p: p.response_time)
    
    def _get_intelligent(self, proxies: List[ProxyInfo]) -> ProxyInfo:
        """智能选择：综合成功率、响应时间和使用频率"""
        now = datetime.now()
        best_proxy = None
        best_score = float('-inf')
        
        for proxy in proxies:
            # 成功率权重 40%
            success_score = proxy.success_rate * 0.4
            
            # 响应时间评分 30%（越快越好）
            response_score = max(0, (1 - proxy.response_time / 5) * 0.3)
            
            # 新鲜度评分 30%（最近使用的优先）
            if proxy.last_used:
                age_hours = (now - proxy.last_used).total_seconds() / 3600
                freshness_score = max(0, (1 - age_hours / 24) * 0.3)
            else:
                freshness_score = 0.3
            
            total_score = success_score + response_score + freshness_score
            
            if total_score > best_score:
                best_score = total_score
                best_proxy = proxy
        
        return best_proxy or random.choice(proxies)
    
    def mark_failure(self, proxy: ProxyInfo):
        """标记代理失败"""
        proxy.success_rate = max(0, proxy.success_rate - 0.15)
        proxy.consecutive_failures = getattr(proxy, 'consecutive_failures', 0) + 1
        
        if proxy.success_rate < 0.5 or proxy.consecutive_failures >= 3:
            proxy.is_healthy = False
            print(f"❌ 代理 {proxy.ip}:{proxy.port} 被标记为不健康 (成功率: {proxy.success_rate:.1%}, 连续失败: {proxy.consecutive_failures})")
    
    def mark_success(self, proxy: ProxyInfo, response_time: float):
        """标记代理成功"""
        proxy.success_rate = min(1.0, proxy.success_rate + 0.08)
        proxy.response_time = response_time
        proxy.is_healthy = True
        proxy.consecutive_failures = 0
    
    async def start_health_check(self):
        """启动健康检查任务"""
        if self._health_task:
            return
        
        async def check_loop():
            while True:
                await self._perform_health_check()
                await asyncio.sleep(self.health_check_interval)
        
        self._health_task = asyncio.create_task(check_loop())
        print("🔍 代理健康检查任务已启动")
        
        # 如果启用自动发现，启动发现任务
        if self.auto_discover:
            await self._start_discovery()
    
    async def _start_discovery(self):
        """启动代理自动发现任务"""
        if self._discovery_task:
            return
        
        async def discover_loop():
            while True:
                await self._discover_proxies()
                await asyncio.sleep(self.discovery_interval)
        
        self._discovery_task = asyncio.create_task(discover_loop())
        print("🌐 代理自动发现任务已启动")
    
    async def _discover_proxies(self):
        """自动发现免费代理"""
        discovered_count = 0
        for source_url in self.proxy_sources:
            try:
                result = await self._fetch_proxy_list(source_url)
                discovered_count += len(result)
                self.add_proxies(result)
            except Exception as e:
                print(f"⚠️ 代理发现失败 [{source_url}]: {e}")
        
        if discovered_count > 0:
            print(f"✅ 发现 {discovered_count} 个新代理")
    
    async def _fetch_proxy_list(self, url: str) -> List[ProxyInfo]:
        """从代理列表页面抓取代理"""
        proxies = []
        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        # 简单的代理IP:PORT提取
                        matches = re.findall(r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?):(\d{2,5})\b', html)
                        for match in matches[:20]:
                            ip_port = match.group(0) if hasattr(match, 'group') else match
                            if ':' in ip_port:
                                ip, port = ip_port.split(':')
                                proxies.append(ProxyInfo(
                                    ip=ip.strip(),
                                    port=int(port.strip()),
                                    protocol='http',
                                    response_time=float('inf'),
                                    success_rate=0.5,
                                    is_healthy=False
                                ))
        except Exception:
            pass
        return proxies
    
    async def _perform_health_check(self):
        """执行健康检查"""
        if not self.proxies:
            return
        
        check_tasks = []
        for proxy in self.proxies:
            check_tasks.append(self._check_proxy_health(proxy))
        
        await asyncio.gather(*check_tasks)
    
    async def _check_proxy_health(self, proxy: ProxyInfo):
        """检查单个代理健康状态"""
        try:
            start_time = time.time()
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    "http://httpbin.org/ip",
                    proxy=proxy.proxy_url,
                    timeout=15
                ) as response:
                    response_time = time.time() - start_time
                    if response.status == 200:
                        proxy.is_healthy = True
                        proxy.success_rate = min(1.0, proxy.success_rate + 0.1)
                        proxy.response_time = response_time
                    else:
                        proxy.is_healthy = False
                        proxy.success_rate = max(0, proxy.success_rate - 0.1)
        except Exception as e:
            proxy.is_healthy = False
            proxy.success_rate = max(0, proxy.success_rate - 0.15)
    
    @property
    def healthy_count(self) -> int:
        """健康代理数量"""
        return sum(1 for p in self.proxies if p.is_healthy)
    
    @property
    def stats(self) -> Dict[str, Any]:
        """获取代理池统计信息"""
        if not self.proxies:
            return {'total': 0, 'healthy': 0, 'avg_response_time': 0, 'avg_success_rate': 0}
        
        healthy = [p for p in self.proxies if p.is_healthy]
        return {
            'total': len(self.proxies),
            'healthy': len(healthy),
            'avg_response_time': sum(p.response_time for p in healthy) / max(len(healthy), 1),
            'avg_success_rate': sum(p.success_rate for p in self.proxies) / len(self.proxies)
        }

# ==================== SimHash去重器 ====================
class SimHashDeduplicator:
    """基于SimHash的近似去重器：能识别相似但不完全相同的内容"""
    
    def __init__(self, hash_bits: int = 64, similarity_threshold: float = 0.95, max_fingerprints: int = 50000):
        self.hash_bits = hash_bits
        self.threshold = similarity_threshold
        self.max_fingerprints = max_fingerprints
        self.fingerprints: Dict[int, List[str]] = {}  # 指纹 -> URL列表
        self.fingerprint_timestamps: Dict[int, datetime] = {}  # 指纹 -> 添加时间
        self.lock = asyncio.Lock()
        self.hash_table = {}  # 分桶哈希表，用于加速相似搜索
        self.bucket_size = 4  # 每个桶的位数
    
    def _hash_func(self, content: str) -> int:
        """生成内容的哈希值（混合多种哈希算法）"""
        hash1 = zlib.adler32(content.encode('utf-8')) & 0xffffffff
        hash2 = int(hashlib.md5(content.encode('utf-8')).hexdigest(), 16) & 0xffffffff
        return hash1 ^ (hash2 >> 16)
    
    def _get_features(self, content: str, n_grams: int = 3) -> List[str]:
        """提取内容特征（多级N-gram）"""
        content = content.lower()
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'[^\w\s\u4e00-\u9fff]', '', content)  # 保留中文和基本字符
        
        features = set()  # 使用set避免重复特征
        
        # 字符级N-gram
        for i in range(len(content) - n_grams + 1):
            features.add(content[i:i+n_grams])
        
        # 双字符N-gram
        for i in range(len(content) - 2 + 1):
            features.add(content[i:i+2])
        
        # 添加单词级特征
        words = content.split()
        for i in range(len(words) - 1):
            features.add(f"{words[i]} {words[i+1]}")
        for i in range(len(words) - 2):
            features.add(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        # 添加标题特征（如果存在）
        if len(words) > 0:
            features.add(words[0])  # 首词
            features.add(words[-1])  # 尾词
        
        return list(features)[:2000]  # 限制特征数量
    
    def _compute_simhash(self, content: str) -> int:
        """计算SimHash值（加权版本）"""
        features = self._get_features(content)
        
        if not features:
            return 0
        
        v = [0] * self.hash_bits
        
        for i, feature in enumerate(features):
            h = self._hash_func(feature)
            # 特征权重：位置越靠前权重越高
            weight = 1.0 - (i / len(features) * 0.3)
            
            for j in range(self.hash_bits):
                bit = (h >> j) & 1
                if bit == 1:
                    v[j] += weight
                else:
                    v[j] -= weight
        
        fingerprint = 0
        for i in range(self.hash_bits):
            if v[i] > 0:
                fingerprint |= (1 << i)
        
        return fingerprint
    
    def _hamming_distance(self, hash1: int, hash2: int) -> int:
        """计算汉明距离"""
        return bin(hash1 ^ hash2).count('1')
    
    def _is_similar(self, hash1: int, hash2: int) -> bool:
        """判断两个指纹是否相似"""
        distance = self._hamming_distance(hash1, hash2)
        similarity = 1 - (distance / self.hash_bits)
        return similarity >= self.threshold
    
    def _get_bucket_keys(self, fingerprint: int) -> List[int]:
        """获取指纹所属的桶键（用于分桶搜索）"""
        buckets = []
        for i in range(0, self.hash_bits, self.bucket_size):
            mask = ((1 << self.bucket_size) - 1) << i
            bucket_key = (fingerprint & mask) >> i
            buckets.append(bucket_key)
        return buckets
    
    async def is_duplicate(self, content: str, url: str) -> Tuple[bool, Optional[str]]:
        """
        检查内容是否重复（优化版本，使用分桶加速）
        
        返回:
            (是否重复, 重复的URL)
        """
        async with self.lock:
            # 首先检查URL是否完全相同（快速路径）
            for urls in self.fingerprints.values():
                if url in urls:
                    return True, url
            
            fingerprint = self._compute_simhash(content)
            
            # 使用分桶搜索加速相似检测
            candidates = set()
            bucket_keys = self._get_bucket_keys(fingerprint)
            
            for i, bucket_key in enumerate(bucket_keys):
                bucket_key_with_pos = (i, bucket_key)
                if bucket_key_with_pos in self.hash_table:
                    candidates.update(self.hash_table[bucket_key_with_pos])
            
            # 检查候选指纹
            for existing_fp in candidates:
                if self._is_similar(fingerprint, existing_fp):
                    existing_urls = self.fingerprints.get(existing_fp, [])
                    return True, existing_urls[0] if existing_urls else None
            
            # 添加新指纹到分桶表
            for i, bucket_key in enumerate(bucket_keys):
                bucket_key_with_pos = (i, bucket_key)
                if bucket_key_with_pos not in self.hash_table:
                    self.hash_table[bucket_key_with_pos] = set()
                self.hash_table[bucket_key_with_pos].add(fingerprint)
            
            # 添加新指纹
            if fingerprint not in self.fingerprints:
                self.fingerprints[fingerprint] = []
            self.fingerprints[fingerprint].append(url)
            self.fingerprint_timestamps[fingerprint] = datetime.now()
            
            # 维护最大容量
            await self._enforce_capacity()
            
            return False, None
    
    async def _enforce_capacity(self):
        """强制维护最大容量限制（LRU策略）"""
        if len(self.fingerprints) <= self.max_fingerprints:
            return
        
        # 按时间排序并删除最老的
        sorted_fps = sorted(
            self.fingerprints.keys(),
            key=lambda fp: self.fingerprint_timestamps.get(fp, datetime.min)
        )
        
        # 删除前20%最老的指纹
        to_delete = sorted_fps[:int(len(sorted_fps) * 0.2)]
        
        for fp in to_delete:
            # 从分桶表中删除
            bucket_keys = self._get_bucket_keys(fp)
            for i, bucket_key in enumerate(bucket_keys):
                bucket_key_with_pos = (i, bucket_key)
                if bucket_key_with_pos in self.hash_table:
                    self.hash_table[bucket_key_with_pos].discard(fp)
            
            # 从主表中删除
            del self.fingerprints[fp]
            del self.fingerprint_timestamps[fp]
        
        print(f"🧹 清理了 {len(to_delete)} 个过期指纹")
    
    def clear(self):
        """清空指纹库"""
        self.fingerprints.clear()
        self.fingerprint_timestamps.clear()
        self.hash_table.clear()
    
    @property
    def stats(self) -> Dict[str, Any]:
        """获取去重器统计信息"""
        total_urls = sum(len(urls) for urls in self.fingerprints.values())
        return {
            'fingerprints': len(self.fingerprints),
            'total_urls': total_urls,
            'buckets': len(self.hash_table),
            'avg_urls_per_fingerprint': total_urls / max(len(self.fingerprints), 1)
        }
    
    async def get_similar_content(self, content: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        查找相似内容
        
        返回:
            [(URL, 相似度), ...]
        """
        async with self.lock:
            fingerprint = self._compute_simhash(content)
            results = []
            
            for existing_fp, urls in self.fingerprints.items():
                if existing_fp == fingerprint:
                    continue
                
                distance = self._hamming_distance(fingerprint, existing_fp)
                similarity = 1 - (distance / self.hash_bits)
                
                if similarity >= 0.7:  # 较低阈值用于相似搜索
                    results.append((urls[0], similarity))
            
            results.sort(key=lambda x: -x[1])
            return results[:top_n]

# ==================== 增量爬取管理器 ====================
class IncrementalCrawlManager:
    """增量爬取管理器：基于时间戳的增量更新"""
    
    def __init__(self, storage_path: str = "./crawl_state.json"):
        self.storage_path = storage_path
        self.crawl_state: Dict[str, Dict[str, Any]] = {}  # source -> {last_crawl, last_modified, urls}
        self.lock = asyncio.Lock()
        self._load_state()
    
    def _load_state(self):
        """加载爬取状态"""
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                self.crawl_state = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.crawl_state = {}
    
    def _save_state(self):
        """保存爬取状态"""
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.crawl_state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 保存爬取状态失败: {e}")
    
    async def get_last_crawl_time(self, source: str) -> Optional[datetime]:
        """获取上次爬取时间"""
        async with self.lock:
            state = self.crawl_state.get(source)
            if state and state.get('last_crawl'):
                return datetime.fromisoformat(state['last_crawl'])
            return None
    
    async def update_crawl_time(self, source: str):
        """更新爬取时间"""
        async with self.lock:
            if source not in self.crawl_state:
                self.crawl_state[source] = {
                    'last_crawl': None,
                    'urls': {}
                }
            self.crawl_state[source]['last_crawl'] = datetime.now().isoformat()
            self._save_state()
    
    async def is_url_new(self, source: str, url: str, last_modified: Optional[str] = None) -> bool:
        """检查URL是否为新内容"""
        async with self.lock:
            if source not in self.crawl_state:
                self.crawl_state[source] = {
                    'last_crawl': None,
                    'urls': {}
                }
            
            url_info = self.crawl_state[source]['urls'].get(url)
            
            # URL不存在，是新的
            if not url_info:
                self.crawl_state[source]['urls'][url] = {
                    'last_modified': last_modified or datetime.now().isoformat(),
                    'crawled_at': datetime.now().isoformat()
                }
                self._save_state()
                return True
            
            # URL存在但有更新
            if last_modified and url_info['last_modified'] != last_modified:
                self.crawl_state[source]['urls'][url]['last_modified'] = last_modified
                self.crawl_state[source]['urls'][url]['crawled_at'] = datetime.now().isoformat()
                self._save_state()
                return True
            
            return False
    
    async def get_url_state(self, source: str, url: str) -> Optional[Dict[str, Any]]:
        """获取URL的爬取状态"""
        async with self.lock:
            return self.crawl_state.get(source, {}).get('urls', {}).get(url)
    
    async def cleanup_old_entries(self, days_to_keep: int = 30):
        """清理过期的爬取记录"""
        async with self.lock:
            cutoff = datetime.now() - timedelta(days=days_to_keep)
            
            for source, state in self.crawl_state.items():
                urls = state.get('urls', {})
                old_urls = [
                    url for url, info in urls.items()
                    if datetime.fromisoformat(info['crawled_at']) < cutoff
                ]
                
                for url in old_urls:
                    del urls[url]
            
            self._save_state()

# ==================== 基础爬虫类 ====================
class BaseCrawler(ABC):
    def __init__(self, name: str, timeout: int = DEFAULT_TIMEOUT, 
                 max_retry: int = DEFAULT_MAX_RETRY, 
                 max_requests_per_minute: int = DEFAULT_MAX_REQUESTS_PER_MINUTE):
        self.name = name
        self.timeout = timeout
        self.max_retry = max_retry
        
        # 高级组件
        self.rate_limiter = SmartRateLimiter(max_requests_per_minute)
        self.circuit_breaker = SmartCircuitBreaker()
        self.proxy_pool = SmartProxyPool()
        self.deduplicator = SimHashDeduplicator()
        self.incremental_manager = IncrementalCrawlManager()
        self.metrics = CrawlMetrics()
        
        # 请求头池（随机轮换）
        self.headers_pool = [
            {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Referer': 'https://www.baidu.com/',
                'Accept-Encoding': 'gzip, deflate, br'
            },
            {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9',
                'Referer': 'https://www.sogou.com/',
                'Accept-Encoding': 'gzip, deflate'
            },
            {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.5',
                'Referer': 'https://www.bing.com/',
                'Accept-Encoding': 'gzip, deflate, br'
            },
            {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Referer': 'https://www.google.com/',
                'Accept-Encoding': 'gzip, deflate, br'
            },
            {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.5',
                'Referer': 'https://www.baidu.com/',
                'Accept-Encoding': 'gzip, deflate'
            }
        ]
        
        # 会话管理器
        self._session = None
        self._session_created_at = None
        self._session_lock = asyncio.Lock()
        
        # 随机延迟配置
        self.min_delay = 0.5
        self.max_delay = 2.0
    
    @property
    def headers(self) -> Dict[str, str]:
        """随机获取请求头"""
        return random.choice(self.headers_pool)
    
    async def _get_session(self, use_proxy: bool = False) -> aiohttp.ClientSession:
        """获取或创建aiohttp会话"""
        async with self._session_lock:
            now = datetime.now()
            
            # 每10分钟重建会话
            if self._session is None or (now - self._session_created_at).total_seconds() > 600:
                if self._session:
                    await self._session.close()
                
                timeout = aiohttp.ClientTimeout(
                    total=self.timeout,
                    connect=10,
                    sock_read=15,
                    sock_connect=10
                )
                
                connector = aiohttp.TCPConnector(
                    limit=15,
                    limit_per_host=5,
                    ttl_dns_cache=300,
                    force_close=False,
                    ssl=False
                )
                
                self._session = aiohttp.ClientSession(
                    headers=self.headers,
                    timeout=timeout,
                    connector=connector,
                    cookie_jar=aiohttp.CookieJar(unsafe=True),
                    auto_decompress=True
                )
                self._session_created_at = now
                print(f"🔄 重建HTTP会话")
            
            return self._session
    
    async def _apply_random_delay(self):
        """应用随机延迟（模拟人类行为）"""
        delay = random.uniform(self.min_delay, self.max_delay)
        await asyncio.sleep(delay)
    
    @abstractmethod
    async def crawl(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        pass
    
    async def fetch(self, url: str, params: Optional[Dict] = None, 
                    method: str = 'GET', use_proxy: bool = False,
                    check_duplicate: bool = True) -> CrawlResult:
        """
        增强版HTTP请求方法
        
        返回:
            CrawlResult对象
        """
        start_time = time.time()
        
        # 1. 检查熔断
        if not self.circuit_breaker.should_allow():
            return CrawlResult(
                status=CrawlStatus.FAILED,
                url=url,
                error_message="Circuit breaker is open",
                metadata={"stage": "circuit_breaker"}
            )
        
        # 2. 获取限流许可
        try:
            await self.rate_limiter.acquire()
        except Exception as e:
            return CrawlResult(
                status=CrawlStatus.RATE_LIMITED,
                url=url,
                error_message=f"Rate limit exceeded: {e}",
                metadata={"stage": "rate_limit"}
            )
        
        # 3. 检查URL去重
        if check_duplicate:
            if await self.is_url_duplicate(url):
                self.metrics.duplicates_found += 1
                return CrawlResult(
                    status=CrawlStatus.FAILED,
                    url=url,
                    error_message="URL already crawled",
                    metadata={"stage": "duplicate_check", "is_duplicate": True}
                )
        
        # 4. 智能重试逻辑
        retry_count = 0
        last_exception = None
        current_proxy = None
        
        while retry_count < self.max_retry:
            try:
                # 随机延迟
                await self._apply_random_delay()
                
                session = await self._get_session(use_proxy)
                
                # 获取代理
                if use_proxy and not current_proxy:
                    current_proxy = await self.proxy_pool.get_proxy()
                
                print(f"🌐 [{retry_count+1}/{self.max_retry}] 请求: {url[:60]}...")
                
                proxy_url = current_proxy.proxy_url if current_proxy else None
                
                if method == 'GET':
                    async with session.get(
                        url, 
                        params=params,
                        proxy=proxy_url,
                        headers=self.headers
                    ) as response:
                        content = await response.text()
                        response_time = time.time() - start_time
                        
                        if response.status == 200:
                            # 记录成功
                            await self.circuit_breaker.record_success()
                            self.metrics.record_success(response_time, len(content))
                            
                            # 标记代理成功
                            if current_proxy:
                                self.proxy_pool.mark_success(current_proxy, response_time)
                            
                            return CrawlResult(
                                status=CrawlStatus.SUCCESS,
                                content=content,
                                url=url,
                                headers=dict(response.headers),
                                status_code=response.status,
                                response_time=response_time,
                                metadata={"stage": "success"}
                            )
                            
                        elif response.status == 403:
                            last_exception = Exception("403 Forbidden")
                            if current_proxy:
                                self.proxy_pool.mark_failure(current_proxy)
                                self.metrics.proxy_switches += 1
                                current_proxy = None  # 切换代理
                            
                        elif response.status == 429:
                            last_exception = Exception("429 Too Many Requests")
                            self.metrics.rate_limit_triggers += 1
                            self.rate_limiter.adjust_rate(0.5)  # 降低速率
                            await asyncio.sleep(10 * (retry_count + 1))
                            
                        elif response.status in [500, 502, 503, 504]:
                            last_exception = Exception(f"Server error: {response.status}")
                            # 服务器错误，可能需要切换代理
                            if current_proxy:
                                self.proxy_pool.mark_failure(current_proxy)
                                current_proxy = None
                            
                        else:
                            last_exception = Exception(f"HTTP {response.status}")
                
                elif method == 'POST':
                    async with session.post(
                        url, 
                        data=params,
                        proxy=proxy_url,
                        headers=self.headers
                    ) as response:
                        response_time = time.time() - start_time
                        
                        if response.status == 200:
                            content = await response.text()
                            await self.circuit_breaker.record_success()
                            self.metrics.record_success(response_time, len(content))
                            
                            return CrawlResult(
                                status=CrawlStatus.SUCCESS,
                                content=content,
                                url=url,
                                status_code=response.status,
                                response_time=response_time,
                                metadata={"stage": "success"}
                            )
                        else:
                            last_exception = Exception(f"HTTP {response.status}")
            
            except aiohttp.ClientError as e:
                last_exception = e
                if current_proxy:
                    self.proxy_pool.mark_failure(current_proxy)
                    current_proxy = None
            
            except asyncio.TimeoutError:
                last_exception = TimeoutError("Request timeout")
                self.circuit_breaker.record_failure()
            
            except Exception as e:
                last_exception = e
            
            retry_count += 1
            if retry_count < self.max_retry:
                backoff_time = self._calculate_backoff(retry_count)
                print(f"🔄 重试等待: {backoff_time:.1f}秒")
                await asyncio.sleep(backoff_time)
        
        # 记录失败
        await self.circuit_breaker.record_failure()
        self.metrics.record_failure()
        
        print(f"💀 所有重试均失败: {url}")
        return CrawlResult(
            status=CrawlStatus.FAILED,
            url=url,
            error_message=str(last_exception),
            response_time=time.time() - start_time,
            retry_count=retry_count,
            metadata={"stage": "final_failure"}
        )
    
    def _calculate_backoff(self, retry_count: int) -> float:
        """计算指数退避时间"""
        base_backoff = 2 ** retry_count
        jitter = random.uniform(0, 1)
        return min(base_backoff + jitter, 30)  # 最大30秒
    
    async def fetch_json(self, url: str, params: Optional[Dict] = None, 
                         method: str = 'GET') -> Optional[Dict]:
        """获取JSON响应"""
        result = await self.fetch(url, params, method)
        if result.status == CrawlStatus.SUCCESS and result.content:
            try:
                return json.loads(result.content)
            except json.JSONDecodeError:
                print(f"❌ JSON解析失败")
                return None
        return None
    
    async def fetch_binary(self, url: str, params: Optional[Dict] = None) -> Optional[bytes]:
        """获取二进制响应"""
        if not self.circuit_breaker.should_allow():
            return None
        
        await self.rate_limiter.acquire()
        
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    await self.circuit_breaker.record_success()
                    return await response.read()
        except Exception as e:
            print(f"❌ 二进制请求失败: {e}")
            await self.circuit_breaker.record_failure()
        
        return None
    
    async def is_url_duplicate(self, url: str) -> bool:
        """检查URL是否已爬取过"""
        async with _global_url_cache_lock:
            url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
            if url_hash in _global_url_cache:
                return True
            
            _global_url_cache.add(url_hash)
            if len(_global_url_cache) > MAX_CACHE_SIZE:
                _global_url_cache = set(list(_global_url_cache)[-int(MAX_CACHE_SIZE * 0.8):])
        
        return False
    
    def parse_datetime(self, time_str: str) -> str:
        """智能解析时间字符串"""
        if not time_str:
            return datetime.now().isoformat()
        
        time_str = time_str.strip()
        
        # 相对时间解析
        relative_patterns = [
            (r'(\d+)\s*分钟前', lambda m: timedelta(minutes=int(m.group(1)))),
            (r'(\d+)\s*小时前', lambda m: timedelta(hours=int(m.group(1)))),
            (r'(\d+)\s*天前', lambda m: timedelta(days=int(m.group(1)))),
            (r'(\d+)\s*日前', lambda m: timedelta(days=int(m.group(1)))),
            (r'昨天', lambda m: timedelta(days=1)),
            (r'前天', lambda m: timedelta(days=2)),
            (r'刚刚', lambda m: timedelta(minutes=1)),
            (r'刚才', lambda m: timedelta(minutes=5)),
            (r'近\s*(\d+)\s*小时', lambda m: timedelta(hours=int(m.group(1)))),
            (r'近\s*(\d+)\s*天', lambda m: timedelta(days=int(m.group(1)))),
        ]
        
        for pattern, delta_func in relative_patterns:
            match = re.match(pattern, time_str, re.IGNORECASE)
            if match:
                delta = delta_func(match)
                return (datetime.now() - delta).isoformat()
        
        # 绝对时间解析
        patterns = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%Y-%m-%d',
            '%Y/%m/%d %H:%M:%S',
            '%Y/%m/%d %H:%M',
            '%Y/%m/%d',
            '%Y年%m月%d日 %H:%M:%S',
            '%Y年%m月%d日 %H:%M',
            '%Y年%m月%d日',
            '%m-%d %H:%M',
            '%m/%d %H:%M',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S.%f',
        ]
        
        for pattern in patterns:
            try:
                return datetime.strptime(time_str, pattern).isoformat()
            except ValueError:
                continue
        
        return datetime.now().isoformat()
    
    def create_news_item(self, title: str, content: str, source: str, 
                         publish_time: str, url: str, **kwargs) -> Dict[str, Any]:
        """创建标准化新闻条目"""
        base_item = {
            'news_id': f"{self.name}_{hash(title + str(publish_time)) % 10000000}",
            'title': title.strip()[:200] if title else '',
            'content': content.strip()[:3000] if content else '',
            'source': source,
            'publish_time': self.parse_datetime(publish_time) if publish_time else datetime.now().isoformat(),
            'url': url[:500] if url else '',
            'platform': self.name,
            'crawled_at': datetime.now().isoformat(),
            'content_hash': hashlib.md5((title + content).encode('utf-8')).hexdigest()
        }
        
        # 添加额外字段
        base_item.update(kwargs)
        
        return base_item
    
    def clean_text(self, text: str) -> str:
        """清理文本内容"""
        if not text:
            return ''
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 移除多余空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符（保留基本标点）
        text = re.sub(r'[^\u4e00-\u9fff0-9a-zA-Z，。！？、：；""\'\'（）\s]', '', text)
        
        # 移除首尾空白
        text = text.strip()
        
        return text
    
    def extract_text_from_html(self, html_content: str) -> str:
        """从HTML中提取纯文本"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 移除脚本、样式等无用标签
            for tag in soup(['script', 'style', 'noscript', 'iframe', 'head', 'nav', 'footer']):
                tag.decompose()
            
            # 获取文本
            text = soup.get_text(separator=' ', strip=True)
            
            # 清理文本
            return self.clean_text(text)
        
        except Exception as e:
            print(f"❌ HTML解析失败: {e}")
            return self.clean_text(html_content)
    
    def extract_metadata_from_html(self, html_content: str) -> Dict[str, Any]:
        """从HTML中提取元数据"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            metadata = {}
            
            # 提取标题
            title_tag = soup.find('title')
            if title_tag:
                metadata['page_title'] = title_tag.get_text(strip=True)
            
            # 提取meta标签
            meta_tags = soup.find_all('meta')
            for tag in meta_tags:
                name = tag.get('name', '').lower()
                property = tag.get('property', '').lower()
                content = tag.get('content', '')
                
                if name in ['keywords', 'description', 'author', 'publisher']:
                    metadata[name] = content
                elif property in ['og:title', 'og:description', 'og:url', 'og:type']:
                    metadata[property] = content
            
            # 提取发布时间
            time_tags = soup.find_all('time')
            for tag in time_tags:
                if 'datetime' in tag.attrs:
                    metadata['publish_time'] = tag['datetime']
                    break
            
            return metadata
        
        except Exception as e:
            print(f"❌ 元数据提取失败: {e}")
            return {}
    
    def generate_summary(self, text: str, max_length: int = 150) -> str:
        """生成文本摘要（基于TF-IDF关键词提取）"""
        if not text:
            return ''
        
        # 简单摘要：取前max_length个字符，在句子边界处截断
        if len(text) <= max_length:
            return text
        
        # 找到最后一个完整句子
        text = text[:max_length + 50]
        sentence_enders = ['。', '！', '？', '；', '\n']
        
        for i in range(len(text) - 1, max_length - 1, -1):
            if text[i] in sentence_enders:
                return text[:i + 1]
        
        return text[:max_length] + '...'
    
    async def close(self):
        """关闭资源"""
        if self._session:
            await self._session.close()
            self._session = None
            print(f"✅ 关闭{self.name}爬虫资源")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取爬虫状态统计"""
        return {
            'name': self.name,
            'rate_limit_current': self.rate_limiter.current_rate,
            'rate_limit_max': self.rate_limiter.max_requests,
            'circuit_breaker_state': self.circuit_breaker.state.value,
            'circuit_breaker_failure_rate': self.circuit_breaker.failure_rate,
            'consecutive_failures': self.circuit_breaker.consecutive_failures,
            'retry_count': self.max_retry,
            'timeout': self.timeout,
            'proxy_healthy_count': self.proxy_pool.healthy_count,
            'proxy_total_count': len(self.proxy_pool.proxies),
            **self.metrics.__dict__
        }
    
    def reset_metrics(self):
        """重置性能指标"""
        self.metrics.reset()
