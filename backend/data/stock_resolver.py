"""
股票代码动态解析器
支持：股票名称、代码、模糊搜索
数据源：AKShare A股列表
"""

import akshare as ak
import pandas as pd
from typing import Optional, List, Dict
import time


class StockResolver:
    """动态股票代码解析器 - 无需硬编码映射"""

    def __init__(self):
        self._stock_list: Optional[pd.DataFrame] = None
        self._last_load_time: float = 0
        self._cache_ttl: int = 3600  # 缓存1小时

    def _ensure_loaded(self) -> pd.DataFrame:
        """确保股票列表已加载（带缓存）"""
        now = time.time()
        if self._stock_list is None or (now - self._last_load_time) > self._cache_ttl:
            self._load_stock_list()
        return self._stock_list

    def _load_stock_list(self):
        """加载A股列表"""
        try:
            self._stock_list = ak.stock_info_a_code_name()
            self._last_load_time = time.time()
            print(f"[RESOLVER] 加载A股列表成功: {len(self._stock_list)} 只股票")
        except Exception as e:
            print(f"[RESOLVER] 加载股票列表失败: {e}")
            if self._stock_list is None:
                self._stock_list = pd.DataFrame(columns=['code', 'name'])

    def resolve(self, user_input: str) -> Dict[str, str]:
        """
        解析用户输入，返回股票代码和名称

        参数:
            user_input: 用户输入（可以是名称、代码、或模糊匹配）

        返回:
            {
                "code": "600519",
                "name": "贵州茅台",
                "match_type": "exact_name" | "exact_code" | "fuzzy_name" | "not_found",
                "confidence": 1.0 | 0.8 | 0.0
            }
        """
        df = self._ensure_loaded()
        if df.empty:
            return self._not_found(user_input)

        user_input = user_input.strip()

        # 1. 精确匹配代码
        exact_code = df[df['code'] == user_input]
        if not exact_code.empty:
            row = exact_code.iloc[0]
            return {
                "code": row['code'],
                "name": row['name'].strip(),
                "match_type": "exact_code",
                "confidence": 1.0
            }

        # 2. 精确匹配名称（去除空格）
        df['name_clean'] = df['name'].str.replace(r'\s+', '', regex=True)
        clean_input = user_input.replace(' ', '')
        exact_name = df[df['name_clean'] == clean_input]
        if not exact_name.empty:
            row = exact_name.iloc[0]
            return {
                "code": row['code'],
                "name": row['name'].strip(),
                "match_type": "exact_name",
                "confidence": 1.0
            }

        # 3. 模糊匹配名称（包含关键词）
        fuzzy_match = df[df['name_clean'].str.contains(clean_input, na=False)]
        if not fuzzy_match.empty:
            # 返回最相关的（名称最短的优先，因为通常更精确）
            fuzzy_match = fuzzy_match.copy()
            fuzzy_match['name_len'] = fuzzy_match['name_clean'].str.len()
            fuzzy_match = fuzzy_match.sort_values('name_len')
            row = fuzzy_match.iloc[0]
            return {
                "code": row['code'],
                "name": row['name'].strip(),
                "match_type": "fuzzy_name",
                "confidence": 0.8
            }

        return self._not_found(user_input)

    def _not_found(self, user_input: str) -> Dict[str, str]:
        """返回未找到结果"""
        return {
            "code": user_input,
            "name": user_input,
            "match_type": "not_found",
            "confidence": 0.0
        }

    def search(self, keyword: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        搜索股票（返回多个候选）

        参数:
            keyword: 搜索关键词
            limit: 返回数量

        返回:
            [{code, name}, ...]
        """
        df = self._ensure_loaded()
        if df.empty:
            return []

        keyword = keyword.strip()
        df['name_clean'] = df['name'].str.replace(r'\s+', '', regex=True)

        # 优先精确匹配
        exact = df[df['name_clean'] == keyword.replace(' ', '')]
        if not exact.empty:
            return [{"code": r['code'], "name": r['name'].strip()} for _, r in exact.head(limit).iterrows()]

        # 模糊匹配
        fuzzy = df[df['name_clean'].str.contains(keyword, na=False)]
        if not fuzzy.empty:
            fuzzy = fuzzy.copy()
            fuzzy['name_len'] = fuzzy['name_clean'].str.len()
            fuzzy = fuzzy.sort_values('name_len')
            return [{"code": r['code'], "name": r['name'].strip()} for _, r in fuzzy.head(limit).iterrows()]

        return []

    def is_valid_stock(self, user_input: str) -> bool:
        """检查是否为有效股票"""
        result = self.resolve(user_input)
        return result["match_type"] != "not_found"


# 全局单例
stock_resolver = StockResolver()


# 便捷函数
def resolve_stock(user_input: str) -> Dict[str, str]:
    """解析股票名称/代码"""
    return stock_resolver.resolve(user_input)


def search_stocks(keyword: str, limit: int = 10) -> List[Dict[str, str]]:
    """搜索股票"""
    return stock_resolver.search(keyword, limit)
