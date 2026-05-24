"""
股票数据获取器 - 免费数据源版本
功能：使用AKShare等免费数据源获取股票历史价格、成交量等数据
支持：任意股票名称/代码自动解析（无需硬编码映射）
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import asyncio

# 导入动态解析器
from backend.data.stock_resolver import stock_resolver, resolve_stock

# ==================== 保留常用映射作为快速缓存 ====================
# 优先使用 stock_resolver 动态解析，此映射仅作备用
STOCK_CODE_MAPPING = {
    '贵州茅台': '600519',
    '比亚迪': '002594',
    '宁德时代': '300750',
    '五粮液': '000858',
    '招商银行': '600036',
    '海康威视': '002415',
    '隆基绿能': '601012',
    '药明康德': '603259',
    '万科A': '000002',
    '格力电器': '000651',
    '腾讯控股': '00700',
    '美团': '03690',
    '京东': 'JD',
    '网易': 'NTES',
}

# 指数代码（AKShare格式）
INDEX_CODES = {
    '上证指数': '000001',
    '沪深300': '000300',
    '创业板指': '399006',
    '深证成指': '399001'
}

class StockDataFetcher:
    """股票数据获取器 - 使用AKShare免费数据源"""
    
    def __init__(self):
        print("✅ 股票数据获取器初始化成功（免费数据源）")
        print("   ✓ 使用AKShare作为主要数据源")
    
    def get_stock_code(self, stock_name: str) -> str:
        """
        获取股票代码 - 支持动态解析

        优先级：
        1. 精确匹配本地映射（快速）
        2. 动态解析（支持任意股票）
        """
        # 1. 先查本地缓存
        if stock_name in STOCK_CODE_MAPPING:
            return STOCK_CODE_MAPPING[stock_name]

        # 2. 动态解析
        result = resolve_stock(stock_name)
        if result["match_type"] != "not_found":
            print(f"[RESOLVER] 动态解析: {stock_name} -> {result['code']} ({result['name']})")
            # 缓存到本地映射
            STOCK_CODE_MAPPING[stock_name] = result["code"]
            return result["code"]

        # 3. 返回原输入（可能是纯代码）
        return stock_name
    
    def get_kline_data(self, stock_code: str, start_date: str = None, 
                      end_date: str = None, freq: str = 'daily') -> pd.DataFrame:
        """
        获取股票K线数据（使用AKShare免费数据源）
        
        参数:
            stock_code: 股票代码（6位数字，不含.SH/.SZ）
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            freq: 频率 (daily/weekly/monthly)
        
        返回:
            K线数据DataFrame
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        try:
            # 尝试A股数据
            if len(stock_code) == 6:
                if stock_code.startswith('6'):  # 沪市
                    df = ak.stock_zh_a_daily(symbol=stock_code, start_date=start_date, end_date=end_date, adjust="qfq")
                else:  # 深市
                    df = ak.stock_zh_a_daily(symbol=stock_code, start_date=start_date, end_date=end_date, adjust="qfq")
                
                if not df.empty:
                    return self._format_akshare_kline(df)
            
            # 尝试港股
            elif len(stock_code) == 5 and stock_code[0] == '0':
                df = ak.stock_hk_daily(symbol=stock_code, start_date=start_date, end_date=end_date)
                if not df.empty:
                    return self._format_akshare_kline(df)
            
            # 尝试美股
            else:
                try:
                    df = ak.stock_us_daily(symbol=stock_code, start_date=start_date, end_date=end_date)
                    if not df.empty:
                        return self._format_akshare_kline(df)
                except:
                    pass
        
        except Exception as e:
            import traceback
            print(f"   ⚠️ 获取K线数据失败: {e}")
            traceback.print_exc()   # 这会打印出完整的错误栈，方便定位
            return pd.DataFrame()
    
    def get_kline_data_async(self, stock_code: str, start_date: str = None,
                            end_date: str = None, freq: str = 'daily') -> pd.DataFrame:
        """异步获取K线数据"""
        return self.get_kline_data(stock_code, start_date, end_date, freq)
    
    def _format_akshare_kline(self, df: pd.DataFrame) -> pd.DataFrame:
        """格式化AKShare返回的K线数据（增强版）"""
        if df.empty:
            return df

        # 打印原始列名，便于调试（正式环境可注释掉）
        print(f"   [调试] 原始列名: {list(df.columns)}")

        # 完整的中英文列名映射
        column_mapping = {
            # 中文常见列名
            '日期': 'date',
            '时间': 'date',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume',
            '成交额': 'amount',
            '涨跌幅': 'change_pct',
            '涨跌额': 'change',
            '换手率': 'turnover_rate',
            '振幅': 'amplitude',
            # 英文小写
            'date': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'amount': 'amount',
            'change_pct': 'change_pct',
        }

        # 只重命名存在的列
        rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=rename_dict)

        # 确保 date 列存在且转换为 datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
        else:
            print("   ❌ 缺少 'date' 列，无法继续。当前列名：", list(df.columns))
            return pd.DataFrame()

        # 检查 OHLCV 必须列，缺失则用近似值补全
        if 'close' not in df.columns:
            print("   ❌ 缺少 'close' 列")
            return pd.DataFrame()

        for col in ['open', 'high', 'low']:
            if col not in df.columns:
                print(f"   ⚠️ 缺少 '{col}' 列，用 'close' 代替")
                df[col] = df['close']

        if 'volume' not in df.columns:
            print("   ⚠️ 缺少 'volume' 列，填充 0")
            df['volume'] = 0

        # 调整列顺序
        desired_order = ['date', 'open', 'high', 'low', 'close', 'volume']
        available_cols = [c for c in desired_order if c in df.columns]
        other_cols = [c for c in df.columns if c not in desired_order]
        df = df[available_cols + other_cols]

        return df
    
    def _format_akshare_kline(self, df: pd.DataFrame) -> pd.DataFrame:
        """格式化AKShare返回的K线数据（终极版）"""
        if df.empty:
            return df

        # 打印原始列名，帮助定位问题
        print(f"   [调试] 原始列名: {list(df.columns)}")

        # 1. 直接处理常见的两种返回格式
        # 格式A：中文列名（如 '日期', '开盘'）
        # 格式B：英文小写（如 'date', 'open'）
        col_map = {}
        for col in df.columns:
            col_str = str(col).strip()
            col_lower = col_str.lower()
            # 日期
            if any(keyword in col_str for keyword in ['日期', '时间', 'date', 'time']):
                col_map[col] = 'date'
            # 开盘
            elif any(keyword in col_str for keyword in ['开盘', 'open']):
                col_map[col] = 'open'
            # 最高
            elif any(keyword in col_str for keyword in ['最高', 'high']):
                col_map[col] = 'high'
            # 最低
            elif any(keyword in col_str for keyword in ['最低', 'low']):
                col_map[col] = 'low'
            # 收盘
            elif any(keyword in col_str for keyword in ['收盘', 'close']):
                col_map[col] = 'close'
            # 成交量
            elif any(keyword in col_str for keyword in ['成交量', 'volume']):
                col_map[col] = 'volume'
            # 成交额
            elif any(keyword in col_str for keyword in ['成交额', 'amount']):
                col_map[col] = 'amount'
            # 涨跌幅
            elif any(keyword in col_str for keyword in ['涨跌幅', 'pct_chg', 'change_pct', '涨跌额']):
                col_map[col] = 'change_pct'
            # 其他保留原样

        # 应用映射
        if col_map:
            df = df.rename(columns=col_map)
            print(f"   [调试] 重命名后的列名: {list(df.columns)}")

        # 确保必须的列存在
        required = {'date', 'open', 'high', 'low', 'close'}
        missing = required - set(df.columns)
        if missing:
            # 如果缺少关键列，抛出明确的错误
            raise KeyError(f"数据列映射后仍缺少这些列: {missing}。当前列名: {list(df.columns)}")

        # 转换日期格式
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)

        # 如果缺少成交量，填充0
        if 'volume' not in df.columns:
            df['volume'] = 0.0

        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR指标"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(period).mean()
    
    def get_market_overview(self) -> Dict[str, Any]:
        """获取市场概览数据"""
        overview = {}
        
        for index_name, code in INDEX_CODES.items():
            try:
                df = self.get_index_data(index_name)
                if not df.empty and len(df) > 0:
                    latest = df.iloc[-1]
                    prev = df.iloc[-2] if len(df) > 1 else latest
                    change_pct = (latest['close'] - prev['close']) / prev['close'] * 100
                    overview[index_name] = {
                        'close': float(latest['close']),
                        'change_pct': float(change_pct),
                        'volume': float(latest.get('volume', 0))
                    }
            except Exception as e:
                pass
        
        return overview
    
    def calculate_relative_performance(self, stock_df: pd.DataFrame, 
                                      index_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算相对表现
        
        参数:
            stock_df: 股票K线数据
            index_df: 指数K线数据
        
        返回:
            包含相对表现的DataFrame
        """
        if stock_df.empty or index_df.empty:
            return stock_df
        
        merged = pd.merge(stock_df, index_df, on='date', suffixes=('_stock', '_index'))
        
        merged['stock_return'] = merged['close_stock'].pct_change().cumsum()
        merged['index_return'] = merged['close_index'].pct_change().cumsum()
        merged['relative_return'] = merged['stock_return'] - merged['index_return']
        merged['relative_strength'] = merged['close_stock'] / merged['close_index']
        
        # 计算Beta系数
        returns = merged[['close_stock', 'close_index']].pct_change().dropna()
        if len(returns) >= 2:
            cov_matrix = np.cov(returns['close_stock'], returns['close_index'])
            beta = cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-8)
            merged['beta'] = beta
        
        return merged
    
    def get_stock_summary(self, stock_name: str, days: int = 30) -> Dict[str, Any]:
        """
        获取股票综合摘要

        参数:
            stock_name: 股票名称或代码（支持模糊匹配）
            days: 时间范围

        返回:
            综合摘要字典
        """
        # 动态解析股票代码
        resolution = resolve_stock(stock_name)
        stock_code = resolution["code"]
        resolved_name = resolution["name"]

        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        kline_df = self.get_kline_data(stock_code, start_date, end_date)

        if kline_df.empty:
            return {
                'stock_name': resolved_name,
                'stock_code': stock_code,
                'input_name': stock_name,
                'resolution': resolution,
                'error': f'无法获取 {resolved_name}({stock_code}) 的数据'
            }
        
        kline_df = self.get_tech_indicators(kline_df)
        
        index_df = self.get_index_data('沪深300', start_date, end_date)
        
        if not index_df.empty:
            kline_df = self.calculate_relative_performance(kline_df, index_df)
        
        latest = kline_df.iloc[-1]
        prev = kline_df.iloc[-2] if len(kline_df) > 1 else latest
        
        trend = 'up' if latest['close'] > latest.get('ma20', latest['close']) else 'down' if latest['close'] < latest.get('ma20', latest['close']) else 'sideways'
        
        summary = {
            'stock_name': resolved_name,
            'stock_code': stock_code,
            'input_name': stock_name,
            'resolution': resolution,
            'latest_price': float(latest['close']),
            'change_pct': float(latest.get('change_pct', (latest['close'] - prev['close']) / prev['close'] * 100)),
            'volume': float(latest.get('volume', 0)),
            'trend': trend,
            'technical_indicators': {
                'ma5': float(latest.get('ma5', 0)),
                'ma20': float(latest.get('ma20', 0)),
                'ma60': float(latest.get('ma60', 0)),
                'rsi': float(latest.get('rsi', 50)),
                'macd': float(latest.get('macd', 0)),
                'macd_signal': float(latest.get('signal', 0)),
                'atr': float(latest.get('atr', 0)),
                'bb_position': ((latest['close'] - latest.get('bb_lower', latest['close'])) / 
                               (latest.get('bb_upper', latest['close']) - latest.get('bb_lower', latest['close']) + 1e-8))
            },
            'relative_performance': {
                'vs_index': float(kline_df.iloc[-1].get('relative_return', 0)),
                'beta': float(kline_df.iloc[-1].get('beta', 1.0))
            } if 'relative_return' in kline_df.columns else {},
            'data_points': len(kline_df),
            'last_update': datetime.now().isoformat(),
            'data_source': 'AKShare（免费数据源）'
        }
        
        return summary
    
    async def get_stock_summary_async(self, stock_name: str, days: int = 30) -> Dict[str, Any]:
        """异步获取股票摘要"""
        return self.get_stock_summary(stock_name, days)

# ==================== 全局实例 ====================
stock_data_fetcher = StockDataFetcher()

# ==================== 便捷函数 ====================
def fetch_stock_data(stock_name: str, days: int = 30) -> Dict[str, Any]:
    """便捷函数：获取股票数据"""
    return stock_data_fetcher.get_stock_summary(stock_name, days)

async def fetch_stock_data_async(stock_name: str, days: int = 30) -> Dict[str, Any]:
    """异步便捷函数"""
    return await stock_data_fetcher.get_stock_summary_async(stock_name, days)
