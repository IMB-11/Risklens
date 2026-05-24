"""
相对表现分析模块 - 无市场环境 → 相对表现分析
功能：计算个股相对于大盘和行业的表现
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from scipy import stats

# ==================== 行业分类 ====================
INDUSTRY_MAPPING = {
    '贵州茅台': '食品饮料',
    '五粮液': '食品饮料',
    '泸州老窖': '食品饮料',
    '比亚迪': '汽车整车',
    '宁德时代': '电力设备',
    '隆基绿能': '电力设备',
    '海康威视': '电子',
    '药明康德': '医药生物',
    '恒瑞医药': '医药生物',
    '招商银行': '银行',
    '平安银行': '银行',
    '万科A': '房地产',
    '格力电器': '家用电器',
    '美的集团': '家用电器',
    '腾讯控股': '互联网服务',
    '美团': '互联网服务',
    '阿里巴巴': '互联网服务',
    '京东': '互联网服务',
    '网易': '互联网服务',
    '小米集团': '电子',
    '京东集团': '互联网服务'
}

class RelativePerformanceAnalyzer:
    """相对表现分析器"""
    
    def __init__(self):
        self.cache = {}
        print("✅ 相对表现分析器初始化成功")
    
    def calculate_relative_performance(self, stock_df: pd.DataFrame, 
                                      index_df: pd.DataFrame,
                                      industry_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        计算相对表现
        
        参数:
            stock_df: 个股价格数据
            index_df: 指数价格数据
            industry_df: 可选的行业指数数据
        
        返回:
            相对表现分析结果
        """
        if stock_df.empty or index_df.empty:
            return {
                'error': '数据不足',
                'relative_return': 0.0,
                'beta': 1.0,
                'alpha': 0.0
            }
        
        # 对齐日期
        stock_df['date'] = pd.to_datetime(stock_df['date'])
        index_df['date'] = pd.to_datetime(index_df['date'])
        
        merged = pd.merge(stock_df[['date', 'close']], 
                        index_df[['date', 'close']], 
                        on='date', 
                        suffixes=('_stock', '_index'))
        
        if industry_df is not None and not industry_df.empty:
            industry_df['date'] = pd.to_datetime(industry_df['date'])
            merged = pd.merge(merged, industry_df[['date', 'close']], 
                            on='date', 
                            suffixes=('', '_industry'))
        
        # 计算收益率
        merged['stock_return'] = merged['close_stock'].pct_change().fillna(0)
        merged['index_return'] = merged['close_index'].pct_change().fillna(0)
        
        if 'close_industry' in merged.columns:
            merged['industry_return'] = merged['close_industry'].pct_change().fillna(0)
        
        # 计算累计收益率
        merged['stock_cum_return'] = (1 + merged['stock_return']).cumprod() - 1
        merged['index_cum_return'] = (1 + merged['index_return']).cumprod() - 1
        
        # 相对收益
        merged['relative_return'] = merged['stock_cum_return'] - merged['index_cum_return']
        
        # 计算Beta和Alpha
        beta, alpha = self._calculate_beta_alpha(merged['stock_return'].values, 
                                                merged['index_return'].values)
        
        # 计算其他指标
        sharpe_ratio = self._calculate_sharpe_ratio(merged['stock_return'].values)
        max_drawdown = self._calculate_max_drawdown(merged['stock_cum_return'].values)
        volatility = np.std(merged['stock_return'].values) * np.sqrt(252)  # 年化波动率
        
        # 相对强弱指标
        rsi_relative = self._calculate_rsi_relative(merged['stock_cum_return'].values, 
                                                    merged['index_cum_return'].values)
        
        result = {
            'relative_return': float(merged['relative_return'].iloc[-1]) if not merged.empty else 0.0,
            'beta': float(beta),
            'alpha': float(alpha),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'volatility': float(volatility),
            'rsi_relative': float(rsi_relative),
            'outperformance_days': int((merged['stock_return'] > merged['index_return']).sum()),
            'underperformance_days': int((merged['stock_return'] < merged['index_return']).sum()),
            'tracking_error': float(np.std(merged['stock_return'] - merged['index_return']) * np.sqrt(252)),
            'information_ratio': float(self._calculate_information_ratio(merged['stock_return'].values, 
                                                                        merged['index_return'].values)),
            'sample_count': len(merged),
            'period_start': merged['date'].min().strftime('%Y-%m-%d') if not merged.empty else None,
            'period_end': merged['date'].max().strftime('%Y-%m-%d') if not merged.empty else None
        }
        
        # 添加行业对比
        if 'industry_return' in merged.columns:
            merged['industry_cum_return'] = (1 + merged['industry_return']).cumprod() - 1
            merged['vs_industry_return'] = merged['stock_cum_return'] - merged['industry_cum_return']
            
            industry_beta, industry_alpha = self._calculate_beta_alpha(
                merged['stock_return'].values, merged['industry_return'].values
            )
            
            result.update({
                'vs_industry_return': float(merged['vs_industry_return'].iloc[-1]),
                'industry_beta': float(industry_beta),
                'industry_alpha': float(industry_alpha),
                'vs_industry_outperformance': int((merged['stock_return'] > merged['industry_return']).sum())
            })
        
        return result
    
    def _calculate_beta_alpha(self, stock_returns: np.ndarray, 
                             index_returns: np.ndarray) -> Tuple[float, float]:
        """
        计算Beta和Alpha
        
        参数:
            stock_returns: 个股收益率序列
            index_returns: 指数收益率序列
        
        返回:
            (beta, alpha)
        """
        # 移除NaN和Inf
        mask = np.isfinite(stock_returns) & np.isfinite(index_returns)
        stock_returns = stock_returns[mask]
        index_returns = index_returns[mask]
        
        if len(stock_returns) < 10:
            return 1.0, 0.0
        
        # 添加无风险收益率（简化为0）
        excess_stock = stock_returns
        excess_index = index_returns
        
        # 计算协方差和方差
        cov_matrix = np.cov(excess_stock, excess_index)
        if cov_matrix[1, 1] == 0:
            return 1.0, 0.0
        
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        
        # 计算Alpha（截距）
        alpha = np.mean(excess_stock) - beta * np.mean(excess_index)
        
        return beta, alpha
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, 
                               risk_free_rate: float = 0.0) -> float:
        """
        计算夏普比率
        
        参数:
            returns: 收益率序列
            risk_free_rate: 无风险利率
        
        返回:
            夏普比率
        """
        mask = np.isfinite(returns)
        returns = returns[mask]
        
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # 日化无风险利率
        
        std_dev = np.std(excess_returns)
        if std_dev == 0:
            return 0.0
        
        return np.sqrt(252) * np.mean(excess_returns) / std_dev
    
    def _calculate_max_drawdown(self, cum_returns: np.ndarray) -> float:
        """
        计算最大回撤
        
        参数:
            cum_returns: 累计收益率序列
        
        返回:
            最大回撤
        """
        if len(cum_returns) == 0:
            return 0.0
        
        # 转换为净值曲线
        equity_curve = 1 + cum_returns
        
        # 计算峰值
        running_max = np.maximum.accumulate(equity_curve)
        
        # 计算回撤
        drawdown = (equity_curve - running_max) / running_max
        
        return abs(np.min(drawdown))
    
    def _calculate_rsi_relative(self, stock_cum: np.ndarray, 
                               index_cum: np.ndarray) -> float:
        """
        计算相对强弱指数
        
        参数:
            stock_cum: 个股累计收益率
            index_cum: 指数累计收益率
        
        返回:
            相对强弱指数 (0-100)
        """
        # 相对强弱 = 个股涨幅 / 指数涨幅
        relative_strength = (1 + stock_cum) / (1 + index_cum + 1e-8)
        
        # 取最近N期计算RSI
        N = 14
        if len(relative_strength) < N:
            return 50.0
        
        recent = relative_strength[-N:]
        changes = np.diff(recent)
        
        avg_gain = np.mean(changes[changes > 0]) if any(changes > 0) else 0.0
        avg_loss = np.mean(-changes[changes < 0]) if any(changes < 0) else 0.0
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_information_ratio(self, stock_returns: np.ndarray, 
                                    index_returns: np.ndarray) -> float:
        """
        计算信息比率
        
        参数:
            stock_returns: 个股收益率
            index_returns: 指数收益率
        
        返回:
            信息比率
        """
        excess_returns = stock_returns - index_returns
        
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)
        
        if std_excess == 0:
            return 0.0
        
        return np.sqrt(252) * mean_excess / std_excess
    
    def analyze_market_positioning(self, stock_df: pd.DataFrame,
                                  index_df: pd.DataFrame,
                                  industry_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        分析市场定位
        
        参数:
            stock_df: 个股数据
            index_df: 指数数据
            industry_df: 可选行业数据
        
        返回:
            市场定位分析
        """
        performance = self.calculate_relative_performance(stock_df, index_df, industry_df)
        
        # 评估表现等级
        relative_return = performance.get('relative_return', 0.0)
        beta = performance.get('beta', 1.0)
        alpha = performance.get('alpha', 0.0)
        
        # 表现评级
        if relative_return > 0.2:
            performance_rating = '优秀'
        elif relative_return > 0.1:
            performance_rating = '良好'
        elif relative_return > 0:
            performance_rating = '一般'
        elif relative_return > -0.1:
            performance_rating = '较差'
        else:
            performance_rating = '很差'
        
        # 风险评级
        if beta < 0.7:
            risk_rating = '低风险'
        elif beta < 1.0:
            risk_rating = '中低风险'
        elif beta < 1.3:
            risk_rating = '中等风险'
        else:
            risk_rating = '高风险'
        
        # 超额收益能力
        if alpha > 0.001:
            alpha_rating = '正alpha'
        elif alpha > -0.001:
            alpha_rating = '中性'
        else:
            alpha_rating = '负alpha'
        
        return {
            **performance,
            'performance_rating': performance_rating,
            'risk_rating': risk_rating,
            'alpha_rating': alpha_rating,
            'analysis_summary': self._generate_summary(performance, 
                                                     performance_rating, 
                                                     risk_rating, 
                                                     alpha_rating)
        }
    
    def _generate_summary(self, performance: Dict, 
                         performance_rating: str, 
                         risk_rating: str, 
                         alpha_rating: str) -> str:
        """
        生成分析摘要
        
        参数:
            performance: 表现数据
            performance_rating: 表现评级
            risk_rating: 风险评级
            alpha_rating: Alpha评级
        
        返回:
            分析摘要文本
        """
        summary = f"个股相对表现评级为{performance_rating}，"
        summary += f"风险等级为{risk_rating}，"
        
        rel_return = performance.get('relative_return', 0) * 100
        summary += f"相对沪深300累计超额收益{rel_return:.1f}%。"
        
        if alpha_rating == '正alpha':
            summary += "存在超额收益能力（正Alpha）。"
        elif alpha_rating == '负alpha':
            summary += "未表现出超额收益能力（负Alpha）。"
        
        if performance.get('beta', 1.0) < 1.0:
            summary += "个股波动低于市场平均水平。"
        else:
            summary += "个股波动高于市场平均水平。"
        
        return summary
    
    def get_industry_peers(self, stock_name: str) -> List[str]:
        """
        获取同行业股票列表
        
        参数:
            stock_name: 股票名称
        
        返回:
            同行业股票列表
        """
        industry = INDUSTRY_MAPPING.get(stock_name, '其他')
        
        peers = [name for name, ind in INDUSTRY_MAPPING.items() 
                if ind == industry and name != stock_name]
        
        return peers[:5]  # 最多返回5个
    
    async def analyze_relative_performance_async(self, stock_df: pd.DataFrame,
                                                 index_df: pd.DataFrame,
                                                 industry_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """异步分析相对表现"""
        return self.calculate_relative_performance(stock_df, index_df, industry_df)

# ==================== 全局实例 ====================

relative_performance_analyzer = RelativePerformanceAnalyzer()

# ==================== 便捷函数 ====================

def analyze_relative_performance(stock_df: pd.DataFrame, 
                                index_df: pd.DataFrame,
                                industry_df: Optional[pd.DataFrame] = None) -> Dict:
    """便捷函数：分析相对表现"""
    return relative_performance_analyzer.calculate_relative_performance(stock_df, index_df, industry_df)

def analyze_market_positioning(stock_df: pd.DataFrame, 
                               index_df: pd.DataFrame,
                               industry_df: Optional[pd.DataFrame] = None) -> Dict:
    """便捷函数：分析市场定位"""
    return relative_performance_analyzer.analyze_market_positioning(stock_df, index_df, industry_df)

async def analyze_relative_performance_async(stock_df: pd.DataFrame,
                                             index_df: pd.DataFrame,
                                             industry_df: Optional[pd.DataFrame] = None) -> Dict:
    """异步便捷函数"""
    return await relative_performance_analyzer.analyze_relative_performance_async(stock_df, index_df, industry_df)

def get_industry_peers(stock_name: str) -> List[str]:
    """便捷函数：获取同行业股票"""
    return relative_performance_analyzer.get_industry_peers(stock_name)