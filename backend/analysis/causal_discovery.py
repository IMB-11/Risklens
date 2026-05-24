"""
因果发现模块 - 数据驱动因果发现
功能：从时序数据中自动发现因果关系
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Any, Tuple, Optional
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import asyncio

class CausalDiscoveryEngine:
    """因果发现引擎 - 数据驱动的因果关系发现"""
    
    def __init__(self):
        self.min_lag = 1
        self.max_lag = 5
        self.significance_threshold = 0.05
        print("✅ 因果发现引擎初始化成功")
    
    def granger_causality_test(self, df: pd.DataFrame, cause_col: str, 
                              effect_col: str, max_lag: int = 5) -> Dict[str, Any]:
        """
        格兰杰因果检验
        
        参数:
            df: 包含时间序列的DataFrame
            cause_col: 原因变量列名
            effect_col: 结果变量列名
            max_lag: 最大滞后阶数
        
        返回:
            检验结果
        """
        try:
            # 准备数据
            data = df[[cause_col, effect_col]].dropna()
            
            if len(data) < max_lag * 2:
                return {
                    'significant': False,
                    'p_value': 1.0,
                    'f_statistic': 0.0,
                    'message': '数据不足'
                }
            
            # 执行检验
            results = grangercausalitytests(data.values, maxlag=max_lag, verbose=False)
            
            # 提取最佳结果
            best_result = None
            best_p_value = 1.0
            
            for lag, result in results.items():
                p_value = result[0]['ssr_ftest'][1]
                if p_value < best_p_value:
                    best_p_value = p_value
                    best_result = {
                        'lag': lag,
                        'f_statistic': result[0]['ssr_ftest'][0],
                        'p_value': p_value,
                        'ssr': result[0]['ssr_ftest'][2]
                    }
            
            return {
                'significant': best_p_value < self.significance_threshold,
                'p_value': best_p_value,
                'f_statistic': best_result['f_statistic'] if best_result else 0.0,
                'optimal_lag': best_result['lag'] if best_result else 1,
                'cause': cause_col,
                'effect': effect_col
            }
        
        except Exception as e:
            return {
                'significant': False,
                'p_value': 1.0,
                'f_statistic': 0.0,
                'message': f'检验失败: {str(e)}'
            }
    
    def mutual_information_analysis(self, df: pd.DataFrame, feature_col: str,
                                   target_col: str, discrete_features: bool = False) -> Dict[str, Any]:
        """
        互信息分析 - 衡量两个变量的依赖程度
        
        参数:
            df: 数据DataFrame
            feature_col: 特征列
            target_col: 目标列
            discrete_features: 是否为离散特征
        
        返回:
            互信息分析结果
        """
        try:
            data = df[[feature_col, target_col]].dropna()
            
            if len(data) < 10:
                return {'mutual_info': 0.0, 'normalized': 0.0}
            
            X = data[[feature_col]].values
            y = data[target_col].values
            
            # 计算互信息
            mi = mutual_info_regression(X, y, discrete_features=discrete_features)[0]
            
            # 归一化（除以H(y)的上界）
            y_entropy = self._estimate_entropy(y)
            normalized_mi = mi / max(y_entropy, 1e-8) if y_entropy > 0 else 0.0
            
            return {
                'mutual_info': mi,
                'normalized': min(normalized_mi, 1.0),
                'feature': feature_col,
                'target': target_col
            }
        
        except Exception as e:
            return {'mutual_info': 0.0, 'normalized': 0.0, 'error': str(e)}
    
    def _estimate_entropy(self, data: np.ndarray, bins: int = 10) -> float:
        """估计熵值"""
        hist, _ = np.histogram(data, bins=bins, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    
    def correlation_analysis(self, df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """
        相关性分析 - 计算皮尔逊和斯皮尔曼相关系数
        
        参数:
            df: 数据DataFrame
            col1: 第一列
            col2: 第二列
        
        返回:
            相关性分析结果
        """
        try:
            data = df[[col1, col2]].dropna()
            
            if len(data) < 5:
                return {
                    'pearson_r': 0.0,
                    'pearson_p': 1.0,
                    'spearman_r': 0.0,
                    'spearman_p': 1.0
                }
            
            # 皮尔逊相关（线性关系）
            pearson_r, pearson_p = pearsonr(data[col1], data[col2])
            
            # 斯皮尔曼相关（单调关系）
            spearman_r, spearman_p = spearmanr(data[col1], data[col2])
            
            return {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'significant': pearson_p < self.significance_threshold or spearman_p < self.significance_threshold
            }
        
        except Exception as e:
            return {
                'pearson_r': 0.0,
                'pearson_p': 1.0,
                'spearman_r': 0.0,
                'spearman_p': 1.0,
                'error': str(e)
            }
    
    def discover_causal_relationships(self, df: pd.DataFrame, 
                                     target_col: str = 'close',
                                     feature_cols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        从数据中发现因果关系
        
        参数:
            df: 包含时间序列的DataFrame
            target_col: 目标变量（通常是收盘价或收益率）
            feature_cols: 待检验的特征列
        
        返回:
            因果关系列表
        """
        if feature_cols is None:
            # 自动检测数值列
            feature_cols = [col for col in df.columns 
                          if col != target_col and df[col].dtype in [np.float64, np.int64]]
        
        results = []
        
        # 对每个特征进行检验
        for feature in feature_cols:
            if feature == target_col:
                continue
            
            # 1. 格兰杰因果检验
            gc_result = self.granger_causality_test(df, feature, target_col)
            
            # 2. 互信息分析
            mi_result = self.mutual_information_analysis(df, feature, target_col)
            
            # 3. 相关性分析
            corr_result = self.correlation_analysis(df, feature, target_col)
            
            # 综合评估因果强度
            causal_strength = self._calculate_causal_strength(gc_result, mi_result, corr_result)
            
            result = {
                'cause': feature,
                'effect': target_col,
                'significant': gc_result['significant'],
                'causal_strength': causal_strength,
                'confidence': 1.0 - gc_result['p_value'],
                'granger_test': gc_result,
                'mutual_info': mi_result,
                'correlation': corr_result,
                'optimal_lag': gc_result.get('optimal_lag', 1)
            }
            
            results.append(result)
        
        # 按因果强度排序
        results.sort(key=lambda x: x['causal_strength'], reverse=True)
        
        return results
    
    def _calculate_causal_strength(self, gc_result: Dict, mi_result: Dict, 
                                   corr_result: Dict) -> float:
        """
        综合计算因果强度
        
        参数:
            gc_result: 格兰杰检验结果
            mi_result: 互信息结果
            corr_result: 相关性结果
        
        返回:
            综合因果强度 (0-1)
        """
        weights = {
            'granger': 0.4,
            'mutual_info': 0.3,
            'correlation': 0.3
        }
        
        # 格兰杰得分（基于p值）
        granger_score = min(1.0, max(0.0, 1.0 - gc_result['p_value']))
        
        # 互信息得分（归一化值）
        mi_score = mi_result.get('normalized', 0.0)
        
        # 相关性得分（取绝对值最大的）
        corr_score = max(abs(corr_result.get('pearson_r', 0.0)), 
                         abs(corr_result.get('spearman_r', 0.0)))
        
        # 综合得分
        strength = (
            granger_score * weights['granger'] +
            mi_score * weights['mutual_info'] +
            corr_score * weights['correlation']
        )
        
        # 如果格兰杰检验不显著，降低强度
        if not gc_result.get('significant', False):
            strength *= 0.5
        
        return min(1.0, max(0.0, strength))
    
    def build_causal_graph(self, df: pd.DataFrame, target_col: str = 'close',
                          min_strength: float = 0.3) -> nx.DiGraph:
        """
        构建因果图
        
        参数:
            df: 数据DataFrame
            target_col: 目标变量
            min_strength: 最小因果强度阈值
        
        返回:
            有向因果图
        """
        G = nx.DiGraph()
        
        # 发现因果关系
        relationships = self.discover_causal_relationships(df, target_col)
        
        # 添加节点和边
        for rel in relationships:
            if rel['causal_strength'] >= min_strength:
                G.add_node(rel['cause'], type='feature')
                G.add_node(rel['effect'], type='target')
                G.add_edge(rel['cause'], rel['effect'],
                          strength=rel['causal_strength'],
                          confidence=rel['confidence'],
                          lag=rel['optimal_lag'])
        
        return G
    
    def analyze_time_lag_effects(self, df: pd.DataFrame, cause_col: str,
                                effect_col: str, max_lag: int = 10) -> pd.DataFrame:
        """
        分析时间滞后效应
        
        参数:
            df: 数据DataFrame
            cause_col: 原因变量
            effect_col: 结果变量
            max_lag: 最大滞后阶数
        
        返回:
            不同滞后阶数的效应分析
        """
        results = []
        
        for lag in range(1, max_lag + 1):
            # 计算滞后效应
            shifted = df[[cause_col, effect_col]].copy()
            shifted[cause_col] = shifted[cause_col].shift(lag)
            
            corr = self.correlation_analysis(shifted, cause_col, effect_col)
            
            results.append({
                'lag': lag,
                'pearson_r': corr['pearson_r'],
                'pearson_p': corr['pearson_p'],
                'spearman_r': corr['spearman_r'],
                'spearman_p': corr['spearman_p'],
                'effect_strength': abs(corr['pearson_r']) if corr['pearson_p'] < 0.05 else 0.0
            })
        
        return pd.DataFrame(results)
    
    def detect_leading_indicators(self, df: pd.DataFrame, target_col: str,
                                 look_ahead: int = 5) -> List[Dict[str, Any]]:
        """
        检测领先指标
        
        参数:
            df: 数据DataFrame
            target_col: 目标变量
            look_ahead: 领先期数
        
        返回:
            领先指标列表
        """
        # 自动检测特征列
        feature_cols = [col for col in df.columns 
                      if col != target_col and df[col].dtype in [np.float64, np.int64]]
        
        indicators = []
        
        for feature in feature_cols:
            # 计算特征对未来目标的预测能力
            shifted_target = df[target_col].shift(-look_ahead)
            temp_df = pd.DataFrame({
                'feature': df[feature],
                'target': shifted_target
            }).dropna()
            
            if len(temp_df) < 10:
                continue
            
            # 计算相关性
            corr = self.correlation_analysis(temp_df, 'feature', 'target')
            
            # 评估预测能力
            predictive_power = abs(corr['pearson_r']) if corr['pearson_p'] < 0.05 else 0.0
            
            indicators.append({
                'indicator': feature,
                'look_ahead': look_ahead,
                'predictive_power': predictive_power,
                'correlation': corr['pearson_r'],
                'p_value': corr['pearson_p'],
                'significant': corr['significant']
            })
        
        # 排序
        indicators.sort(key=lambda x: x['predictive_power'], reverse=True)
        
        return indicators
    
    def analyze_causal_chain(self, df: pd.DataFrame, start_col: str, 
                            end_col: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """
        分析因果链
        
        参数:
            df: 数据DataFrame
            start_col: 起始变量
            end_col: 目标变量
            max_depth: 最大深度
        
        返回:
            因果链路径
        """
        # 发现所有因果关系
        all_relationships = self.discover_causal_relationships(df, end_col)
        
        # 构建因果图
        G = self.build_causal_graph(df, end_col, min_strength=0.2)
        
        # 寻找路径
        try:
            paths = list(nx.all_simple_paths(G, source=start_col, target=end_col, cutoff=max_depth))
        except nx.NetworkXNoPath:
            paths = []
        
        # 分析每条路径
        results = []
        for path in paths:
            path_strength = 1.0
            path_confidence = 1.0
            
            for i in range(len(path) - 1):
                edge_data = G.get_edge_data(path[i], path[i+1])
                if edge_data:
                    path_strength *= edge_data['strength']
                    path_confidence *= edge_data['confidence']
            
            results.append({
                'path': path,
                'length': len(path) - 1,
                'strength': path_strength,
                'confidence': path_confidence
            })
        
        # 排序
        results.sort(key=lambda x: x['strength'] * x['confidence'], reverse=True)
        
        return results
    
    async def analyze_sentiment_price_causality(self, sentiment_df: pd.DataFrame,
                                               price_df: pd.DataFrame) -> Dict[str, Any]:
        """
        分析情感与价格的因果关系
        
        参数:
            sentiment_df: 包含情感分数的DataFrame
            price_df: 包含价格数据的DataFrame
        
        返回:
            因果分析结果
        """
        # 对齐日期
        merged = pd.merge(sentiment_df, price_df, on='date', how='inner')
        
        if len(merged) < 20:
            return {'error': '数据不足，无法进行因果分析'}
        
        # 发现因果关系
        relationships = self.discover_causal_relationships(merged, 'close')
        
        # 检测领先指标
        leading_indicators = self.detect_leading_indicators(merged, 'close', look_ahead=3)
        
        # 分析滞后效应
        sentiment_cols = [col for col in sentiment_df.columns if col not in ['date', 'timestamp']]
        lag_analysis = {}
        for col in sentiment_cols:
            if col in merged.columns:
                lag_analysis[col] = self.analyze_time_lag_effects(merged, col, 'close', max_lag=7)
        
        return {
            'causal_relationships': relationships[:5],
            'leading_indicators': leading_indicators[:5],
            'lag_effects': lag_analysis,
            'sample_size': len(merged),
            'analysis_time': datetime.now().isoformat()
        }

# ==================== 全局实例 ====================

causal_discovery_engine = CausalDiscoveryEngine()

# ==================== 便捷函数 ====================

async def discover_causal_relationships(df: pd.DataFrame, target_col: str = 'close') -> List[Dict]:
    """便捷函数：发现因果关系"""
    return causal_discovery_engine.discover_causal_relationships(df, target_col)

async def analyze_sentiment_price_causality(sentiment_df: pd.DataFrame, 
                                           price_df: pd.DataFrame) -> Dict:
    """便捷函数：分析情感与价格的因果关系"""
    return await causal_discovery_engine.analyze_sentiment_price_causality(sentiment_df, price_df)