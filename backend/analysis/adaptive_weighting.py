"""
自适应权重模块 - 动态调整各因素权重
功能：根据市场状态自动学习最优权重配置
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import os

# ==================== 市场状态枚举 ====================
class MarketState(Enum):
    BULL = "bull"                   # 牛市
    BEAR = "bear"                   # 熊市
    SIDEWAYS = "sideways"           # 震荡市
    VOLATILE = "volatile"           # 高波动
    QUIET = "quiet"                 # 低波动
    TRENDING = "trending"           # 趋势明显
    UNCERTAIN = "uncertain"         # 不确定

# ==================== 策略类型 ====================
class StrategyType(Enum):
    SENTIMENT_BASED = "sentiment_based"
    TECHNICAL_BASED = "technical_based"
    FUNDAMENTAL_BASED = "fundamental_based"
    HYBRID = "hybrid"

# ==================== 配置 ====================
CONFIG = {
    'state_detection': {
        'trend_threshold': 0.02,      # 趋势判断阈值
        'volatility_threshold': 2.0,  # 波动率阈值（ATR/价格）
        'lookback_period': 20,        # 回溯周期
        'min_samples': 30             # 最小样本数
    },
    'weight_adaptation': {
        'learning_rate': 0.05,        # 学习率
        'discount_factor': 0.95,      # 折扣因子
        'exploration_rate': 0.1,      # 探索率
        'reward_scaling': 1.0         # 奖励缩放
    },
    'default_weights': {
        'sentiment': 0.3,
        'technical': 0.3,
        'momentum': 0.2,
        'volume': 0.1,
        'fundamental': 0.1
    }
}

class MarketStateDetector:
    """市场状态检测器"""
    
    def __init__(self):
        self.cache = {}
        print("✅ 市场状态检测器初始化成功")
    
    def detect_market_state(self, price_df: pd.DataFrame, 
                           sentiment_score: Optional[float] = None) -> Dict[str, Any]:
        """
        检测当前市场状态
        
        参数:
            price_df: 包含价格数据的DataFrame
            sentiment_score: 可选的情感分数
        
        返回:
            市场状态分析结果
        """
        if len(price_df) < CONFIG['state_detection']['min_samples']:
            return {
                'state': MarketState.UNCERTAIN.value,
                'confidence': 0.3,
                'reason': '数据不足'
            }
        
        # 计算关键指标
        close = price_df['close'].values
        volume = price_df.get('volume', np.ones(len(close))).values
        
        # 1. 趋势检测
        trend_score = self._calculate_trend_score(close)
        
        # 2. 波动率检测
        volatility_score = self._calculate_volatility_score(close)
        
        # 3. 成交量检测
        volume_score = self._calculate_volume_score(volume)
        
        # 4. 动量检测
        momentum_score = self._calculate_momentum_score(close)
        
        # 综合判断市场状态
        state, confidence = self._classify_state(
            trend_score, volatility_score, volume_score, sentiment_score
        )
        
        return {
            'state': state.value,
            'state_label': self._get_state_label(state),
            'confidence': confidence,
            'indicators': {
                'trend_score': trend_score,
                'volatility_score': volatility_score,
                'volume_score': volume_score,
                'momentum_score': momentum_score
            },
            'sentiment_score': sentiment_score
        }
    
    def _calculate_trend_score(self, close: np.ndarray) -> float:
        """计算趋势强度分数"""
        # 使用线性回归斜率
        x = np.arange(len(close))
        slope, _ = np.polyfit(x, close, 1)
        
        # 归一化
        price_range = np.max(close) - np.min(close)
        if price_range == 0:
            return 0.0
        
        normalized_slope = slope * len(close) / price_range
        return np.tanh(normalized_slope)  # 映射到[-1, 1]
    
    def _calculate_volatility_score(self, close: np.ndarray) -> float:
        """计算波动率分数"""
        # 计算ATR（简化版）
        high = close  # 用当日变化代替
        low = close
        tr = np.max([
            np.diff(close, prepend=close[0]),
            np.abs(np.diff(close, prepend=close[0]))
        ], axis=0)
        
        atr = np.mean(tr[-CONFIG['state_detection']['lookback_period']:])
        avg_price = np.mean(close[-CONFIG['state_detection']['lookback_period']:])
        
        return min(3.0, atr / avg_price * 100)  # 波动率百分比
    
    def _calculate_volume_score(self, volume: np.ndarray) -> float:
        """计算成交量分数"""
        recent_volume = volume[-CONFIG['state_detection']['lookback_period']:]
        avg_volume = np.mean(volume)
        
        if avg_volume == 0:
            return 1.0
        
        return np.mean(recent_volume) / avg_volume  # 相对于历史平均
    
    def _calculate_momentum_score(self, close: np.ndarray) -> float:
        """计算动量分数"""
        # 相对强弱指标（简化版）
        lookback = CONFIG['state_detection']['lookback_period']
        returns = np.diff(close) / close[:-1]
        
        if len(returns) < lookback:
            return 0.0
        
        recent_returns = returns[-lookback:]
        positive = np.sum(recent_returns > 0)
        negative = np.sum(recent_returns < 0)
        
        total = positive + negative
        if total == 0:
            return 0.0
        
        return (positive - negative) / total  # 动量方向和强度
    
    def _classify_state(self, trend_score: float, volatility_score: float,
                       volume_score: float, sentiment_score: Optional[float]) -> tuple:
        """
        综合分类市场状态
        
        参数:
            trend_score: 趋势分数 [-1, 1]
            volatility_score: 波动率分数 [0, 3]
            volume_score: 成交量分数
            sentiment_score: 情感分数（可选）
        
        返回:
            (市场状态, 置信度)
        """
        # 高波动状态
        if volatility_score > CONFIG['state_detection']['volatility_threshold']:
            return MarketState.VOLATILE, min(0.9, volatility_score / 3.0)
        
        # 低波动状态
        if volatility_score < 0.5:
            return MarketState.QUIET, 0.8
        
        # 趋势判断
        if abs(trend_score) > CONFIG['state_detection']['trend_threshold']:
            if trend_score > 0:
                return MarketState.BULL, min(0.9, abs(trend_score))
            else:
                return MarketState.BEAR, min(0.9, abs(trend_score))
        
        # 震荡状态
        return MarketState.SIDEWAYS, 0.7
    
    def _get_state_label(self, state: MarketState) -> str:
        """获取状态标签"""
        labels = {
            MarketState.BULL: '牛市',
            MarketState.BEAR: '熊市',
            MarketState.SIDEWAYS: '震荡市',
            MarketState.VOLATILE: '高波动',
            MarketState.QUIET: '低波动',
            MarketState.TRENDING: '趋势明显',
            MarketState.UNCERTAIN: '不确定'
        }
        return labels.get(state, '未知')

class AdaptiveWeightManager:
    """自适应权重管理器"""
    
    def __init__(self):
        self.weights = CONFIG['default_weights'].copy()
        self.history = []
        self.performance = defaultdict(list)
        self.market_state_detector = MarketStateDetector()
        print("✅ 自适应权重管理器初始化成功")
    
    def get_weights(self, market_state: str, 
                   context: Optional[Dict] = None) -> Dict[str, float]:
        """
        获取当前状态下的最优权重
        
        参数:
            market_state: 当前市场状态
            context: 可选的上下文信息
        
        返回:
            各因素权重
        """
        # 根据市场状态调整权重
        state_adjustments = self._get_state_adjustments(market_state)
        
        # 应用调整
        weights = {
            key: self.weights[key] * (1 + state_adjustments.get(key, 0))
            for key in self.weights
        }
        
        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {key: val / total for key, val in weights.items()}
        
        return weights
    
    def _get_state_adjustments(self, market_state: str) -> Dict[str, float]:
        """
        获取市场状态对应的权重调整
        
        参数:
            market_state: 市场状态
        
        返回:
            权重调整因子
        """
        adjustments = {
            MarketState.BULL.value: {
                'sentiment': 0.1,   # 牛市中情感更重要
                'momentum': 0.1,    # 动量效应增强
                'volume': 0.05      # 成交量确认
            },
            MarketState.BEAR.value: {
                'technical': 0.1,   # 技术支撑更重要
                'momentum': -0.1,   # 动量效应减弱
                'volume': 0.1       # 放量下跌是信号
            },
            MarketState.SIDEWAYS.value: {
                'technical': 0.15,  # 震荡市技术指标更有效
                'sentiment': -0.1,  # 情感信号噪音大
                'volume': 0.1       # 量能突破是信号
            },
            MarketState.VOLATILE.value: {
                'technical': 0.2,   # 高波动依赖技术指标
                'sentiment': -0.15, # 情感反应过度
                'momentum': -0.1    # 动量不可持续
            },
            MarketState.QUIET.value: {
                'sentiment': 0.2,   # 低波动时情感是先行指标
                'momentum': 0.1,    # 动量开始积聚
                'volume': -0.1      # 缩量无意义
            },
            MarketState.TRENDING.value: {
                'momentum': 0.2,    # 趋势明显时动量最重要
                'sentiment': 0.05,  # 情感确认趋势
                'technical': -0.05  # 技术指标滞后
            }
        }
        
        return adjustments.get(market_state, {})
    
    def update_weights(self, reward: float, context: Dict):
        """
        根据奖励信号更新权重
        
        参数:
            reward: 奖励信号（预测准确率或收益）
            context: 包含市场状态和预测特征的上下文
        """
        state = context.get('market_state', 'uncertain')
        current_weights = self.get_weights(state)
        
        # 计算权重梯度
        gradients = self._calculate_gradients(reward, current_weights, context)
        
        # 更新权重
        lr = CONFIG['weight_adaptation']['learning_rate']
        for key in self.weights:
            self.weights[key] += lr * gradients.get(key, 0)
        
        # 约束权重范围
        self.weights = {
            key: max(0.01, min(0.9, val))
            for key, val in self.weights.items()
        }
        
        # 归一化
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {key: val / total for key, val in self.weights.items()}
        
        # 记录历史
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'market_state': state,
            'weights': self.weights.copy(),
            'reward': reward
        })
        
        # 保存性能记录
        self.performance[state].append(reward)
    
    def _calculate_gradients(self, reward: float, weights: Dict, 
                            context: Dict) -> Dict[str, float]:
        """
        计算权重更新梯度
        
        参数:
            reward: 奖励信号
            weights: 当前权重
            context: 上下文信息
        
        返回:
            各权重的梯度
        """
        gradients = {}
        
        # 简单策略梯度：奖励越高，当前权重越应该增加
        for key in weights:
            # 特征重要性（来自上下文）
            feature_importance = context.get('feature_importance', {}).get(key, 1.0)
            
            # 梯度 = 奖励 * 特征重要性
            gradients[key] = reward * feature_importance * weights[key]
        
        return gradients
    
    def evaluate_performance(self, lookback: int = 100) -> Dict[str, float]:
        """
        评估当前权重配置的性能
        
        参数:
            lookback: 回溯评估周期
        
        返回:
            性能评估结果
        """
        recent = self.history[-lookback:] if lookback < len(self.history) else self.history
        
        if not recent:
            return {'mean_reward': 0.0, 'win_rate': 0.5, 'sharpe_ratio': 0.0}
        
        rewards = [h['reward'] for h in recent]
        mean_reward = np.mean(rewards)
        win_rate = sum(1 for r in rewards if r > 0) / len(rewards)
        
        # 计算夏普比率（简化版）
        if len(rewards) > 1:
            std_reward = np.std(rewards)
            sharpe_ratio = mean_reward / max(std_reward, 1e-8) if mean_reward != 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        return {
            'mean_reward': mean_reward,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'sample_count': len(recent)
        }
    
    def save_weights(self, filepath: str = 'weights.json'):
        """保存权重配置"""
        data = {
            'weights': self.weights,
            'history': self.history[-1000:],  # 只保存最近1000条
            'updated_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_weights(self, filepath: str = 'weights.json'):
        """加载权重配置"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.weights = data.get('weights', CONFIG['default_weights'])
            self.history = data.get('history', [])
            return True
        except FileNotFoundError:
            return False

class MultiFactorWeightOptimizer:
    """多因子权重优化器"""
    
    def __init__(self):
        self.market_state_detector = MarketStateDetector()
        self.weight_manager = AdaptiveWeightManager()
        print("✅ 多因子权重优化器初始化成功")
    
    def optimize_weights(self, price_df: pd.DataFrame, 
                        sentiment_score: Optional[float] = None,
                        features: Optional[Dict] = None) -> Dict[str, Any]:
        """
        优化权重配置
        
        参数:
            price_df: 价格数据
            sentiment_score: 情感分数
            features: 可选的特征数据
        
        返回:
            权重优化结果
        """
        # 1. 检测市场状态
        state_result = self.market_state_detector.detect_market_state(price_df, sentiment_score)
        market_state = state_result['state']
        
        # 2. 获取自适应权重
        weights = self.weight_manager.get_weights(market_state, features)
        
        # 3. 评估性能
        performance = self.weight_manager.evaluate_performance()
        
        return {
            'market_state': market_state,
            'state_label': state_result['state_label'],
            'state_confidence': state_result['confidence'],
            'weights': weights,
            'performance': performance,
            'indicators': state_result.get('indicators', {})
        }
    
    async def optimize_weights_async(self, price_df: pd.DataFrame,
                                    sentiment_score: Optional[float] = None,
                                    features: Optional[Dict] = None) -> Dict[str, Any]:
        """异步优化权重"""
        return self.optimize_weights(price_df, sentiment_score, features)

# ==================== 全局实例 ====================

market_state_detector = MarketStateDetector()
adaptive_weight_manager = AdaptiveWeightManager()
multi_factor_optimizer = MultiFactorWeightOptimizer()

# ==================== 便捷函数 ====================

def optimize_weights(price_df: pd.DataFrame, sentiment_score: Optional[float] = None,
                    features: Optional[Dict] = None) -> Dict:
    """便捷函数：优化权重"""
    return multi_factor_optimizer.optimize_weights(price_df, sentiment_score, features)

async def optimize_weights_async(price_df: pd.DataFrame, 
                                sentiment_score: Optional[float] = None,
                                features: Optional[Dict] = None) -> Dict:
    """异步便捷函数"""
    return await multi_factor_optimizer.optimize_weights_async(price_df, sentiment_score, features)

def detect_market_state(price_df: pd.DataFrame, sentiment_score: Optional[float] = None) -> Dict:
    """便捷函数：检测市场状态"""
    return market_state_detector.detect_market_state(price_df, sentiment_score)