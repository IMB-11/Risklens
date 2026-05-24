"""
推理引擎模块 - 强化推理能力和趋势预测

功能特性：
1. NLI自然语言推理 - 识别文本蕴含关系
2. 因果推理 - 分析事件因果关系
3. 上下文推理 - 理解复杂语境
4. 时序趋势预测 - 预测未来情感走向
5. 事件影响预测 - 预测事件对市场的影响
6. 知识图谱推理 - 基于图谱的逻辑推理
7. 多数据源融合 - 股票价格、成交量、资金流向
8. 数据驱动因果发现 - 格兰杰因果检验、互信息分析
9. 自适应权重机制 - 动态调整各因素权重
10. 相对表现分析 - 个股相对于大盘/行业的表现
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
from scipy.special import softmax
from collections import defaultdict, deque
from datetime import datetime, timedelta
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import pandas as pd

# 导入新增模块
from backend.data.stock_data_fetcher import stock_data_fetcher, fetch_stock_data_async
from backend.analysis.causal_discovery import causal_discovery_engine, analyze_sentiment_price_causality
from backend.analysis.adaptive_weighting import multi_factor_optimizer, optimize_weights_async
from backend.models.tft_predictor import time_series_predictor, prepare_time_series_data, train_predictor, predict_future
from backend.analysis.relative_performance import relative_performance_analyzer, analyze_relative_performance

# ==================== 配置参数 ====================
LOCAL_MODEL_PATHS = {
    "nli_deberta": "./nli-deberta-v3-base",
    "financial_sentiment": "./bert-base-chinese-finetuning-financial-news-sentiment-v2",
}

# ==================== 数据结构 ====================
class InferenceType(Enum):
    ENTAILMENT = "entailment"      # 蕴含
    CONTRADICTION = "contradiction" # 矛盾
    NEUTRAL = "neutral"             # 中性

class TrendDirection(Enum):
    UP = "up"                       # 上升
    DOWN = "down"                   # 下降
    STABLE = "stable"               # 稳定
    UNCERTAIN = "uncertain"         # 不确定

@dataclass
class InferenceResult:
    """推理结果"""
    premise: str
    hypothesis: str
    inference_type: InferenceType
    confidence: float
    explanation: str

@dataclass
class TrendPrediction:
    """趋势预测结果"""
    stock_name: str
    time_horizon: str               # 预测时间范围 (1h, 1d, 1w, 1m)
    direction: TrendDirection
    confidence: float
    prediction_score: float         # 预测分数 (-1到1)
    factors: List[str]              # 影响因素
    historical_data: Dict[str, Any] # 历史数据摘要

@dataclass
class EventImpact:
    """事件影响预测"""
    event_title: str
    event_time: datetime
    impact_score: float             # 影响分数 (-1到1)
    confidence: float
    affected_stocks: List[str]
    expected_duration: str          # 预期持续时间
    scenario_analysis: Dict[str, float]  # 情景分析

# ==================== 推理引擎核心 ====================

class NLIEngine:
    """自然语言推理引擎 - 识别文本蕴含关系"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.analyzer = None
        self._initialized = False
        
        self.label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        self._initialize()
    
    def _initialize(self):
        """初始化NLI模型"""
        try:
            model_path = LOCAL_MODEL_PATHS["nli_deberta"]
            
            if not __import__('os').path.exists(model_path):
                print(f"⚠️ NLI模型路径不存在: {model_path}")
                print("   将使用基于规则的推理作为备用方案")
                return
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=3,
                ignore_mismatched_sizes=True
            )
            self.model = self.model.to(self.device)
            
            self.analyzer = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            
            self._initialized = True
            print("✅ NLI推理引擎初始化成功")
            
        except Exception as e:
            print(f"⚠️ NLI引擎初始化失败: {str(e)[:100]}")
            print("   将使用基于规则的推理作为备用方案")
    
    def infer(self, premise: str, hypothesis: str) -> InferenceResult:
        """
        执行自然语言推理
        
        参数:
            premise: 前提（已知事实）
            hypothesis: 假设（待验证的结论）
        
        返回:
            InferenceResult: 推理结果
        """
        if self._initialized and self.analyzer:
            return self._infer_ml(premise, hypothesis)
        else:
            return self._infer_rule_based(premise, hypothesis)
    
    def _infer_ml(self, premise: str, hypothesis: str) -> InferenceResult:
        """使用机器学习模型进行推理"""
        # 构造NLI输入
        nli_input = f"{premise} [SEP] {hypothesis}"
        
        try:
            results = self.analyzer(nli_input)
            
            if results:
                # 获取最高置信度的预测
                scores = results[0]
                max_score = max(scores, key=lambda x: x['score'])
                inference_type_str = max_score['label']
                confidence = max_score['score']
                
                inference_type = InferenceType(inference_type_str)
                
                return InferenceResult(
                    premise=premise,
                    hypothesis=hypothesis,
                    inference_type=inference_type,
                    confidence=confidence,
                    explanation=self._generate_explanation(inference_type, premise, hypothesis, confidence)
                )
        except Exception as e:
            print(f"⚠️ ML推理失败: {e}")
        
        return self._infer_rule_based(premise, hypothesis)
    
    def _infer_rule_based(self, premise: str, hypothesis: str) -> InferenceResult:
        """基于规则的推理（备用方案）"""
        premise_lower = premise.lower()
        hypothesis_lower = hypothesis.lower()
        
        # 关键词匹配分析
        premise_tokens = set(premise_lower.replace('，', ' ').replace('。', ' ').split())
        hypothesis_tokens = set(hypothesis_lower.replace('，', ' ').replace('。', ' ').split())
        
        # 计算重叠度
        overlap = len(premise_tokens & hypothesis_tokens)
        hypothesis_coverage = overlap / max(len(hypothesis_tokens), 1)
        
        # 检查否定词
        negation_words = {"不", "没有", "无", "非", "未", "从未", "绝不"}
        premise_has_negation = any(word in premise_lower for word in negation_words)
        hypothesis_has_negation = any(word in hypothesis_lower for word in negation_words)
        
        # 推理逻辑
        if hypothesis_coverage > 0.7:
            # 高度重叠
            if premise_has_negation == hypothesis_has_negation:
                # 否定词数量相同 -> 蕴含
                return InferenceResult(
                    premise=premise,
                    hypothesis=hypothesis,
                    inference_type=InferenceType.ENTAILMENT,
                    confidence=0.7 + hypothesis_coverage * 0.2,
                    explanation=f"假设内容与前提高度一致（关键词覆盖率{hypothesis_coverage:.1%}）"
                )
            else:
                # 否定词数量不同 -> 矛盾
                return InferenceResult(
                    premise=premise,
                    hypothesis=hypothesis,
                    inference_type=InferenceType.CONTRADICTION,
                    confidence=0.6 + hypothesis_coverage * 0.2,
                    explanation="前提和假设中的否定词存在矛盾"
                )
        elif hypothesis_coverage > 0.3:
            # 部分重叠
            return InferenceResult(
                premise=premise,
                hypothesis=hypothesis,
                inference_type=InferenceType.NEUTRAL,
                confidence=0.5 + hypothesis_coverage * 0.3,
                explanation=f"假设与前提部分相关（关键词覆盖率{hypothesis_coverage:.1%}），但无法确定蕴含关系"
            )
        else:
            # 几乎无重叠
            return InferenceResult(
                premise=premise,
                hypothesis=hypothesis,
                inference_type=InferenceType.NEUTRAL,
                confidence=0.4,
                explanation="假设与前提关联性较弱，无法进行有效推理"
            )
    
    def _generate_explanation(self, inference_type: InferenceType, premise: str, 
                             hypothesis: str, confidence: float) -> str:
        """生成推理解释"""
        explanations = {
            InferenceType.ENTAILMENT: f"根据前提「{premise[:30]}...」，可以推断假设「{hypothesis[:30]}...」是成立的（置信度{confidence:.1%}）",
            InferenceType.CONTRADICTION: f"前提「{premise[:30]}...」与假设「{hypothesis[:30]}...」存在矛盾（置信度{confidence:.1%}）",
            InferenceType.NEUTRAL: f"前提「{premise[:30]}...」无法确定假设「{hypothesis[:30]}...」的真假（置信度{confidence:.1%}）"
        }
        return explanations.get(inference_type, "无法生成解释")

class CausalReasoner:
    """因果推理器 - 分析事件因果关系"""
    
    def __init__(self):
        self.causal_patterns = [
            # (触发词, 结果词, 因果强度)
            ("因为", "所以", 0.9),
            ("由于", "导致", 0.9),
            ("引发", "使得", 0.85),
            ("造成", "从而", 0.85),
            ("基于", "因此", 0.8),
            ("根据", "得出", 0.75),
            ("随着", "将会", 0.7),
            ("如果", "那么", 0.85),
            ("一旦", "就会", 0.8),
            ("除非", "否则", 0.75),
        ]
        
        self.positive_effects = [
            "增长", "上升", "提高", "增加", "利好", "上涨", "突破",
            "创新高", "盈利", "收益", "分红", "回购", "增持", "看好"
        ]
        
        self.negative_effects = [
            "下跌", "下降", "减少", "亏损", "利空", "暴跌", "跌破",
            "减持", "卖出", "看空", "风险", "警告", "下调", "违约"
        ]
    
    def analyze_causality(self, text: str) -> List[Dict[str, Any]]:
        """
        分析文本中的因果关系
        
        参数:
            text: 待分析文本
        
        返回:
            因果关系列表
        """
        results = []
        
        for trigger, result, base_strength in self.causal_patterns:
            if trigger in text and result in text:
                # 找到因果结构
                trigger_idx = text.index(trigger)
                result_idx = text.index(result)
                
                if trigger_idx < result_idx:
                    cause = text[trigger_idx + len(trigger):result_idx].strip()
                    effect = text[result_idx + len(result):].strip()
                    
                    if cause and effect:
                        # 计算效果极性
                        polarity = self._detect_polarity(effect)
                        
                        results.append({
                            "cause": cause[:50],
                            "effect": effect[:50],
                            "trigger_pair": (trigger, result),
                            "strength": base_strength,
                            "polarity": polarity,
                            "confidence": min(0.95, base_strength + 0.05 if polarity else base_strength)
                        })
        
        # 基于关键词的因果分析
        keyword_results = self._analyze_by_keywords(text)
        results.extend(keyword_results)
        
        return sorted(results, key=lambda x: x['confidence'], reverse=True)
    
    def _detect_polarity(self, text: str) -> str:
        """检测文本极性"""
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.positive_effects if word in text_lower)
        negative_count = sum(1 for word in self.negative_effects if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _analyze_by_keywords(self, text: str) -> List[Dict[str, Any]]:
        """基于关键词的因果分析"""
        results = []
        text_lower = text.lower()
        
        # 查找潜在的因果关系
        for positive in self.positive_effects:
            if positive in text_lower:
                # 找到正向效果的上下文
                results.append({
                    "cause": self._extract_context(text, positive, window=20),
                    "effect": positive,
                    "trigger_pair": ("keyword", "positive"),
                    "strength": 0.6,
                    "polarity": "positive",
                    "confidence": 0.65
                })
        
        for negative in self.negative_effects:
            if negative in text_lower:
                results.append({
                    "cause": self._extract_context(text, negative, window=20),
                    "effect": negative,
                    "trigger_pair": ("keyword", "negative"),
                    "strength": 0.6,
                    "polarity": "negative",
                    "confidence": 0.65
                })
        
        return results
    
    def _extract_context(self, text: str, target: str, window: int = 20) -> str:
        """提取目标词的上下文"""
        idx = text.lower().index(target.lower())
        start = max(0, idx - window)
        end = min(len(text), idx + len(target) + window)
        return text[start:end]

class TemporalTrendPredictor:
    """时序趋势预测器 - 预测未来情感走向"""
    
    def __init__(self):
        self.nli_engine = NLIEngine()
        self.causal_reasoner = CausalReasoner()
        
        # 趋势预测参数
        self.short_term_window = 3   # 短期窗口（小时）
        self.medium_term_window = 24  # 中期窗口（小时）
        self.long_term_window = 168   # 长期窗口（小时=7天）
        
        # 衰减因子
        self.time_decay = 0.95       # 时间衰减系数
        self.sentiment_momentum = 0.3 # 情感动量系数
    
    async def predict_trend(self, stock_name: str, news_items: List[Dict],
                           time_horizon: str = "1d") -> TrendPrediction:
        """
        预测股票未来趋势
        
        参数:
            stock_name: 股票名称
            news_items: 新闻数据列表
            time_horizon: 预测时间范围 (1h, 1d, 1w, 1m)
        
        返回:
            TrendPrediction: 趋势预测结果
        """
        if not news_items:
            return TrendPrediction(
                stock_name=stock_name,
                time_horizon=time_horizon,
                direction=TrendDirection.UNCERTAIN,
                confidence=0.1,
                prediction_score=0.0,
                factors=["暂无足够数据进行预测"],
                historical_data={}
            )
        
        # 1. 分析历史情感趋势
        historical_trend = self._analyze_historical_trend(news_items)
        
        # 2. 识别关键事件和因果关系
        causal_factors = self._extract_causal_factors(news_items)
        
        # 3. 计算预测分数
        prediction_score, confidence = self._compute_prediction(
            news_items, historical_trend, causal_factors, time_horizon
        )
        
        # 4. 确定趋势方向
        direction = self._determine_direction(prediction_score, confidence)
        
        # 5. 生成影响因素说明
        factors = self._generate_factors_description(causal_factors, historical_trend)
        
        return TrendPrediction(
            stock_name=stock_name,
            time_horizon=time_horizon,
            direction=direction,
            confidence=confidence,
            prediction_score=prediction_score,
            factors=factors,
            historical_data=historical_trend
        )
    
    def _analyze_historical_trend(self, news_items: List[Dict]) -> Dict[str, Any]:
        """分析历史情感趋势"""
        # 按时间排序（假设新闻有时间戳）
        sorted_news = sorted(news_items, key=lambda x: x.get('publish_time', ''), reverse=True)
        
        # 计算时间窗口内的情感分布
        window_sizes = [3, 6, 12]  # 最近3条、6条、12条
        trend_data = {}
        
        for window in window_sizes:
            recent = sorted_news[:window]
            if not recent:
                continue
            
            pos = sum(1 for n in recent if n.get('sentiment_type') == 'positive')
            neg = sum(1 for n in recent if n.get('sentiment_type') == 'negative')
            total = len(recent)
            
            trend_data[f'window_{window}'] = {
                'positive': pos,
                'negative': neg,
                'neutral': total - pos - neg,
                'ratio': (pos - neg) / max(total, 1)
            }
        
        # 计算趋势变化
        if len(window_sizes) >= 2:
            short = trend_data.get(f'window_{window_sizes[0]}', {}).get('ratio', 0)
            medium = trend_data.get(f'window_{window_sizes[1]}', {}).get('ratio', 0)
            
            trend_data['momentum'] = short - medium  # 动量变化
            trend_data['trend_strength'] = abs(trend_data['momentum'])
        
        return trend_data
    
    def _extract_causal_factors(self, news_items: List[Dict]) -> List[Dict]:
        """提取因果因素"""
        factors = []
        
        for news in news_items[:10]:  # 只分析前10条重要新闻
            content = news.get('content', '') + news.get('title', '')
            
            # 使用因果推理器分析
            causal_results = self.causal_reasoner.analyze_causality(content)
            
            for causal in causal_results:
                factors.append({
                    'cause': causal['cause'],
                    'effect': causal['effect'],
                    'polarity': causal['polarity'],
                    'confidence': causal['confidence'],
                    'source': news.get('source', 'unknown'),
                    'weight': news.get('weight', 0.5)
                })
        
        # 按权重和置信度排序
        factors.sort(key=lambda x: x['weight'] * x['confidence'], reverse=True)
        return factors[:5]  # 取前5个最重要的因素
    
    def _compute_prediction(self, news_items: List[Dict], historical_trend: Dict,
                           causal_factors: List[Dict], time_horizon: str) -> Tuple[float, float]:
        """计算预测分数和置信度"""
        # 1. 基础情感分数
        pos_count = sum(1 for n in news_items if n.get('sentiment_type') == 'positive')
        neg_count = sum(1 for n in news_items if n.get('sentiment_type') == 'negative')
        total = len(news_items)
        
        base_score = (pos_count - neg_count) / max(total, 1)
        
        # 2. 时间加权
        time_weighted_score = 0.0
        total_weight = 0.0
        
        for i, news in enumerate(sorted(news_items, key=lambda x: x.get('publish_time', ''), reverse=True)[:10]):
            decay = self.time_decay ** i  # 越新权重越高
            sentiment = 1 if news.get('sentiment_type') == 'positive' else (-1 if news.get('sentiment_type') == 'negative' else 0)
            weight = news.get('weight', 0.5)
            
            time_weighted_score += sentiment * decay * weight
            total_weight += decay * weight
        
        time_weighted_score = time_weighted_score / max(total_weight, 1)
        
        # 3. 因果因素影响
        causal_score = 0.0
        for factor in causal_factors:
            polarity = 1 if factor['polarity'] == 'positive' else (-1 if factor['polarity'] == 'negative' else 0)
            causal_score += polarity * factor['confidence'] * factor['weight']
        
        causal_score = causal_score / max(len(causal_factors), 1)
        
        # 4. 动量影响
        momentum = historical_trend.get('momentum', 0)
        
        # 5. 综合分数
        weights = {
            'base': 0.2,
            'time_weighted': 0.4,
            'causal': 0.3,
            'momentum': 0.1
        }
        
        prediction_score = (
            base_score * weights['base'] +
            time_weighted_score * weights['time_weighted'] +
            causal_score * weights['causal'] +
            momentum * weights['momentum']
        )
        
        # 6. 计算置信度
        data_quality = sum(n.get('weight', 0.5) for n in news_items) / max(len(news_items), 1)
        factor_diversity = min(len(causal_factors) / 5, 1.0)
        trend_consistency = min(abs(base_score - time_weighted_score) * 10, 1.0)
        
        confidence = (
            data_quality * 0.4 +
            factor_diversity * 0.3 +
            trend_consistency * 0.3
        )
        
        # 根据时间范围调整置信度
        horizon_adjustment = {
            '1h': 1.1,
            '1d': 1.0,
            '1w': 0.7,
            '1m': 0.4
        }
        
        confidence = min(0.99, confidence * horizon_adjustment.get(time_horizon, 1.0))
        
        return prediction_score, confidence
    
    def _determine_direction(self, prediction_score: float, confidence: float) -> TrendDirection:
        """确定趋势方向"""
        threshold = 0.15  # 决策阈值
        
        if confidence < 0.3:
            return TrendDirection.UNCERTAIN
        
        if prediction_score > threshold:
            return TrendDirection.UP
        elif prediction_score < -threshold:
            return TrendDirection.DOWN
        else:
            return TrendDirection.STABLE
    
    def _generate_factors_description(self, causal_factors: List[Dict],
                                     historical_trend: Dict) -> List[str]:
        """生成影响因素说明"""
        factors = []
        
        # 基于因果因素
        for factor in causal_factors:
            effect_type = {"positive": "利好", "negative": "利空", "neutral": "中性"}.get(factor['polarity'], "中性")
            factors.append(f"{effect_type}: {factor['cause']} -> {factor['effect']}")
        
        # 基于历史趋势
        momentum = historical_trend.get('momentum', 0)
        if momentum > 0.2:
            factors.append("情感趋势正在加速上升")
        elif momentum < -0.2:
            factors.append("情感趋势正在加速下降")
        
        # 数据质量提示
        if len(causal_factors) == 0:
            factors.append("数据量较少，建议持续关注")
        
        return factors[:5]

class EventImpactPredictor:
    """事件影响预测器 - 预测事件对市场的影响"""
    
    def __init__(self):
        self.event_templates = {
            "earnings": {
                "keywords": ["财报", "业绩", "盈利", "净利润", "营收"],
                "impact_duration": "2-5天",
                "confidence_factor": 0.85
            },
            "regulation": {
                "keywords": ["政策", "监管", "新规", "条例", "通知"],
                "impact_duration": "1-2周",
                "confidence_factor": 0.9
            },
            "merger": {
                "keywords": ["并购", "重组", "收购", "入股", "战略合作"],
                "impact_duration": "1-3周",
                "confidence_factor": 0.75
            },
            "product": {
                "keywords": ["新品", "发布", "上市", "技术", "突破"],
                "impact_duration": "3-7天",
                "confidence_factor": 0.7
            },
            "management": {
                "keywords": ["人事", "变动", "辞职", "任命", "CEO"],
                "impact_duration": "1-3天",
                "confidence_factor": 0.65
            },
            "scandal": {
                "keywords": ["丑闻", "造假", "违规", "调查", "处罚"],
                "impact_duration": "1-2周",
                "confidence_factor": 0.85
            }
        }
        
        self.impact_multipliers = {
            "positive": 0.8,
            "negative": -0.9,
            "neutral": 0.1
        }
    
    def predict_event_impact(self, event_title: str, event_content: str,
                            related_stocks: List[str]) -> EventImpact:
        """
        预测事件影响
        
        参数:
            event_title: 事件标题
            event_content: 事件内容
            related_stocks: 相关股票列表
        
        返回:
            EventImpact: 事件影响预测
        """
        # 1. 识别事件类型
        event_type, confidence = self._identify_event_type(event_title, event_content)
        
        # 2. 分析情感极性
        sentiment = self._analyze_event_sentiment(event_content)
        
        # 3. 计算影响分数
        base_impact = self.impact_multipliers.get(sentiment, 0.1)
        impact_score = base_impact * confidence * self.event_templates.get(event_type, {}).get('confidence_factor', 0.7)
        
        # 4. 确定影响持续时间
        duration = self.event_templates.get(event_type, {}).get('impact_duration', "3-7天")
        
        # 5. 情景分析
        scenario_analysis = self._generate_scenario_analysis(impact_score, confidence)
        
        return EventImpact(
            event_title=event_title,
            event_time=datetime.now(),
            impact_score=impact_score,
            confidence=confidence,
            affected_stocks=related_stocks,
            expected_duration=duration,
            scenario_analysis=scenario_analysis
        )
    
    def _identify_event_type(self, title: str, content: str) -> Tuple[str, float]:
        """识别事件类型"""
        text = title + content
        best_type = "other"
        best_score = 0.0
        
        for event_type, template in self.event_templates.items():
            score = sum(1 for kw in template['keywords'] if kw in text) / len(template['keywords'])
            
            if score > best_score:
                best_score = score
                best_type = event_type
        
        return best_type, best_score
    
    def _analyze_event_sentiment(self, content: str) -> str:
        """分析事件情感"""
        positive_words = ["利好", "增长", "突破", "创新高", "超预期", "强劲", "稳健"]
        negative_words = ["利空", "下滑", "不及预期", "亏损", "暴跌", "风险", "警告"]
        
        pos_count = sum(1 for w in positive_words if w in content)
        neg_count = sum(1 for w in negative_words if w in content)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"
    
    def _generate_scenario_analysis(self, impact_score: float, confidence: float) -> Dict[str, float]:
        """生成情景分析"""
        base_score = impact_score
        volatility = (1 - confidence) * 0.3  # 不确定性带来的波动
        
        return {
            "best_case": min(1.0, base_score + volatility),
            "base_case": base_score,
            "worst_case": max(-1.0, base_score - volatility)
        }

class KnowledgeGraphReasoner:
    """知识图谱推理器 - 基于图谱的逻辑推理"""
    
    def __init__(self):
        # 构建简易金融知识图谱
        self.graph = {
            "贵州茅台": {
                "related_stocks": ["五粮液", "泸州老窖", "洋河股份"],
                "industry": "白酒",
                "concepts": ["消费", "高端白酒", "MSCI"],
                "suppliers": ["高粱供应商", "包装材料商"],
                "competitors": ["五粮液", "洋河股份"]
            },
            "比亚迪": {
                "related_stocks": ["宁德时代", "特斯拉", "小鹏汽车"],
                "industry": "新能源汽车",
                "concepts": ["新能源", "电池", "智能驾驶"],
                "suppliers": ["宁德时代", "半导体厂商"],
                "competitors": ["特斯拉", "蔚来", "小鹏汽车"]
            },
            "宁德时代": {
                "related_stocks": ["比亚迪", "赣锋锂业", "天齐锂业"],
                "industry": "动力电池",
                "concepts": ["新能源", "储能", "锂电池"],
                "suppliers": ["锂矿企业", "石墨电极商"],
                "competitors": ["比亚迪", "国轩高科"]
            },
            "白酒": {
                "related_concepts": ["消费复苏", "高端消费", "通胀预期"],
                "sensitive_to": ["货币政策", "消费信心", "疫情防控"]
            },
            "新能源汽车": {
                "related_concepts": ["碳中和", "汽车智能化", "补贴政策"],
                "sensitive_to": ["电池价格", "芯片供应", "油价"]
            },
            "货币政策": {
                "indicators": ["CPI", "PPI", "M2", "利率"]
            }
        }
        
        # 关系传播规则
        self.propagation_rules = [
            ("related_stocks", 0.7, "同行业关联"),
            ("competitors", -0.5, "竞争关系"),
            ("suppliers", 0.4, "供应链关联"),
            ("industry", 0.6, "行业整体影响")
        ]
    
    def infer_related_impact(self, stock_name: str, event_sentiment: str) -> List[Dict]:
        """
        基于知识图谱推理相关股票的影响
        
        参数:
            stock_name: 股票名称
            event_sentiment: 事件情感 (positive/negative/neutral)
        
        返回:
            相关股票影响列表
        """
        results = []
        
        if stock_name not in self.graph:
            return results
        
        stock_data = self.graph[stock_name]
        base_multiplier = 1 if event_sentiment == "positive" else (-1 if event_sentiment == "negative" else 0)
        
        for relation, weight, description in self.propagation_rules:
            related_items = stock_data.get(relation, [])
            
            for item in related_items:
                impact = {
                    "target": item,
                    "relation": relation,
                    "description": description,
                    "expected_impact": base_multiplier * weight,
                    "confidence": 0.6 + abs(weight) * 0.2
                }
                results.append(impact)
        
        # 按影响大小排序
        results.sort(key=lambda x: abs(x['expected_impact']), reverse=True)
        return results[:6]

# ==================== 推理引擎主类 ====================

class AdvancedInferenceEngine:
    """高级推理引擎 - 整合所有推理能力"""
    
    def __init__(self):
        self.nli_engine = NLIEngine()
        self.causal_reasoner = CausalReasoner()
        self.trend_predictor = TemporalTrendPredictor()
        self.event_impact_predictor = EventImpactPredictor()
        self.knowledge_graph_reasoner = KnowledgeGraphReasoner()
        
        print("✅ 高级推理引擎初始化完成")
        print("   已加载模块: NLI推理、因果推理、趋势预测、事件影响、知识图谱")
    
    async def analyze_and_predict(self, stock_name: str, news_items: List[Dict],
                                  risk_level: str = "moderate") -> Dict[str, Any]:
        """
        综合分析和预测
        
        参数:
            stock_name: 股票名称
            news_items: 新闻数据列表
            risk_level: 风险偏好
        
        返回:
            综合分析结果
        """
        results = {
            "stock_name": stock_name,
            "inference_results": [],
            "trend_predictions": {},
            "event_impacts": [],
            "knowledge_graph_inferences": [],
            "reasoning_meta": {},
            "summary": {}
        }
        
        # 1. NLI推理分析（对重要新闻进行推理验证）
        for news in news_items[:3]:
            premise = news.get('content', '')[:100]
            hypothesis = news.get('title', '')
            
            if premise and hypothesis:
                inference = self.nli_engine.infer(premise, hypothesis)
                results["inference_results"].append({
                    "title": news.get('title', ''),
                    "inference_type": inference.inference_type.value,
                    "confidence": inference.confidence,
                    "explanation": inference.explanation
                })
        
        # 2. 多时间范围趋势预测
        for horizon in ["1h", "1d", "1w"]:
            prediction = await self.trend_predictor.predict_trend(
                stock_name, news_items, horizon
            )
            results["trend_predictions"][horizon] = {
                "direction": prediction.direction.value,
                "confidence": prediction.confidence,
                "prediction_score": prediction.prediction_score,
                "factors": prediction.factors,
                "historical_data": prediction.historical_data
            }
        
        # 3. 事件影响分析
        for news in news_items[:5]:
            title = news.get('title', '')
            content = news.get('content', '')
            sentiment = news.get('sentiment_type', 'neutral')
            
            related_stocks = self.knowledge_graph_reasoner.infer_related_impact(stock_name, sentiment)
            related_stock_names = [r['target'] for r in related_stocks[:3]]
            
            impact = self.event_impact_predictor.predict_event_impact(
                title, content, related_stock_names
            )
            results["event_impacts"].append({
                "title": title,
                "impact_score": impact.impact_score,
                "confidence": impact.confidence,
                "expected_duration": impact.expected_duration,
                "affected_stocks": impact.affected_stocks,
                "scenario_analysis": impact.scenario_analysis
            })
        
        # 4. 知识图谱推理
        overall_sentiment = self._compute_overall_sentiment(news_items)
        kg_results = self.knowledge_graph_reasoner.infer_related_impact(stock_name, overall_sentiment)
        results["knowledge_graph_inferences"] = kg_results

        # 5. 结构化推理元信息（供风控引擎二次融合）
        results["reasoning_meta"] = self._build_reasoning_meta(results)

        # 6. 生成摘要
        results["summary"] = self._generate_summary(results, risk_level)
        
        return results
    
    def _compute_overall_sentiment(self, news_items: List[Dict]) -> str:
        """计算整体情感"""
        pos = sum(1 for n in news_items if n.get('sentiment_type') == 'positive')
        neg = sum(1 for n in news_items if n.get('sentiment_type') == 'negative')
        
        if pos > neg:
            return "positive"
        elif neg > pos:
            return "negative"
        else:
            return "neutral"
    
    def _generate_summary(self, results: Dict, risk_level: str) -> Dict:
        """生成分析摘要"""
        # 提取关键指标
        trend_1d = results["trend_predictions"].get("1d", {})
        event_impacts = results["event_impacts"]
        kg_inferences = results["knowledge_graph_inferences"]
        reasoning_meta = results.get("reasoning_meta", {}) or {}
        
        # 综合判断
        main_direction = trend_1d.get("direction", "uncertain")
        confidence = trend_1d.get("confidence", 0.5)
        
        # 风险调整
        risk_adjustment = {"aggressive": 0.1, "moderate": 0.0, "conservative": -0.1}.get(risk_level, 0.0)
        adjusted_confidence = min(0.99, max(0.01, confidence + risk_adjustment))
        
        # 最可能的情景
        scenarios = []
        for event in event_impacts[:3]:
            scenario = event.get("scenario_analysis", {})
            if scenario:
                scenarios.append({
                    "event": event["title"][:20],
                    "best": scenario.get("best_case", 0),
                    "base": scenario.get("base_case", 0),
                    "worst": scenario.get("worst_case", 0)
                })
        
        return {
            "main_trend": main_direction,
            "confidence": adjusted_confidence,
            "key_events_count": len(event_impacts),
            "related_stocks_impacted": len(kg_inferences),
            "risk_adjustment": risk_adjustment,
            "scenarios": scenarios,
            "reasoning_health": round(reasoning_meta.get("reasoning_health", 0.5), 6),
            "trend_divergence": round(reasoning_meta.get("trend_divergence", 0.0), 6),
            "nli_contradiction_ratio": round(reasoning_meta.get("nli_contradiction_ratio", 0.0), 6),
            "scenario_tail_risk": round(reasoning_meta.get("scenario_tail_risk", 0.0), 6),
        }

    def _build_reasoning_meta(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        构建推理元信号：
        1) NLI一致性（矛盾占比）
        2) 多周期趋势分歧
        3) 事件情景尾部风险（best/base/worst差异）
        """
        inferences = results.get("inference_results", []) or []
        trend_predictions = results.get("trend_predictions", {}) or {}
        event_impacts = results.get("event_impacts", []) or []

        nli_total = len(inferences)
        contradiction = 0
        entailment = 0
        nli_conf_sum = 0.0
        for item in inferences:
            inf_type = str(item.get("inference_type", "")).lower()
            conf = float(item.get("confidence", 0.5) or 0.5)
            nli_conf_sum += conf
            if inf_type == "contradiction":
                contradiction += 1
            elif inf_type == "entailment":
                entailment += 1

        contradiction_ratio = contradiction / max(nli_total, 1)
        entailment_ratio = entailment / max(nli_total, 1)
        nli_confidence = nli_conf_sum / max(nli_total, 1) if nli_total else 0.5

        # 多周期趋势分歧：方向一致性越差，分歧越高。
        dirs = []
        for horizon in ("1h", "1d", "1w"):
            direction = str((trend_predictions.get(horizon, {}) or {}).get("direction", "uncertain")).lower()
            if direction in {"up", "down", "stable"}:
                dirs.append(direction)
        trend_divergence = 0.0
        if dirs:
            up = sum(1 for d in dirs if d == "up")
            down = sum(1 for d in dirs if d == "down")
            stable = sum(1 for d in dirs if d == "stable")
            mode_ratio = max(up, down, stable) / len(dirs)
            trend_divergence = max(0.0, 1.0 - mode_ratio)

        # 事件尾部风险：worst相对best/base的下行幅度。
        tail_scores = []
        for ev in event_impacts[:8]:
            scenario = ev.get("scenario_analysis", {}) or {}
            best = float(scenario.get("best_case", 0.0) or 0.0)
            base = float(scenario.get("base_case", 0.0) or 0.0)
            worst = float(scenario.get("worst_case", 0.0) or 0.0)
            drawdown = max(0.0, max(best - worst, base - worst))
            tail_scores.append(min(1.0, drawdown / 0.12))
        scenario_tail_risk = sum(tail_scores) / len(tail_scores) if tail_scores else 0.0

        # reasoning_health越高表示推理链条更一致、更可解释。
        reasoning_health = max(
            0.0,
            min(
                1.0,
                0.42 * (1.0 - contradiction_ratio) +
                0.28 * (1.0 - trend_divergence) +
                0.18 * nli_confidence +
                0.12 * entailment_ratio
            ),
        )

        return {
            "nli_total": float(nli_total),
            "nli_contradiction_ratio": contradiction_ratio,
            "nli_entailment_ratio": entailment_ratio,
            "nli_confidence": nli_confidence,
            "trend_divergence": trend_divergence,
            "scenario_tail_risk": scenario_tail_risk,
            "reasoning_health": reasoning_health,
        }

# ==================== 增强版推理引擎（集成多数据源）====================

class EnhancedInferenceEngine:
    """
    增强版推理引擎 - 集成多数据源、因果发现、自适应权重、时序预测
    """
    
    def __init__(self):
        self.basic_engine = AdvancedInferenceEngine()
        print("✅ 增强版推理引擎初始化完成")
        print("   已集成: 多数据源融合、因果发现、自适应权重、时序预测、相对表现分析")
    
    async def analyze_comprehensive(self, stock_name: str, news_items: List[Dict],
                                   risk_level: str = "moderate", 
                                   days: int = 60) -> Dict[str, Any]:
        """
        综合分析 - 集成所有增强功能
        
        参数:
            stock_name: 股票名称
            news_items: 新闻数据列表
            risk_level: 风险偏好
            days: 历史数据天数
        
        返回:
            综合分析结果
        """
        results = {
            "stock_name": stock_name,
            "analysis_time": datetime.now().isoformat(),
            "basic_inference": None,
            "stock_data": None,
            "technical_indicators": None,
            "causal_relationships": None,
            "adaptive_weights": None,
            "temporal_prediction": None,
            "relative_performance": None,
            "final_recommendation": None
        }
        
        # 1. 获取股票基本数据
        try:
            stock_summary = await fetch_stock_data_async(stock_name, days)
            results["stock_data"] = stock_summary
            
            # 获取K线数据用于后续分析
            stock_code = stock_data_fetcher.get_stock_code(stock_name)
            kline_df = stock_data_fetcher.get_kline_data(stock_code, 
                                                        start_date=(datetime.now() - timedelta(days=days)).strftime('%Y%m%d'))
            results["kline_data"] = kline_df.to_dict('records') if not kline_df.empty else []
            
        except Exception as e:
            print(f"⚠️ 获取股票数据失败: {e}")
            results["stock_data_error"] = str(e)
        
        # 2. 基础推理分析
        try:
            basic_result = await self.basic_engine.analyze_and_predict(stock_name, news_items, risk_level)
            results["basic_inference"] = basic_result
        except Exception as e:
            print(f"⚠️ 基础推理失败: {e}")
            results["basic_inference_error"] = str(e)
        
        # 3. 因果发现分析（如果有足够数据）
        if not kline_df.empty and news_items:
            try:
                # 准备情感数据
                sentiment_df = self._prepare_sentiment_df(news_items, kline_df)
                
                # 分析情感与价格的因果关系
                causal_result = await analyze_sentiment_price_causality(sentiment_df, kline_df)
                results["causal_relationships"] = causal_result
                
                # 发现数据中的因果关系
                if not kline_df.empty:
                    causal_factors = causal_discovery_engine.discover_causal_relationships(
                        kline_df, target_col='close'
                    )
                    results["data_causal_factors"] = causal_factors[:5]
                    
            except Exception as e:
                print(f"⚠️ 因果分析失败: {e}")
                results["causal_analysis_error"] = str(e)
        
        # 4. 自适应权重优化
        if not kline_df.empty:
            try:
                # 计算平均情感分数
                avg_sentiment = self._calculate_average_sentiment(news_items)
                
                # 优化权重
                weight_result = await optimize_weights_async(kline_df, avg_sentiment)
                results["adaptive_weights"] = weight_result
                
            except Exception as e:
                print(f"⚠️ 权重优化失败: {e}")
                results["weight_optimization_error"] = str(e)
        
        # 5. 时序预测
        if not kline_df.empty:
            try:
                # 准备时序数据
                sentiment_df = self._prepare_sentiment_df(news_items, kline_df)
                ts_data = prepare_time_series_data(kline_df, sentiment_df)
                
                # 训练并预测
                train_result = train_predictor(ts_data, target_col='close')
                
                if train_result.get('success', False):
                    # 预测未来7天
                    prediction = predict_future(ts_data, steps=7)
                    results["temporal_prediction"] = prediction
                    
                    # 趋势分析
                    trend_analysis = time_series_predictor.analyze_trend(ts_data)
                    results["trend_analysis"] = trend_analysis
                else:
                    # 使用基线预测
                    prediction = predict_future(kline_df, steps=7)
                    results["temporal_prediction"] = prediction
                    results["temporal_method"] = "baseline"
                    
            except Exception as e:
                print(f"⚠️ 时序预测失败: {e}")
                results["temporal_prediction_error"] = str(e)
        
        # 6. 相对表现分析
        if not kline_df.empty:
            try:
                # 获取大盘数据
                index_df = stock_data_fetcher.get_index_data('沪深300', 
                                                            start_date=(datetime.now() - timedelta(days=days)).strftime('%Y%m%d'))
                
                # 分析相对表现
                relative_result = analyze_relative_performance(kline_df, index_df)
                results["relative_performance"] = relative_result
                
                # 市场定位分析
                positioning_result = relative_performance_analyzer.analyze_market_positioning(kline_df, index_df)
                results["market_positioning"] = positioning_result
                
            except Exception as e:
                print(f"⚠️ 相对表现分析失败: {e}")
                results["relative_performance_error"] = str(e)
        
        # 7. 生成最终建议
        results["final_recommendation"] = self._generate_final_recommendation(results, risk_level)
        
        return results
    
    def _prepare_sentiment_df(self, news_items: List[Dict], kline_df: pd.DataFrame) -> pd.DataFrame:
        """准备情感数据DataFrame"""
        if not news_items:
            return pd.DataFrame()
        
        # 创建情感数据
        sentiment_data = []
        for news in news_items:
            publish_time = news.get('publish_time', '')
            if publish_time:
                try:
                    # 尝试多种日期格式
                    date_formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y/%m/%d']
                    dt = None
                    for fmt in date_formats:
                        try:
                            dt = datetime.strptime(publish_time, fmt)
                            break
                        except:
                            continue
                    
                    if dt:
                        sentiment_type = news.get('sentiment_type', 'neutral')
                        score = 1 if sentiment_type == 'positive' else (-1 if sentiment_type == 'negative' else 0)
                        
                        sentiment_data.append({
                            'date': dt.date(),
                            'sentiment_score': score,
                            'title': news.get('title', ''),
                            'weight': news.get('weight', 0.5)
                        })
                except:
                    pass
        
        if not sentiment_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(sentiment_data)
        df['date'] = pd.to_datetime(df['date'])
        
        # 按日期聚合
        daily_sentiment = df.groupby('date').agg(
            sentiment_score=('sentiment_score', 'mean'),
            count=('sentiment_score', 'count')
        ).reset_index()
        
        return daily_sentiment
    
    def _calculate_average_sentiment(self, news_items: List[Dict]) -> float:
        """计算平均情感分数"""
        if not news_items:
            return 0.0
        
        scores = []
        for news in news_items:
            sentiment_type = news.get('sentiment_type', 'neutral')
            weight = news.get('weight', 0.5)
            
            if sentiment_type == 'positive':
                scores.append(1 * weight)
            elif sentiment_type == 'negative':
                scores.append(-1 * weight)
            else:
                scores.append(0 * weight)
        
        return sum(scores) / max(len(scores), 1) if scores else 0.0
    
    def _generate_final_recommendation(self, results: Dict, risk_level: str) -> Dict[str, Any]:
        """
        生成最终投资建议
        
        参数:
            results: 综合分析结果
            risk_level: 风险偏好
        
        返回:
            最终建议
        """
        # 提取关键指标
        basic_inference = results.get('basic_inference', {})
        temporal_prediction = results.get('temporal_prediction', {})
        relative_perf = results.get('relative_performance', {})
        adaptive_weights = results.get('adaptive_weights', {})
        
        # 综合信号
        signals = []
        confidence_scores = []
        
        # 1. 基础趋势信号
        trend_1d = basic_inference.get('trend_predictions', {}).get('1d', {})
        if trend_1d:
            signals.append({
                'source': 'basic_trend',
                'direction': trend_1d.get('direction', 'neutral'),
                'confidence': trend_1d.get('confidence', 0.5)
            })
            confidence_scores.append(trend_1d.get('confidence', 0.5))
        
        # 2. 时序预测信号
        if temporal_prediction.get('success', False):
            predictions = temporal_prediction.get('predictions', [])
            if predictions:
                # 比较第一天预测与最后已知价格
                stock_data = results.get('stock_data', {})
                last_price = stock_data.get('latest_price', 1)
                
                if predictions:
                    first_pred = predictions[0]
                    direction = 'up' if first_pred > last_price * 1.01 else 'down' if first_pred < last_price * 0.99 else 'stable'
                    
                    signals.append({
                        'source': 'temporal_prediction',
                        'direction': direction,
                        'confidence': temporal_prediction.get('confidence', 0.5)
                    })
                    confidence_scores.append(temporal_prediction.get('confidence', 0.5))
        
        # 3. 相对表现信号
        rel_return = relative_perf.get('relative_return', 0)
        if rel_return != 0:
            direction = 'up' if rel_return > 0.05 else 'down' if rel_return < -0.05 else 'stable'
            signals.append({
                'source': 'relative_performance',
                'direction': direction,
                'confidence': min(0.9, abs(rel_return) * 5)
            })
            confidence_scores.append(min(0.9, abs(rel_return) * 5))
        
        # 4. 情感信号
        avg_sentiment = self._calculate_average_sentiment(results.get('basic_inference', {}).get('news_items', []))
        if avg_sentiment != 0:
            direction = 'up' if avg_sentiment > 0.2 else 'down' if avg_sentiment < -0.2 else 'stable'
            signals.append({
                'source': 'sentiment',
                'direction': direction,
                'confidence': min(0.9, abs(avg_sentiment) * 2)
            })
            confidence_scores.append(min(0.9, abs(avg_sentiment) * 2))
        
        # 计算综合方向
        if not signals:
            return {
                'action': 'hold',
                'confidence': 0.3,
                'reason': '数据不足，无法做出明确判断',
                'supporting_signals': [],
                'risk_assessment': 'high'
            }
        
        # 投票决定方向
        up_count = sum(1 for s in signals if s['direction'] == 'up')
        down_count = sum(1 for s in signals if s['direction'] == 'down')
        stable_count = sum(1 for s in signals if s['direction'] == 'stable')
        
        # 加权投票（考虑置信度）
        up_weighted = sum(s['confidence'] for s in signals if s['direction'] == 'up')
        down_weighted = sum(s['confidence'] for s in signals if s['direction'] == 'down')
        total_weighted = up_weighted + down_weighted + sum(s['confidence'] for s in signals if s['direction'] == 'stable')
        
        # 确定最终操作
        if up_weighted > down_weighted * 1.2 and up_weighted > 0.5:
            action = 'buy'
        elif down_weighted > up_weighted * 1.2 and down_weighted > 0.5:
            action = 'sell'
        else:
            action = 'hold'
        
        # 计算综合置信度
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
        # 风险评估
        risk_levels = {
            'aggressive': {'threshold': 0.4},
            'moderate': {'threshold': 0.5},
            'conservative': {'threshold': 0.6}
        }
        risk_threshold = risk_levels.get(risk_level, {}).get('threshold', 0.5)
        
        risk_assessment = 'low' if avg_confidence > risk_threshold + 0.1 else 'medium' if avg_confidence > risk_threshold - 0.1 else 'high'
        
        # 生成理由
        reason_parts = []
        if action == 'buy':
            reason_parts.append('多项指标显示看涨信号')
        elif action == 'sell':
            reason_parts.append('多项指标显示看跌信号')
        else:
            reason_parts.append('信号不明确，建议观望')
        
        # 添加关键因素
        market_state = adaptive_weights.get('market_state', '')
        if market_state:
            state_labels = {
                'bull': '当前处于牛市环境',
                'bear': '当前处于熊市环境',
                'sideways': '当前处于震荡市',
                'volatile': '市场波动较大'
            }
            reason_parts.append(state_labels.get(market_state, ''))
        
        # 相对表现
        if rel_return != 0:
            prefix = '跑赢' if rel_return > 0 else '跑输'
            reason_parts.append(f'{prefix}大盘{abs(rel_return*100):.1f}%')
        
        return {
            'action': action,
            'confidence': avg_confidence,
            'reason': '; '.join(filter(None, reason_parts)),
            'supporting_signals': signals,
            'risk_assessment': risk_assessment,
            'risk_level': risk_level
        }

# ==================== 全局实例 ====================

# 创建全局推理引擎实例
inference_engine = AdvancedInferenceEngine()
enhanced_inference_engine = EnhancedInferenceEngine()

# ==================== 便捷函数 ====================

async def get_inference_and_prediction(stock_name: str, news_items: List[Dict],
                                       risk_level: str = "moderate") -> Dict:
    """
    便捷函数：获取推理和预测结果（基础版）
    
    参数:
        stock_name: 股票名称
        news_items: 新闻数据列表
        risk_level: 风险偏好
    
    返回:
        综合推理和预测结果
    """
    return await inference_engine.analyze_and_predict(stock_name, news_items, risk_level)

async def get_enhanced_analysis(stock_name: str, news_items: List[Dict],
                                risk_level: str = "moderate", days: int = 60) -> Dict:
    """
    便捷函数：获取增强版综合分析（推荐使用）
    
    参数:
        stock_name: 股票名称
        news_items: 新闻数据列表
        risk_level: 风险偏好
        days: 历史数据天数
    
    返回:
        增强版综合分析结果，包含：
        - 基础推理
        - 股票数据和技术指标
        - 因果关系发现
        - 自适应权重
        - 时序预测
        - 相对表现分析
        - 最终投资建议
    """
    return await enhanced_inference_engine.analyze_comprehensive(stock_name, news_items, risk_level, days)
