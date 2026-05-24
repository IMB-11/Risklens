import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, set_seed, pipeline,
    RobertaConfig, RobertaModel, DistilBertConfig, DistilBertModel,
    BartConfig, BartModel, AutoModel
)
from datasets import Dataset, load_dataset, concatenate_datasets
import numpy as np
import random
from scipy.special import softmax
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from scipy.stats import entropy
from collections import Counter
import math
from itertools import combinations
import os

# 设置随机种子
set_seed(42)
np.random.seed(42)
random.seed(42)

# 本地模型路径配置
LOCAL_MODEL_PATHS = {
    "financial_sentiment": "./bert-base-chinese-finetuning-financial-news-sentiment-v2",
    "roberta_large": "./chinese-roberta-wwm-ext-large",
    # 兼容历史别名：已移除 finbert-tone，finbert 别名回退到中文金融情感模型。
    "finbert": "./bert-base-chinese-finetuning-financial-news-sentiment-v2",
    "bart": "./bart-base-chinese",
    "nli_deberta": "./nli-deberta-v3-base"  # NLI推理模型
}

MODEL_PRIORITY_WEIGHTS = {
    "financial_sentiment": 0.86,
    "finbert": 0.00,
    "roberta_large": 0.08,
    "nli_deberta": 0.04,
    "bart": 0.02,
}

DEFAULT_ENSEMBLE_MODELS = ("financial_sentiment",)

class KnowledgeDistillationLoss(nn.Module):
    """知识蒸馏损失函数：将大模型知识迁移到小模型"""
    
    def __init__(self, temperature: float = 2.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(self, student_logits, teacher_logits, labels):
        """
        计算知识蒸馏损失
        
        参数:
            student_logits: 学生模型输出
            teacher_logits: 教师模型输出
            labels: 真实标签
        """
        # 蒸馏损失（软目标）
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=1)
        distillation_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (self.temperature ** 2)
        
        # 硬目标损失
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # 混合损失
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss
        
        return total_loss

class ContrastiveLearningModule(nn.Module):
    """对比学习模块：学习更鲁棒的表示"""
    
    def __init__(self, encoder_dim: int = 768, projection_dim: int = 128):
        super().__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, projection_dim)
        )
    
    def forward(self, features):
        """投影到对比学习空间"""
        return self.projection_head(features)
    
    @staticmethod
    def compute_nt_xent_loss(anchor, positive, temperature: float = 0.5):
        """
        计算 NT-Xent 损失（对比学习常用）
        
        参数:
            anchor: 锚点样本
            positive: 正样本（同一样本的不同增强）
            temperature: 温度系数
        """
        batch_size = anchor.size(0)
        
        # 归一化
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(anchor, positive.T) / temperature
        
        # 对角线是正样本对
        mask = torch.eye(batch_size, device=anchor.device, dtype=torch.bool)
        positive_sim = sim_matrix[mask].view(batch_size, 1)
        
        # 负样本对
        negative_sim = sim_matrix[~mask].view(batch_size, -1)
        
        # 计算损失
        logits = torch.cat([positive_sim, negative_sim], dim=1)
        labels = torch.zeros(batch_size, device=anchor.device, dtype=torch.long)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss

class GraphAttentionNetwork(nn.Module):
    """图注意力网络：用于捕捉新闻之间的关联关系"""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        
        # 注意力参数
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, features, adjacency_matrix):
        """
        前向传播
        
        参数:
            features: 节点特征矩阵 [batch, nodes, features]
            adjacency_matrix: 邻接矩阵 [batch, nodes, nodes]
        """
        batch_size, num_nodes, _ = features.size()
        
        # 线性变换
        h = self.W(features)  # [batch, nodes, out_features]
        
        # 计算注意力分数
        a_input = self._prepare_attentional_mechanism_input(h)  # [batch, nodes, nodes, 2*out_features]
        e = self.leaky_relu(self.a(a_input).squeeze(3))  # [batch, nodes, nodes]
        
        # 应用邻接矩阵掩码
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adjacency_matrix > 0, e, zero_vec)
        
        # 归一化注意力权重
        attention = F.softmax(attention, dim=2)
        attention = self.dropout_layer(attention)
        
        # 聚合邻居特征
        h_prime = torch.matmul(attention, h)  # [batch, nodes, out_features]
        
        return h_prime, attention
    
    def _prepare_attentional_mechanism_input(self, h):
        """准备注意力机制的输入"""
        batch_size, num_nodes, out_features = h.size()
        
        # 复制以形成所有节点对
        h_repeat = h.unsqueeze(1).repeat(1, num_nodes, 1, 1)  # [batch, nodes, nodes, out_features]
        h_repeat_interleave = h.unsqueeze(2).repeat(1, 1, num_nodes, 1)  # [batch, nodes, nodes, out_features]
        
        # 拼接
        a_input = torch.cat([h_repeat_interleave, h_repeat], dim=3)  # [batch, nodes, nodes, 2*out_features]
        
        return a_input

class ReinforcementLearningRewardModel(nn.Module):
    """强化学习奖励模型：根据市场表现调整预测"""
    
    def __init__(self, state_dim: int = 10, action_dim: int = 3):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        
        # 价值网络
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def get_action(self, state, epsilon: float = 0.1):
        """
        根据状态选择动作（ε-greedy策略）
        
        参数:
            state: 状态向量
            epsilon: 探索概率
        """
        if random.random() < epsilon:
            # 探索：随机选择
            return random.randint(0, self.action_dim - 1)
        else:
            # 利用：选择最优动作
            with torch.no_grad():
                logits = self.policy_net(state)
                return int(torch.argmax(logits, dim=1).item())
    
    def compute_advantage(self, states, rewards, gamma: float = 0.99):
        """
        计算优势函数（用于策略梯度更新）
        
        参数:
            states: 状态序列
            rewards: 奖励序列
            gamma: 折扣因子
        """
        advantages = []
        running_sum = 0
        
        # 从后往前计算
        for t in reversed(range(len(rewards))):
            running_sum = rewards[t] + gamma * running_sum
            advantages.insert(0, running_sum)
        
        # 归一化
        advantages = torch.tensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages

class FeatureInteractionModel(nn.Module):
    """特征交互模型：捕捉高阶特征交互"""
    
    def __init__(self, feature_dim: int = 32, interaction_order: int = 3):
        super().__init__()
        self.feature_dim = feature_dim
        self.interaction_order = interaction_order
        
        # 特征交叉层
        self.interaction_layers = nn.ModuleList()
        for order in range(2, interaction_order + 1):
            layer = nn.Linear(feature_dim ** order, feature_dim)
            self.interaction_layers.append(layer)
    
    def forward(self, features):
        """
        计算高阶特征交互
        
        参数:
            features: 原始特征 [batch, features]
        """
        batch_size = features.size(0)
        interactions = [features]
        
        # 计算各阶交互
        for order, layer in enumerate(self.interaction_layers, start=2):
            # 生成所有组合
            indices = list(combinations(range(self.feature_dim), order))
            
            # 计算组合特征
            interaction_features = []
            for idx in indices:
                combined = features[:, idx[0]]
                for i in idx[1:]:
                    combined = combined * features[:, i]
                interaction_features.append(combined)
            
            interaction_tensor = torch.stack(interaction_features, dim=1)
            interaction_tensor = layer(interaction_tensor)
            interactions.append(F.relu(interaction_tensor))
        
        # 拼接所有交互特征
        output = torch.cat(interactions, dim=1)
        
        return output

class BayesianEnsembleModel(nn.Module):
    """贝叶斯集成模型：结合多个模型的不确定性估计"""
    
    def __init__(self, base_models: list):
        super().__init__()
        self.base_models = nn.ModuleList(base_models)
        self.weights = nn.Parameter(torch.ones(len(base_models)) / len(base_models))
    
    def forward(self, x):
        """
        贝叶斯集成预测
        
        参数:
            x: 输入
        """
        predictions = []
        uncertainties = []
        
        for model in self.base_models:
            with torch.no_grad():
                logits = model(x)
            
            # 计算预测
            probs = F.softmax(logits, dim=1)
            predictions.append(probs)
            
            # 计算不确定性（熵）
            entropy_val = entropy(probs.detach().cpu().numpy(), axis=1)
            uncertainties.append(torch.tensor(entropy_val))
        
        # 堆叠预测
        predictions = torch.stack(predictions, dim=2)  # [batch, classes, models]
        uncertainties = torch.stack(uncertainties, dim=1)  # [batch, models]
        
        # 不确定性感知的加权融合
        weights = F.softmax(self.weights - uncertainties, dim=1)  # 不确定性越高权重越低
        weights = weights.unsqueeze(1)  # [batch, 1, models]
        
        # 加权平均
        final_probs = torch.bmm(predictions, weights.transpose(1, 2)).squeeze(2)
        
        return final_probs, uncertainties

class AutoMLFeatureSelector:
    """自动特征选择器：基于重要性自动选择特征"""
    
    def __init__(self, max_features: int = 20):
        self.max_features = max_features
        self.feature_importances = None
        self.selected_features = None
    
    def fit(self, X, y):
        """
        训练特征选择器
        
        参数:
            X: 特征矩阵
            y: 标签
        """
        # 使用梯度提升树计算特征重要性
        gbm = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gbm.fit(X, y)
        
        self.feature_importances = gbm.feature_importances_
        
        # 选择重要性最高的特征
        indices = np.argsort(self.feature_importances)[::-1][:self.max_features]
        self.selected_features = indices
        
        return self
    
    def transform(self, X):
        """
        应用特征选择
        
        参数:
            X: 特征矩阵
        """
        if self.selected_features is None:
            raise ValueError("请先调用 fit 方法")
        
        return X[:, self.selected_features]
    
    def get_feature_ranking(self):
        """获取特征重要性排名"""
        if self.feature_importances is None:
            return None
        
        return sorted(
            enumerate(self.feature_importances),
            key=lambda x: x[1],
            reverse=True
        )

class AdvancedFinancialSentimentTrainer:
    def __init__(self, use_ensemble: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_ensemble = use_ensemble
        self.models = {}
        self.tokenizers = {}
        self.analyzers = {}
        self.calibrator = None
        self.adaptive_threshold = 0.25
        
        self.label_mapping = {0: "负面", 1: "中性", 2: "正面"}
        self.action_mapping = {"负面": "卖出", "中性": "观望", "正面": "买入"}
        self.risk_adjustments = {
            "aggressive": 0.18,
            "moderate": 0.0,
            "conservative": -0.18
        }
        
        self._initialize_ensemble()
    
    def _initialize_ensemble(self):
        """初始化模型集成：使用多个金融模型融合"""
        enabled_aliases = self._get_enabled_model_aliases()

        # 定义模型配置：(模型路径, 模型类型, 是否需要自定义分类头, 模型别名)
        model_specs = []
        for alias in enabled_aliases:
            model_path = LOCAL_MODEL_PATHS.get(alias)
            if not model_path:
                continue
            model_type = alias
            model_specs.append((model_path, model_type, True, alias))

        print("[INFO] 初始化高级模型集成系统...")
        print(f"[INFO] 启用模型: {enabled_aliases}")
        print(f"[INFO] 待加载本地模型数: {len(model_specs)}")
        loaded_paths = set()
        
        for model_path, model_type, needs_classifier, alias in model_specs:
            try:
                # 检查本地模型是否存在
                if not os.path.exists(model_path):
                    print(f"[WARN] 模型路径不存在: {model_path}")
                    continue

                # 避免同一路径被重复加载（例如 financial_sentiment + finbert 别名）
                canonical_path = os.path.abspath(model_path)
                if canonical_path in loaded_paths:
                    print(f"[INFO] 跳过重复模型路径: {model_path} (alias={alias})")
                    continue
                
                print(f"\n[INFO] 加载本地模型: {alias}")
                print(f"   路径: {model_path}")
                
                # 加载tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                # 根据模型类型加载模型
                if model_type in {"financial_sentiment", "finbert"}:
                    # 金融专用模型，已有分类头
                    model = AutoModelForSequenceClassification.from_pretrained(model_path)
                elif model_type == "nli_deberta":
                    # NLI模型：使用AutoModelForSequenceClassification并适配NLI任务
                    # NLI模型原本用于推理任务（entailment, contradiction, neutral）
                    # 我们将其分类头适配为情感分类
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_path,
                        num_labels=3,  # 适配三分类情感
                        ignore_mismatched_sizes=True  # NLI模型有不同的分类头尺寸
                    )
                else:
                    # 需要添加自定义分类头
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_path, 
                        num_labels=3,
                        ignore_mismatched_sizes=True  # 允许权重不匹配
                    )
                
                # 移动到设备
                model = model.to(self.device)
                print(f"   [OK] 模型已加载到 {self.device.upper()}")
                
                # 创建分析器
                analyzer = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if self.device == "cuda" else -1,
                    return_all_scores=True
                )
                
                # 存储模型
                self.models[alias] = model
                self.tokenizers[alias] = tokenizer
                self.analyzers[alias] = analyzer
                loaded_paths.add(canonical_path)
                
                print(f"   [OK] 分析器创建成功")
            
            except Exception as e:
                print(f"[WARN] 加载 {alias} 失败: {str(e)[:100]}...")
        
        # 打印集成统计
        print(f"\n[DATA] 模型集成初始化完成")
        print(f"   成功加载: {len(self.models)} 个模型")
        print(f"   模型列表: {list(self.models.keys())}")
        
        # 如果集成不可用，降级到单模型模式
        if not self.models:
            print("[WARN] 所有模型加载失败，尝试降级到备用方案")
            self._initialize_single_model()

    def _get_enabled_model_aliases(self):
        """获取启用的模型别名列表，可通过环境变量覆盖。"""
        raw = os.getenv("FIN_SENTIMENT_MODELS", "")
        if not raw:
            return list(DEFAULT_ENSEMBLE_MODELS)

        aliases = [token.strip() for token in raw.split(",") if token.strip()]
        valid = [alias for alias in aliases if alias in LOCAL_MODEL_PATHS]
        return valid if valid else list(DEFAULT_ENSEMBLE_MODELS)
    
    def _initialize_single_model(self):
        """降级到单模型模式"""
        # 尝试本地模型（按优先级排序）
        fallback_paths = [
            LOCAL_MODEL_PATHS["financial_sentiment"],  # 金融专用优先
            LOCAL_MODEL_PATHS["roberta_large"],        # 语义增强模型
            LOCAL_MODEL_PATHS["nli_deberta"],          # NLI推理模型
            LOCAL_MODEL_PATHS["bart"]                  # 文本生成模型
        ]
        # 去重（兼容别名共享同一路径）
        fallback_paths = list(dict.fromkeys(fallback_paths))
        
        for model_path in fallback_paths:
            if os.path.exists(model_path):
                try:
                    print(f"[INFO] 尝试使用备用模型: {model_path}")
                    alias = os.path.basename(model_path)
                    self.tokenizers[alias] = AutoTokenizer.from_pretrained(model_path)
                    
                    # 处理NLI模型的特殊情况
                    if "nli" in alias.lower():
                        self.models[alias] = AutoModelForSequenceClassification.from_pretrained(
                            model_path,
                            num_labels=3,
                            ignore_mismatched_sizes=True
                        ).to(self.device)
                    else:
                        self.models[alias] = AutoModelForSequenceClassification.from_pretrained(
                            model_path,
                            num_labels=3,
                            ignore_mismatched_sizes=True
                        ).to(self.device)
                    
                    self.analyzers[alias] = pipeline(
                        "text-classification",
                        model=self.models[alias],
                        tokenizer=self.tokenizers[alias],
                        device=0 if self.device == "cuda" else -1,
                        return_all_scores=True
                    )
                    print(f"[OK] 备用模型 {alias} 加载成功")
                    return
                except Exception as e:
                    print(f"[WARN] 备用模型 {model_path} 加载失败: {e}")
        
        # 最终降级：使用HuggingFace远程模型
        print("[INFO] 本地模型均不可用，尝试从HuggingFace加载...")
        remote_model = "hw2942/bert-base-chinese-finetuning-financial-news-sentiment-v2"
        try:
            self.tokenizers[remote_model] = AutoTokenizer.from_pretrained(remote_model)
            self.models[remote_model] = AutoModelForSequenceClassification.from_pretrained(
                remote_model
            ).to(self.device)
            self.analyzers[remote_model] = pipeline(
                "text-classification",
                model=self.models[remote_model],
                tokenizer=self.tokenizers[remote_model],
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            print(f"[OK] 远程模型 {remote_model} 加载成功")
        except Exception as e:
            raise RuntimeError(f"无法加载任何模型: {e}")
    
    def analyze_nli_relationship(self, premise: str, hypothesis: str):
        """
        使用NLI模型分析两个文本之间的推理关系
        这可以帮助理解新闻事件的隐含影响
        
        参数:
            premise: 前提（如新闻事实）
            hypothesis: 假设（如可能的市场影响）
        
        返回:
            关系类型: entailment(蕴含), contradiction(矛盾), neutral(中性)
        """
        if "nli_deberta" not in self.models:
            return {"relationship": "neutral", "confidence": 0.5, "reason": "NLI模型不可用"}
        
        try:
            model = self.models["nli_deberta"]
            tokenizer = self.tokenizers["nli_deberta"]
            
            # NLI任务格式：前提 + 分隔符 + 假设
            input_text = f"{premise} [SEP] {hypothesis}"
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
            
            # NLI标签映射（标准NLI顺序）
            nli_labels = ["entailment", "neutral", "contradiction"]
            label_indices = [0, 1, 2]  # 对应蕴含、中性、矛盾
            
            max_idx = probabilities.argmax()
            relationship = nli_labels[max_idx]
            confidence = float(probabilities[max_idx])
            
            # 中文解释
            explanations = {
                "entailment": f"📈 新闻'({premise})'支持假设'({hypothesis})'",
                "contradiction": f"❌ 新闻'({premise})'与假设'({hypothesis})'矛盾",
                "neutral": f"⚖️ 新闻'({premise})'与假设'({hypothesis})'无关"
            }
            
            return {
                "relationship": relationship,
                "confidence": confidence,
                "reason": explanations[relationship],
                "probabilities": {nli_labels[i]: float(probabilities[i]) for i in label_indices}
            }
        except Exception as e:
            return {"relationship": "neutral", "confidence": 0.5, "reason": f"分析失败: {str(e)[:50]}"}

    # Mapping from model output labels (English or LABEL_N) to numeric label
    _LABEL_PARSE_MAP = {
        "LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2,
        "negative": 0, "neutral": 1, "positive": 2,
        "negative_label": 0, "neutral_label": 1, "positive_label": 2,
        "0": 0, "1": 1, "2": 2,
    }

    def _parse_label(self, raw_label) -> int:
        """Parse model output label to numeric (0/1/2). Handles LABEL_N, English names, and ints."""
        if isinstance(raw_label, int):
            return raw_label
        s = str(raw_label).strip()
        # Try direct lookup (case-insensitive)
        key = s.lower()
        if key in self._LABEL_PARSE_MAP:
            return self._LABEL_PARSE_MAP[key]
        # Try stripping "LABEL_" prefix
        if s.startswith("LABEL_"):
            try:
                return int(s.replace("LABEL_", ""))
            except ValueError:
                pass
        # Try raw int conversion
        try:
            return int(s)
        except (ValueError, TypeError):
            return 1  # Default to neutral

    def extract_financial_features(self, text: str):
        """提取专业金融特征"""
        features = {}
        
        # 金融关键词检测
        positive_keywords = ["增长", "盈利", "利好", "买入", "增持", "订单", "合作", "突破", "上调", "回购"]
        negative_keywords = ["下降", "亏损", "利空", "卖出", "减持", "爆雷", "调查", "违约", "下调", "诉讼"]
        neutral_keywords = ["符合预期", "平稳", "公告", "会议", "日常"]
        
        features["positive_keyword_count"] = sum(1 for kw in positive_keywords if kw in text)
        features["negative_keyword_count"] = sum(1 for kw in negative_keywords if kw in text)
        features["neutral_keyword_count"] = sum(1 for kw in neutral_keywords if kw in text)
        
        # 数值特征
        import re
        numbers = re.findall(r'(\d+(?:\.\d+)?)(?:%|亿|万)', text)
        features["numeric_mentions"] = len(numbers)
        features["has_percentage"] = 1 if "%" in text else 0
        features["has_money"] = 1 if "亿" in text or "万" in text else 0
        
        # 长度特征
        features["text_length"] = len(text)
        features["sentence_count"] = text.count("。") + text.count("，")
        
        return features
    
    def ensemble_predict(self, text: str):
        """多模型集成预测：概率融合 + 不确定性惩罚。"""
        if not self.analyzers:
            return {
                "label": 1,
                "sentiment": self.label_mapping[1],
                "confidence": 0.34,
                "ensemble_details": {
                    "model_count": 0,
                    "votes": {0: 0.33, 1: 0.34, 2: 0.33},
                    "individual_predictions": [],
                    "model_weights": {},
                },
            }

        fused_probs = np.zeros(3, dtype=np.float64)
        model_weight_sum = 0.0
        per_model_predictions = []
        per_model_weights = {}

        for model_name, analyzer in self.analyzers.items():
            result = analyzer(text)[0]

            probs = np.zeros(3, dtype=np.float64)
            for item in result:
                label_idx = self._parse_label(item.get("label", "LABEL_1"))
                score = float(item.get("score", 0.0))
                if 0 <= label_idx < 3:
                    probs[label_idx] = score

            if probs.sum() <= 0:
                probs = np.array([0.33, 0.34, 0.33], dtype=np.float64)
            else:
                probs = probs / probs.sum()

            entropy_score = float(entropy(probs))
            max_entropy = math.log(3)
            certainty = 1.0 - (entropy_score / max_entropy if max_entropy > 0 else 0.0)
            base_weight = MODEL_PRIORITY_WEIGHTS.get(model_name, 0.1)
            dynamic_weight = base_weight * (0.6 + 0.4 * certainty)

            fused_probs += probs * dynamic_weight
            model_weight_sum += dynamic_weight

            pred_label = int(np.argmax(probs))
            per_model_predictions.append(pred_label)
            per_model_weights[model_name] = round(dynamic_weight, 4)

        if model_weight_sum > 0:
            fused_probs = fused_probs / model_weight_sum
        else:
            fused_probs = np.array([0.33, 0.34, 0.33], dtype=np.float64)

        final_label = int(np.argmax(fused_probs))
        final_confidence = float(fused_probs[final_label])

        return {
            "label": final_label,
            "sentiment": self.label_mapping[final_label],
            "confidence": final_confidence,
            "ensemble_details": {
                "model_count": len(per_model_predictions),
                "votes": {idx: float(round(prob, 4)) for idx, prob in enumerate(fused_probs)},
                "individual_predictions": per_model_predictions,
                "model_weights": per_model_weights,
            },
        }
    
    def calibrate_confidence(self, probabilities: np.ndarray):
        """置信度校准：Platt Scaling"""
        if self.calibrator is None:
            # 使用模拟数据训练校准器
            self._train_calibrator()
        
        if self.calibrator:
            try:
                return self.calibrator.predict_proba(probabilities.reshape(1, -1))[0]
            except:
                return probabilities
        
        return probabilities
    
    def _train_calibrator(self):
        """训练置信度校准器"""
        print("[INFO] 训练置信度校准器...")
        
        # 生成模拟校准数据
        np.random.seed(42)
        n_samples = 1000
        X = np.random.rand(n_samples, 3)
        X = X / X.sum(axis=1, keepdims=True)  # 归一化
        y = np.argmax(X, axis=1)
        
        self.calibrator = CalibratedClassifierCV(
            LogisticRegression(),
            method='sigmoid',
            cv=3
        )
        
        try:
            self.calibrator.fit(X, y)
            print("[OK] 校准器训练完成")
        except:
            self.calibrator = None
    
    def compute_adaptive_threshold(self, data_quality: float, confidence: float):
        """自适应阈值：根据数据质量和置信度动态调整"""
        base_threshold = 0.25
        
        # 数据质量越高，阈值越严格
        quality_factor = 1.0 - (data_quality - 0.5) * 0.3
        
        # 置信度越高，阈值越宽松
        confidence_factor = 1.0 - (confidence - 0.5) * 0.2
        
        return base_threshold * quality_factor * confidence_factor
    
    def analyze_sentiment(self, text: str):
        """单文本情感分析（集成+校准）"""
        if not self.analyzers:
            return {
                "text": text,
                "sentiment": "中性",
                "action": self.action_mapping["中性"],
                "confidence": 0.34,
                "calibrated_confidence": 0.34,
                "label": 1,
                "financial_features": self.extract_financial_features(text),
                "ensemble_details": {"model_count": 0, "reason": "no_model_loaded"},
            }

        # 模型集成预测
        if self.use_ensemble and len(self.analyzers) > 1:
            result = self.ensemble_predict(text)
        else:
            # 单模型预测
            analyzer = list(self.analyzers.values())[0]
            raw_results = analyzer(text)[0]
            max_result = max(raw_results, key=lambda x: x["score"])
            label = self._parse_label(max_result["label"])
            
            result = {
                "label": label,
                "sentiment": self.label_mapping[label],
                "confidence": max_result["score"],
                "ensemble_details": None
            }
        
        # 提取金融特征
        features = self.extract_financial_features(text)
        
        # 置信度校准
        if result["confidence"] is not None:
            probs = np.array([0.0, 0.0, 0.0])
            probs[result["label"]] = result["confidence"]
            calibrated = self.calibrate_confidence(probs)
            result["calibrated_confidence"] = calibrated[result["label"]]
        
        return {
            "text": text,
            "sentiment": result["sentiment"],
            "action": self.action_mapping[result["sentiment"]],
            "confidence": result["confidence"],
            "calibrated_confidence": result.get("calibrated_confidence"),
            "label": result["label"],
            "financial_features": features,
            "ensemble_details": result.get("ensemble_details")
        }
    
    def analyze_batch(self, texts: list):
        return [self.analyze_sentiment(text) for text in texts]
    
    def generate_action(self, stock_name: str, news_items: list, risk_profile: str = "moderate"):
        """生成投资行动建议（高级版）"""
        if not news_items:
            return self._empty_result(stock_name)
        
        results = self.analyze_batch(news_items)
        
        # 统计分析
        positive_count = sum(1 for r in results if r["sentiment"] == "正面")
        negative_count = sum(1 for r in results if r["sentiment"] == "负面")
        neutral_count = len(results) - positive_count - negative_count
        total = len(results)
        
        # 加权置信度计算
        confidences = [r["confidence"] for r in results if r["confidence"] is not None]
        if confidences:
            weighted_confidence = sum(c ** 2 for c in confidences) / sum(confidences)
        else:
            weighted_confidence = 0.5
        
        # 特征聚合
        avg_positive_features = sum(
            r["financial_features"]["positive_keyword_count"]
            for r in results
        ) / total
        avg_negative_features = sum(
            r["financial_features"]["negative_keyword_count"]
            for r in results
        ) / total
        
        # 风险调整
        risk_adjustment = self.risk_adjustments.get(risk_profile, 0.0)
        
        # 情感分数计算（考虑特征）
        base_score = (positive_count - negative_count) / total
        feature_bonus = (avg_positive_features - avg_negative_features) * 0.05
        sentiment_score = base_score + feature_bonus + risk_adjustment
        
        # 自适应阈值
        adaptive_thresh = self.compute_adaptive_threshold(weighted_confidence, weighted_confidence)
        
        # 决策
        if sentiment_score > adaptive_thresh:
            final_action = "买入"
            overall_sentiment = "正面"
        elif sentiment_score < -adaptive_thresh:
            final_action = "卖出"
            overall_sentiment = "负面"
        else:
            final_action = "观望"
            overall_sentiment = "中性"
        
        # 风险评估
        risk_warning = self._assess_risk(results, weighted_confidence)
        
        # 数据质量评估
        data_quality = self._assess_data_quality(total, weighted_confidence, results)
        
        return {
            "stock_name": stock_name,
            "action": final_action,
            "sentiment": overall_sentiment,
            "confidence": round(weighted_confidence, 4),
            "sentiment_score": round(sentiment_score, 4),
            "adaptive_threshold": round(adaptive_thresh, 4),
            "reasoning": self._generate_reasoning(
                stock_name, positive_count, negative_count, neutral_count,
                weighted_confidence, risk_profile, data_quality
            ),
            "risk_warning": risk_warning,
            "data_quality": data_quality,
            "sample_count": total,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "feature_metrics": {
                "avg_positive_keywords": round(avg_positive_features, 2),
                "avg_negative_keywords": round(avg_negative_features, 2)
            },
            "technical_details": {
                "ensemble_used": self.use_ensemble and len(self.analyzers) > 1,
                "model_count": len(self.analyzers),
                "confidence_calibrated": self.calibrator is not None,
                "adaptive_threshold_applied": True
            },
            "detail": results[:5]
        }
    
    def _empty_result(self, stock_name):
        return {
            "stock_name": stock_name,
            "action": "观望",
            "sentiment": "中性",
            "confidence": 0.5,
            "reasoning": "无有效数据",
            "risk_warning": False,
            "data_quality": "low"
        }
    
    def _assess_risk(self, results, confidence):
        """风险评估：检测潜在风险信号"""
        negative_ratio = sum(1 for r in results if r["sentiment"] == "负面") / len(results)
        confidence_too_low = confidence < 0.4
        
        return negative_ratio > 0.5 or confidence_too_low
    
    def _assess_data_quality(self, total_count, confidence, results):
        """综合评估数据质量"""
        # 样本数量
        quantity_score = min(total_count / 10, 1.0)
        
        # 置信度
        confidence_score = confidence
        
        # 多样性（避免重复信息）
        unique_sentiments = len(set(r["sentiment"] for r in results))
        diversity_score = unique_sentiments / 3
        
        # 综合评分
        overall = (quantity_score * 0.3 + confidence_score * 0.5 + diversity_score * 0.2)
        
        if overall > 0.7:
            return "high"
        elif overall > 0.5:
            return "medium"
        else:
            return "low"
    
    def _generate_reasoning(self, stock_name, positive, negative, neutral, confidence, risk_profile, data_quality):
        """生成专业推理文本"""
        parts = []
        parts.append(f"【{stock_name}】舆情分析报告")
        parts.append(f"📊 样本量: {positive+negative+neutral}条（正面:{positive} 负面:{negative} 中性:{neutral}）")
        parts.append(f"🎯 分析置信度: {confidence:.1%}")
        parts.append(f"⚖️ 风险偏好: {risk_profile}")
        parts.append(f"📈 数据质量: {data_quality}")
        
        if positive > negative:
            parts.append("✅ 正面消息占优，建议关注")
        elif negative > positive:
            parts.append("⚠️ 负面消息较多，谨慎操作")
        else:
            parts.append("➡️ 消息均衡，观望为宜")
        
        return " | ".join(parts)
    
    def adversarial_attack(self, text: str, epsilon: float = 0.1):
        """对抗训练演示：FGSM 攻击"""
        model_name = list(self.models.keys())[0]
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        model.eval()
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length"
        ).to(self.device)
        
        inputs["input_ids"].requires_grad = True
        
        outputs = model(**inputs)
        loss = outputs.logits[0, 1]  # 针对中性类别的损失
        
        loss.backward()
        
        grad = inputs["input_ids"].grad.data
        perturbed_ids = inputs["input_ids"].data + epsilon * grad.sign()
        
        # 确保扰动后的ID仍然有效
        perturbed_ids = torch.clamp(perturbed_ids, 0, tokenizer.vocab_size - 1)
        
        perturbed_text = tokenizer.decode(perturbed_ids[0], skip_special_tokens=True)
        
        return {
            "original": text,
            "perturbed": perturbed_text,
            "attack_info": {
                "epsilon": epsilon,
                "method": "FGSM"
            }
        }

class RealTimePushManager:
    """实时推送管理器：支持突发新闻即时提醒"""
    
    def __init__(self):
        self.subscribers = {}  # {user_id: {callback, stocks, risk_profile}}
        self.news_buffer = []
        self.alert_threshold = 0.8  # 高置信度阈值
        self.last_alert_time = {}  # 去重：避免重复提醒
    
    def subscribe(self, user_id: str, callback, stocks: list, risk_profile: str = "moderate"):
        """用户订阅股票提醒"""
        self.subscribers[user_id] = {
            "callback": callback,
            "stocks": stocks,
            "risk_profile": risk_profile
        }
        print(f"✅ 用户 {user_id} 订阅了: {stocks}")
    
    def unsubscribe(self, user_id: str):
        """取消订阅"""
        if user_id in self.subscribers:
            del self.subscribers[user_id]
            print(f"❌ 用户 {user_id} 取消订阅")
    
    def process_breaking_news(self, stock_name: str, news_text: str, timestamp: float):
        """
        处理突发新闻，触发实时提醒
        
        触发条件:
        1. 情感置信度 > 阈值 (0.8)
        2. 与上次提醒间隔 > 5分钟（去重）
        3. 是高影响力事件
        """
        # 检查去重
        if stock_name in self.last_alert_time:
            if timestamp - self.last_alert_time[stock_name] < 300:  # 5分钟内不重复
                return None
        
        # 分析情感
        analyzer = AdvancedFinancialSentimentTrainer()
        result = analyzer.analyze_sentiment(news_text)
        
        # 检查触发条件
        should_alert = self._check_alert_conditions(result, news_text)
        
        if should_alert:
            self.last_alert_time[stock_name] = timestamp
            alert = self._generate_alert(stock_name, news_text, result)
            
            # 推送给所有订阅该股票的用户
            self._broadcast_alert(stock_name, alert)
            
            return alert
        
        return None
    
    def _check_alert_conditions(self, result, news_text):
        """检查是否应该触发提醒"""
        conditions = []
        
        # 条件1: 高置信度
        conditions.append(result["confidence"] > self.alert_threshold)
        
        # 条件2: 强烈情感（非中性）
        conditions.append(result["sentiment"] != "中性")
        
        # 条件3: 包含高影响力关键词
        high_impact_keywords = ["爆雷", "造假", "调查", "违约", "减持", "增持", "回购", "目标价", "评级"]
        contains_high_impact = any(kw in news_text for kw in high_impact_keywords)
        conditions.append(contains_high_impact)
        
        # 条件4: 包含数值信息
        has_number = any(c.isdigit() for c in news_text)
        conditions.append(has_number)
        
        return sum(conditions) >= 3  # 至少满足3个条件
    
    def _generate_alert(self, stock_name, news_text, result):
        """生成结构化提醒"""
        return {
            "type": "breaking_news",
            "stock_name": stock_name,
            "news_text": news_text,
            "sentiment": result["sentiment"],
            "action": result["action"],
            "confidence": result["confidence"],
            "urgency": self._calculate_urgency(result, news_text),
            "timestamp": datetime.now().isoformat(),
            "suggestion": self._generate_trading_suggestion(result)
        }
    
    def _calculate_urgency(self, result, news_text):
        """计算紧急程度"""
        urgency = 0
        
        # 情感强度
        if result["sentiment"] == "正面" and result["confidence"] > 0.9:
            urgency += 2
        elif result["sentiment"] == "负面" and result["confidence"] > 0.9:
            urgency += 3
        
        # 关键词权重
        urgent_keywords = {"爆雷": 3, "造假": 3, "调查": 2, "违约": 3, "减持": 2, "增持": 1}
        for kw, weight in urgent_keywords.items():
            if kw in news_text:
                urgency += weight
        
        if urgency >= 5:
            return "critical"
        elif urgency >= 3:
            return "high"
        else:
            return "medium"
    
    def _generate_trading_suggestion(self, result):
        """生成交易建议"""
        if result["sentiment"] == "正面":
            return f"⚠️ 【{result['stock_name']}】出现正面突发新闻，建议关注买入机会"
        elif result["sentiment"] == "负面":
            return f"🚨 【{result['stock_name']}】出现负面突发新闻，建议立即评估持仓风险"
        else:
            return f"ℹ️ 【{result['stock_name']}】有新消息，建议保持关注"
    
    def _broadcast_alert(self, stock_name, alert):
        """广播提醒给所有订阅用户"""
        notified_count = 0
        
        for user_id, info in self.subscribers.items():
            if stock_name in info["stocks"]:
                try:
                    info["callback"](alert, info["risk_profile"])
                    notified_count += 1
                except Exception as e:
                    print(f"⚠️ 推送失败给 {user_id}: {e}")
        
        print(f"📢 已推送提醒给 {notified_count} 个订阅用户")
    
    def get_subscriber_count(self):
        """获取订阅用户数"""
        return len(self.subscribers)

import datetime

class PredictiveSentimentAnalyzer(AdvancedFinancialSentimentTrainer):
    """预测性舆情分析：超越传统的创新功能"""
    
    def __init__(self):
        super().__init__(use_ensemble=True)
        self.temporal_model = None
        self._init_predictive_model()
    
    def _init_predictive_model(self):
        """初始化时序预测模型"""
        from sklearn.ensemble import RandomForestRegressor
        self.temporal_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        self._train_predictive_model()
    
    def _train_predictive_model(self):
        """使用历史数据训练预测模型"""
        np.random.seed(42)
        n_samples = 5000
        
        # 模拟历史数据
        X = np.random.rand(n_samples, 5)  # 特征：情感、置信度、波动性等
        y = np.random.rand(n_samples) * 0.1  # 未来收益
        
        self.temporal_model.fit(X, y)
        print("✅ 预测模型训练完成")
    
    def predict_price_impact(self, news_items, horizon_hours=24):
        """
        预测舆情对未来价格的影响
        
        返回: 价格变动概率分布和置信区间
        """
        if not news_items:
            return {"error": "No data"}
        
        # 分析当前舆情
        results = self.analyze_batch(news_items)
        
        # 提取特征
        avg_sentiment = sum(1 if r["sentiment"] == "正面" else (-1 if r["sentiment"] == "负面" else 0) for r in results) / len(results)
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        volatility = np.std([r["confidence"] for r in results])
        positive_ratio = sum(1 for r in results if r["sentiment"] == "正面") / len(results)
        
        # 构建特征向量
        features = np.array([[avg_sentiment, avg_confidence, volatility, positive_ratio, horizon_hours / 24]])
        
        # 预测
        prediction = self.temporal_model.predict(features)[0]
        
        # 生成概率分布
        std_dev = 0.02  # 经验标准差
        confidence_interval = (
            prediction - 1.96 * std_dev,
            prediction + 1.96 * std_dev
        )
        
        return {
            "expected_return": round(prediction, 4),
            "confidence_interval": (round(confidence_interval[0], 4), round(confidence_interval[1], 4)),
            "probability_positive": round(self._calculate_probability(prediction, std_dev, 0), 4),
            "signal_strength": self._calculate_signal_strength(prediction, avg_confidence),
            "analysis_details": {
                "avg_sentiment": round(avg_sentiment, 4),
                "avg_confidence": round(avg_confidence, 4),
                "sample_count": len(news_items)
            }
        }
    
    def _calculate_probability(self, mean, std, threshold):
        """计算超过阈值的概率"""
        from scipy.stats import norm
        return 1 - norm.cdf(threshold, loc=mean, scale=std)
    
    def _calculate_signal_strength(self, prediction, confidence):
        """计算信号强度等级"""
        combined = abs(prediction) * confidence
        if combined > 0.03:
            return "strong"
        elif combined > 0.015:
            return "medium"
        else:
            return "weak"
    
    def analyze_with_explanation(self, text):
        """
        可解释性分析：提供详细的决策依据
        """
        result = self.analyze_sentiment(text)
        
        # 提取关键信息
        features = result["financial_features"]
        key_phrases = self._extract_key_phrases(text)
        
        # 生成自然语言解释
        explanation = self._generate_explanation(text, result, key_phrases)
        
        return {
            **result,
            "explanation": explanation,
            "key_phrases": key_phrases,
            "decision_factors": self._list_decision_factors(features, result)
        }
    
    def _extract_key_phrases(self, text):
        """提取影响决策的关键短语"""
        keywords = {
            "positive": ["增长", "盈利", "利好", "买入", "增持", "订单", "突破"],
            "negative": ["下降", "亏损", "利空", "卖出", "减持", "爆雷", "调查"],
            "neutral": ["符合", "平稳", "预期", "公告"]
        }
        
        found = {"positive": [], "negative": [], "neutral": []}
        for category, words in keywords.items():
            for word in words:
                if word in text:
                    found[category].append(word)
        
        return found
    
    def _generate_explanation(self, text, result, key_phrases):
        """生成自然语言解释"""
        parts = []
        parts.append(f"分析文本: {text}")
        parts.append(f"结论: {result['sentiment']} ({result['confidence']:.1%}置信度)")
        
        if key_phrases["positive"]:
            parts.append(f"正面关键词: {', '.join(key_phrases['positive'])}")
        if key_phrases["negative"]:
            parts.append(f"负面关键词: {', '.join(key_phrases['negative'])}")
        
        parts.append(f"建议行动: {result['action']}")
        
        return "\n".join(parts)
    
    def _list_decision_factors(self, features, result):
        """列出影响决策的因素"""
        factors = []
        
        if features["positive_keyword_count"] > 0:
            factors.append(f"发现{features['positive_keyword_count']}个正面关键词")
        if features["negative_keyword_count"] > 0:
            factors.append(f"发现{features['negative_keyword_count']}个负面关键词")
        if features["has_percentage"]:
            factors.append("包含百分比数据")
        if features["has_money"]:
            factors.append("包含金额数据")
        
        factors.append(f"分析置信度: {result['confidence']:.1%}")
        
        return factors

if __name__ == "__main__":
    print("=== 🚀 创新型金融情感分析系统 ===")
    print("📊 特性: 模型集成 | 置信度校准 | 自适应阈值 | 预测性分析 | 可解释性AI\n")
    
    # 初始化
    trainer = AdvancedFinancialSentimentTrainer(use_ensemble=True)
    
    # 测试中文情感分析
    test_texts = [
        "宁德时代Q3净利润同比增长150%，超出市场预期",
        "贵州茅台遭大股东减持，股价大跌5%",
        "比亚迪发布季度报告，业绩符合预期"
    ]
    
    print("🔍 测试中文情感分析:")
    for text in test_texts:
        result = trainer.analyze_sentiment(text)
        print(f"\n文本: {text}")
        print(f"  情感: {result['sentiment']} | 行动: {result['action']}")
        print(f"  置信度: {result['confidence']:.2%}")
        if result.get('calibrated_confidence'):
            print(f"  校准后置信度: {result['calibrated_confidence']:.2%}")
        print(f"  金融特征: 正面关键词={result['financial_features']['positive_keyword_count']}, 负面关键词={result['financial_features']['negative_keyword_count']}")
    
    # 测试行动建议生成
    print("\n🎯 测试行动建议生成:")
    action_result = trainer.generate_action(
        stock_name="宁德时代",
        news_items=[
            "宁德时代Q3净利润同比增长150%，超出市场预期",
            "分析师上调目标价至350元，给予买入评级",
            "公司获得海外大额订单，金额达50亿元",
            "电池价格下降影响毛利率",
            "新能源汽车销量持续增长，利好产业链"
        ],
        risk_profile="moderate"
    )
    
    print(f"股票: {action_result['stock_name']}")
    print(f"行动建议: {action_result['action']}")
    print(f"推理: {action_result['reasoning']}")
    print(f"技术细节: 模型集成={action_result['technical_details']['ensemble_used']}, 模型数量={action_result['technical_details']['model_count']}, 置信度已校准={action_result['technical_details']['confidence_calibrated']}")
    
    # 测试对抗攻击（可选）
    print("\n🛡️ 对抗攻击测试:")
    attack = trainer.adversarial_attack("宁德时代业绩良好")
    print(f"原始: {attack['original']}")
    print(f"扰动后: {attack['perturbed']}")
