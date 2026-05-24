[English](README.md) | 中文

# RiskLens — 智能金融风控系统

> 让每一个投资决策都有数据撑腰。

RiskLens 是一个面向 A 股市场的 AI 风控助手。它像一个**永不休息的分析师**，24 小时盯着全网财经资讯、另类数据和市场行情，实时帮你判断"这只股票现在安不安全"。

**完全免费，本地运行，不用花一分钱。**
所有 AI 模型全部本地部署，不需要调用任何付费 API，不需要订阅任何服务。你自己的电脑就是你的专属风控终端。

---

## 核心功能

### 全网舆情自动洞察

自动抓取 15+ 国内财经信息源和龙虎榜、融资融券、搜索热度等另类数据，AI 自动过滤噪音、识别关键信号。你不需要翻遍全网，系统帮你把"值得看的"挑出来。

### 智能风险评分引擎

六大维度（情绪、趋势、波动、事件、不确定性、市场状态）联合打分，输出一个直观的 0-100 风险分。不用盯盘，一个数字告诉你该紧张还是安心。

### 自适应市场感知

系统能自动判断当前处于牛市、熊市、震荡还是高波动，并动态调整分析策略。牛市多听市场情绪，熊市紧盯技术信号——像一个有经验的交易员一样灵活切换视角。

### AI 风控叙述生成

看不懂量化指标没关系。系统用本地部署的大语言模型自动生成自然语言风控报告：

> "当前负面舆情集中，建议降低仓位。"

把数据翻译成人话——**全程本地运行，你的数据不出本机，零成本无限用。**

### 压力测试与情景模拟

想知道"如果大盘跌 20% 会怎样"？一键模拟 2015 股灾、2020 疫情等历史极端场景，提前知道你的仓位能不能扛得住。

### 异常交易预警

自动识别对敲、异常换手、内幕交易等可疑行为，合规风险早发现早处理。

---

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/IMB-11/Risklens.git
cd Risklens
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载模型

所有模型均从 HuggingFace 免费下载，项目启动时会自动加载：

| 模型 | 用途 | HuggingFace 地址 |
|------|------|-----------------|
| Qwen1.5-1.8B-Chat | 风控叙述生成 + 跨模块审计 | `Qwen/Qwen1.5-1.8B-Chat` |
| FinanceLM | 趋势预测 + 投资建议 | `financeLM/stock-movement-prediction` |
| BERT-Chinese-Sentiment | 中文金融情感分析 | `bert-base-chinese-finetuning-financial-news-sentiment-v2` |
| mDeBERTa-v3 | 多语言语义风险评分 | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` |

### 4. 启动服务

```bash
python -m backend.api.main
```

服务启动后访问 `http://localhost:8000` 查看 API 文档。

---

## API 接口一览

| 接口 | 说明 |
|------|------|
| `POST /api/risk/assess` | 核心风控评估（输入股票名，输出风险分 + 建议） |
| `POST /api/action` | 风控动作建议（兼容旧版接口） |
| `POST /api/multi-model/predict` | 多模型综合预测（7 天趋势 + 买卖建议） |
| `POST /api/risk/portfolio` | 组合风控（多标的、协方差、暴露分析） |
| `POST /api/risk/stress-test` | 压力测试（历史情景 + 自定义冲击） |
| `POST /api/risk/anomaly-detect` | 异常交易检测 |
| `POST /api/risk/alternative-data` | 另类数据查询（龙虎榜/融资融券/搜索热度） |
| `POST /api/news` | 实时新闻抓取 |
| `WS /ws/realtime` | WebSocket 实时新闻推送 |

---

## 技术栈

- **后端：** Python + FastAPI
- **前端：** Vue 3 + Vite
- **AI 模型：** PyTorch + Hugging Face Transformers（全部本地运行）
- **数据源：** AKShare + SerpApi + Tavily + 多站爬虫
- **实时通信：** WebSocket

---

## 项目结构

```
RiskLens/
├── backend/
│   ├── api/            # FastAPI 路由与接口
│   ├── crawler/        # 多源新闻爬虫（15+ 数据源）
│   ├── analysis/       # 因果发现、自适应权重、信号关联
│   ├── engine/         # 多模型协调、推理引擎、输出增强
│   ├── risk/           # 风控核心（VaR/CVaR、压力测试、异常检测）
│   ├── data/           # 行情数据、股票解析、风险数据供给
│   └── alt_data/       # 国内另类数据引擎（13 个数据源）
├── frontend/           # Vue 3 前端
├── config/             # 全局配置
├── tests/              # 测试用例
└── risk_sklearn/       # 轻量级融合模型
```

---

## License

MIT License
