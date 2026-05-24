English | [中文](README_zh.md)

# RiskLens — AI-Powered Financial Risk Control System

> Every investment decision, backed by data.

RiskLens is an AI risk control assistant built for China's A-share market. Think of it as an **analyst that never sleeps** — monitoring financial news, alternative data, and market movements 24/7, so you always know whether a stock is safe or not.

**100% free. Runs locally. Zero cost.**
All AI models run on your own machine. No paid APIs, no subscriptions. Your computer becomes your personal risk terminal.

---

## Core Features

### Automated Sentiment Intelligence

Aggregates 15+ domestic financial news sources plus alternative data like Dragon & Tiger lists, margin trading, and search trends. AI automatically filters noise and highlights what matters — so you don't have to browse the entire internet.

### Smart Risk Scoring Engine

Six dimensions — sentiment, trend, volatility, events, uncertainty, and market regime — combine into a single 0–100 risk score. No need to stare at charts; one number tells you whether to worry or relax.

### Adaptive Market Awareness

The system detects whether the market is in a bull, bear, sideways, or high-volatility regime and dynamically adjusts its analysis strategy. Bull markets lean into sentiment; bear markets focus on technical signals — just like an experienced trader.

### AI-Generated Risk Narratives

Don't understand quantitative indicators? No problem. A locally deployed language model generates plain-language risk reports:

> "Negative sentiment is concentrated. Consider reducing your position."

Data translated into human language — **fully local, your data never leaves your machine, unlimited use at zero cost.**

### Stress Testing & Scenario Simulation

Curious about "what if the market drops 20%"? One click simulates extreme historical scenarios like the 2015 crash or the 2020 pandemic, so you know ahead of time whether your portfolio can survive.

### Anomaly Trading Detection

Automatically identifies wash trading, abnormal turnover, insider trading, and other suspicious activities. Compliance risks detected early.

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/IMB-11/Risklens.git
cd Risklens
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download models

All models are free to download from HuggingFace and auto-load on startup:

| Model | Purpose | HuggingFace |
|-------|---------|-------------|
| Qwen1.5-1.8B-Chat | Risk narrative generation + cross-module audit | `Qwen/Qwen1.5-1.8B-Chat` |
| FinanceLM | Trend prediction + investment advice | `financeLM/stock-movement-prediction` |
| BERT-Chinese-Sentiment | Chinese financial sentiment analysis | `bert-base-chinese-finetuning-financial-news-sentiment-v2` |
| mDeBERTa-v3 | Multilingual semantic risk scoring | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` |

### 4. Start the server

```bash
python -m backend.api.main
```

Visit `http://localhost:8000` for API documentation.

---

## API Reference

| Endpoint | Description |
|----------|-------------|
| `POST /api/risk/assess` | Core risk assessment (input stock name, output risk score + advice) |
| `POST /api/action` | Risk action recommendations (legacy-compatible) |
| `POST /api/multi-model/predict` | Multi-model prediction (7-day trend + buy/sell advice) |
| `POST /api/risk/portfolio` | Portfolio risk control (multi-asset, covariance, exposure) |
| `POST /api/risk/stress-test` | Stress testing (historical scenarios + custom shocks) |
| `POST /api/risk/anomaly-detect` | Anomaly trading detection |
| `POST /api/risk/alternative-data` | Alternative data queries (Dragon & Tiger / margin / search trends) |
| `POST /api/news` | Real-time news fetching |
| `WS /ws/realtime` | WebSocket real-time news push |

---

## Tech Stack

- **Backend:** Python + FastAPI
- **Frontend:** Vue 3 + Vite
- **AI Models:** PyTorch + Hugging Face Transformers (all local)
- **Data Sources:** AKShare + SerpApi + Tavily + multi-site crawlers
- **Real-time:** WebSocket

---

## Project Structure

```
RiskLens/
├── backend/
│   ├── api/            # FastAPI routes & endpoints
│   ├── crawler/        # Multi-source news crawlers (15+ sources)
│   ├── analysis/       # Causal discovery, adaptive weighting, signal correlation
│   ├── engine/         # Multi-model coordination, inference engine, output enhancement
│   ├── risk/           # Risk control core (VaR/CVaR, stress testing, anomaly detection)
│   ├── data/           # Market data, stock resolution, risk data provider
│   └── alt_data/       # Domestic alternative data engine (13 sources)
├── frontend/           # Vue 3 frontend
├── config/             # Global configuration
├── tests/              # Test cases
└── risk_sklearn/       # Lightweight fusion model
```

---

## License

MIT License
