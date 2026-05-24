# AI Risk Terminal

AI Risk Terminal 是一个黑客松 Demo 项目：FastAPI 后端提供金融舆情、风险评分、多模型预测、组合风控和 AI 风控报告能力；Vue 3 + Vite 前端提供深色金融终端工作台。

本项目仅用于黑客松演示和工程学习，不构成任何投资建议。

## 技术栈

- Backend: FastAPI, Python, Uvicorn, Pydantic, pandas/numpy/scikit-learn/torch/transformers 等
- Frontend: Vue 3, Vite, vue-router, axios, echarts, lucide-vue-next
- External Search: SerpAPI, Tavily

## 目录结构

```text
backend/api/main.py        FastAPI 入口
frontend/                  Vue 3 + Vite 前端
config/ model/ dataset/    模型与数据相关目录
risk_sklearn/              风控小模型相关目录
requirements.txt           后端依赖
start_backend.bat          后端启动脚本
start_frontend.bat         前端启动脚本
start_all.bat              Windows 一键启动
```

## 环境要求

- Python 3.10+ 推荐
- Node.js 18+ 推荐
- npm 9+ 推荐

## 外部检索服务配置

SerpAPI 注册网址：
https://serpapi.com/

Tavily 注册网址：
https://tavily.com/

复制 `.env.example` 为 `.env`，按需填写：

```env
SERPAPI_API_KEY=你的 SerpAPI Key
SERP_API_KEY=你的 SerpAPI Key
TAVILY_API_KEY=你的 Tavily API Key
RISK_ENABLE_EXTERNAL_CRAWL=false

# LLM 报告生成：auto | local_qwen | qwen_api | deepseek | rules
LLM_PROVIDER=auto
QWEN_API_KEY=你的 Qwen/DashScope Key
QWEN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_API_MODEL=qwen-plus
DEEPSEEK_API_KEY=你的 DeepSeek Key
DEEPSEEK_API_BASE=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

如果不配置 Key：

- 后端仍可启动。
- 前端仍可进入演示模式。
- 新闻、搜索、实时外部信息可能使用 fallback 数据。
- 路演 Demo 不会因为缺少 Key 白屏或崩溃。

`RISK_ENABLE_EXTERNAL_CRAWL=false` 是推荐的路演默认值：后端仍提供真实健康检查、风控、组合和报告接口，但外部新闻抓取会降级，避免免费源或本机爬虫依赖导致现场不稳定。需要真实联网抓取时再改为 `true`。

前端的“系统状态”页预留了 SerpAPI、Tavily、Qwen API、DeepSeek API 和 LLM Provider 配置。黑客松本地演示阶段会保存到浏览器 localStorage，并通过请求 Header 发送给后端；生产环境不建议这样保存敏感 Key。若选择“本地 Qwen”，请在后端 `.env` 配置 `QWEN_MODEL_PATH` 并保证模型文件存在；若选择 Qwen API 或 DeepSeek API，则后端会优先调用对应 OpenAI-compatible `/chat/completions` 接口生成报告。

## 后端启动

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --reload
```

健康检查：

```text
http://127.0.0.1:8000/api/health
```

## 前端启动

```bash
cd frontend
npm install
npm run dev
```

打开：

```text
http://127.0.0.1:5173
```

Vite 已配置代理：

- `/api` -> `http://127.0.0.1:8000`
- `/ws` -> `ws://127.0.0.1:8000`

## 一键启动

Windows 双击或运行：

```bat
start_all.bat
```

也可以分别运行：

```bat
start_backend.bat
start_frontend.bat
```

## Demo 演示流程

1. 打开前端 `http://127.0.0.1:5173`。
2. 默认进入 `NVDA` 单股票风控分析页。
3. 点击 `Analyze`。
4. 查看风险评分、AI 投资建议、未来 7 天预测、VaR/CVaR、舆情、新闻与告警。
5. 切换到“组合风控”查看默认组合 `NVDA/AAPL/MSFT/TSLA`。
6. 切换到“AI 风控报告”查看 LLM/Qwen/FinanceLM 风险报告。
7. 切换到“系统状态”填写自己的 SerpAPI / Tavily / Qwen API / DeepSeek API Key，并选择 LLM Provider。

## 真实 API 与演示模式

前端默认优先请求真实 FastAPI 接口。后端没启动、接口超时、返回为空或用户打开右上角“演示模式”时，前端自动使用稳定演示数据，页面不会白屏。

UI 中不会显示 mock/fake/假数据等字样，只会显示“演示模式”“数据已更新”“外部检索未配置，可继续使用演示模式”等产品状态。

## AI 风控报告模块

报告页优先读取：

```text
GET /api/risk/report/{stock_name}?risk_level=moderate&limit=20
```

后端返回 `narrative` 时显示 LLM 正文；`LLM_PROVIDER=auto` 会优先使用已配置的 Qwen API，其次 DeepSeek API，再尝试本地 Qwen，最后降级到规则增强报告。若 LLM 不可用或正文为空，前端会根据风险评分、VaR/CVaR、模型融合、风险驱动和风控动作自动拼接结构化报告。

报告页支持：

- 复制报告
- 下载 Markdown
- 下载 JSON

## 常见问题

端口被占用：修改 `frontend/vite.config.js` 的 Vite 端口，或停止占用 `8000/5173` 的进程。

`npm install` 失败：确认 Node.js 18+ 和 npm 可用，必要时清理 npm 缓存后重试。

`pip install` 失败：建议使用 Python 3.10+，并确认网络可访问 PyPI。部分 ML 依赖较大，首次安装可能较慢。

大模型文件缺失：系统应降级到 fallback / demo 能力，后端健康状态可能显示 degraded，但前端仍可演示。

SerpAPI / Tavily Key 缺失：外部检索能力会降级，前端仍可演示。

CORS / proxy 问题：开发环境请从 `http://127.0.0.1:5173` 访问前端，由 Vite proxy 转发 `/api` 和 `/ws`。

## 免责声明

本系统仅用于黑客松 Demo。所有风险评分、预测、报告和投资动作建议仅供技术演示，不构成投资建议或交易依据。
