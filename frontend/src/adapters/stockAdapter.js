import { demoStockAnalysis } from '../mock/stockMock'

const clamp = (value, fallback = 0) => {
  const number = Number(value)
  return Number.isFinite(number) ? Math.max(0, Math.min(100, number)) : fallback
}

const confidence = (value, fallback = 0.7) => {
  const number = Number(value)
  if (!Number.isFinite(number)) return fallback
  return number > 1 ? Math.max(0, Math.min(1, number / 100)) : Math.max(0, Math.min(1, number))
}

const levelMap = {
  low: 'low',
  medium: 'medium',
  moderate: 'medium',
  high: 'high',
  critical: 'high',
  conservative: 'low',
  aggressive: 'high',
  '低风险': 'low',
  '中风险': 'medium',
  '高风险': 'high'
}

export const normalizeRiskLevel = (level, score = 50) => {
  const key = String(level || '').toLowerCase()
  if (levelMap[key]) return levelMap[key]
  if (score >= 70) return 'high'
  if (score >= 40) return 'medium'
  return 'low'
}

export const riskLabel = (level) => ({ low: '低风险', medium: '中风险', high: '高风险' }[level] || '中风险')
export const normalizeRiskLabel = (label, level) => {
  const key = String(label || '').toLowerCase()
  if (key === 'low') return '低风险'
  if (key === 'medium' || key === 'moderate') return '中风险'
  if (key === 'high' || key === 'critical') return '高风险'
  return label || riskLabel(level)
}

const mapFactors = (raw) => {
  if (Array.isArray(raw)) {
    return raw.map((item) => ({ name: item.name || item.factor || item.key || '风险因子', value: clamp(item.value ?? item.score ?? item.weight, 10) }))
  }
  if (raw && typeof raw === 'object') {
    return Object.entries(raw).map(([name, value]) => ({ name, value: clamp(typeof value === 'object' ? value.value ?? value.score ?? value.weight : value, 10) }))
  }
  return demoStockAnalysis.riskFactors
}

const mapAlerts = (alerts = []) => (Array.isArray(alerts) ? alerts : []).map((alert, index) => ({
  level: normalizeRiskLevel(alert.level || alert.risk_level || alert.severity, index === 0 ? 80 : 50),
  title: alert.title || alert.message || alert.alert_type || '风险提示',
  time: alert.time || alert.timestamp || '刚刚',
  description: alert.description || alert.detail || alert.message || '请关注风险因子变化。'
}))

const mapNews = (items = []) => (Array.isArray(items) ? items : []).map((item) => ({
  title: item.title || item.headline || '市场新闻',
  source: item.source || item.publisher || 'External News',
  time: item.time || item.publish_time || item.published_at || '刚刚',
  sentiment: item.sentiment || item.sentiment_type || 'neutral',
  impact: Number(item.weight || item.impact_score || 0) > 0.7 ? 'high' : 'medium'
}))

const mapModels = (modelPayload = {}, riskPayload = {}) => {
  const predictions = modelPayload.model_predictions || riskPayload.model_predictions || {}
  const items = Object.entries(predictions)
    .filter(([, pred]) => pred && pred.success !== false)
    .map(([name, pred]) => ({
      name,
      signal: pred.trend || pred.signal || pred.action || '观望',
      confidence: confidence(pred.confidence, 0.7)
    }))

  if (!items.length && modelPayload.current_suggestion) {
    items.push({
      name: 'FinanceLM',
      signal: modelPayload.current_suggestion.action || '谨慎',
      confidence: confidence(modelPayload.current_suggestion.confidence, 0.7)
    })
  }

  return {
    consensus: modelPayload.market_state || modelPayload.consensus || 'neutral',
    confidence: confidence(modelPayload.confidence ?? riskPayload.confidence, 0.7),
    items: items.length ? items : []
  }
}

const mapPrediction = (modelPayload = {}, stock = demoStockAnalysis.stock) => {
  const raw = modelPayload.future_forecast?.predictions_7d || []
  const prices = raw.map((item) => Number(item.price ?? item.predicted_price ?? item.value ?? item)).filter(Number.isFinite)
  const base = Number(stock.price || demoStockAnalysis.stock.price)
  const series = prices.length ? prices.slice(0, 7) : demoStockAnalysis.prediction.prices.map((price, index) => Math.round((price / demoStockAnalysis.stock.price) * base + index))
  return {
    days: demoStockAnalysis.prediction.days,
    prices: series,
    lowerBound: series.map((price, index) => Number((price * (0.982 - index * 0.002)).toFixed(2))),
    upperBound: series.map((price, index) => Number((price * (1.018 + index * 0.002)).toFixed(2)))
  }
}

export const adaptStockAnalysis = ({ riskPayload = {}, modelPayload = {}, newsPayload = {}, stockName = 'NVDA', riskLevel = 'moderate' } = {}) => {
  const fusedScore = clamp(riskPayload.risk_score_fused ?? riskPayload.risk_score ?? riskPayload.score, demoStockAnalysis.risk.score)
  const level = normalizeRiskLevel(riskPayload.risk_level_fused ?? riskPayload.risk_level ?? riskPayload.risk_label, fusedScore)
  const quantile = riskPayload.quantile_risk || {}
  const resolved = riskPayload.stock_resolution || riskPayload.resolution || {}
  const symbol = resolved.code || riskPayload.stock_code || riskPayload.symbol || stockName.toUpperCase()
  const stock = {
    symbol,
    name: resolved.name || riskPayload.stock_name || stockName.toUpperCase(),
    market: resolved.market || riskPayload.market || (symbol.includes('.') ? 'CN' : 'NASDAQ'),
    sector: resolved.sector || riskPayload.sector || 'Technology',
    price: Number(riskPayload.latest_price || riskPayload.price || demoStockAnalysis.stock.price),
    changePercent: Number(riskPayload.change_percent || riskPayload.changePercent || demoStockAnalysis.stock.changePercent),
    updatedAt: riskPayload.timestamp || modelPayload.analysis_time || new Date().toLocaleString()
  }

  const news = mapNews(newsPayload.top_news || riskPayload.diagnostics?.top_news || riskPayload.top_news || [])
  const positive = newsPayload.summary?.positive_count ?? riskPayload.diagnostics?.news_snapshot?.positive_count
  const negative = newsPayload.summary?.negative_count ?? riskPayload.diagnostics?.news_snapshot?.negative_count
  const neutral = newsPayload.summary?.neutral_count ?? riskPayload.diagnostics?.news_snapshot?.neutral_count
  const total = Math.max(1, Number(positive || 0) + Number(negative || 0) + Number(neutral || 0))

  return {
    stock,
    risk: {
      score: fusedScore,
      level,
      label: normalizeRiskLabel(riskPayload.risk_label, level),
      var95: -Math.abs(Number(quantile.var_1d ?? quantile.var95 ?? riskPayload.var95 ?? demoStockAnalysis.risk.var95)),
      cvar95: -Math.abs(Number(quantile.cvar_1d ?? quantile.cvar95 ?? riskPayload.cvar95 ?? demoStockAnalysis.risk.cvar95)),
      volatility: Number(riskPayload.volatility ?? quantile.volatility ?? demoStockAnalysis.risk.volatility),
      maxDrawdown: Number(riskPayload.max_drawdown ?? demoStockAnalysis.risk.maxDrawdown),
      summary: riskPayload.risk_summary || riskPayload.summary || demoStockAnalysis.risk.summary
    },
    recommendation: {
      action: riskPayload.control_actions?.action || modelPayload.current_suggestion?.action || demoStockAnalysis.recommendation.action,
      confidence: confidence(modelPayload.current_suggestion?.confidence ?? riskPayload.confidence, demoStockAnalysis.recommendation.confidence),
      positionAdvice: riskPayload.control_actions?.position_advice || riskPayload.control_actions?.summary || demoStockAnalysis.recommendation.positionAdvice,
      reasons: riskPayload.control_actions?.next_steps || riskPayload.top_risk_drivers || demoStockAnalysis.recommendation.reasons
    },
    prediction: mapPrediction(modelPayload, stock),
    sentiment: {
      overall: newsPayload.summary?.sentiment_trend || modelPayload.sentiment_analysis?.sentiment_score || demoStockAnalysis.sentiment.overall,
      positive: Math.round((Number(positive ?? 3) / total) * 100),
      neutral: Math.round((Number(neutral ?? 4) / total) * 100),
      negative: Math.round((Number(negative ?? 2) / total) * 100),
      keywords: riskPayload.diagnostics?.signal_snapshot?.keywords || demoStockAnalysis.sentiment.keywords,
      summary: newsPayload.summary_text || demoStockAnalysis.sentiment.summary
    },
    riskFactors: mapFactors(riskPayload.factor_breakdown),
    news: news.length ? news : demoStockAnalysis.news,
    alerts: mapAlerts(riskPayload.alerts).length ? mapAlerts(riskPayload.alerts) : demoStockAnalysis.alerts,
    models: mapModels(modelPayload, riskPayload),
    source: riskPayload && Object.keys(riskPayload).length ? 'api' : 'demo',
    riskProfile: riskLevel
  }
}
