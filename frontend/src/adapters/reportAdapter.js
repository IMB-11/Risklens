import { demoRiskReport } from '../mock/reportMock'
import { normalizeRiskLabel, normalizeRiskLevel, riskLabel } from './stockAdapter'

const listFrom = (value, fallback = []) => {
  if (Array.isArray(value)) return value.map((item) => (typeof item === 'string' ? item : item.title || item.message || JSON.stringify(item)))
  if (typeof value === 'string' && value.trim()) return [value]
  return fallback
}

export const adaptReport = (payload = {}, stockName = 'NVDA', riskLevel = 'moderate') => {
  if (!payload || !Object.keys(payload).length) return demoRiskReport

  const fusion = payload.model_fusion || {}
  const score = Number(payload.risk_score_fused ?? payload.risk_score ?? fusion.risk_score_fused ?? demoRiskReport.summary.riskScore)
  const level = normalizeRiskLevel(payload.risk_level_fused ?? payload.risk_level, score)
  const narrative = payload.narrative || ''
  const drivers = listFrom(payload.top_risk_drivers, demoRiskReport.sections[1].content)
  const actions = listFrom(payload.control_actions?.next_steps || payload.control_actions?.actions, demoRiskReport.sections[3].content)
  const quantile = payload.quantile_risk || {}
  const fallbackText = [
    `${stockName} 当前综合风险评分为 ${score}，风险等级为 ${riskLabel(level)}。`,
    drivers.length ? `主要风险驱动包括：${drivers.join('、')}。` : '',
    actions.length ? `建议动作：${actions.join('；')}。` : ''
  ].filter(Boolean).join('\n')

  return {
    stock: {
      symbol: String(payload.stock_code || stockName).toUpperCase(),
      name: payload.stock_name || stockName,
      riskProfile: payload.risk_profile || riskLevel,
      updatedAt: payload.timestamp || new Date().toLocaleString()
    },
    summary: {
      title: `${String(stockName).toUpperCase()} AI 风控分析报告`,
      riskScore: score,
      riskLevel: level,
      riskLabel: normalizeRiskLabel(payload.risk_label, level),
      oneSentenceConclusion: payload.one_sentence_conclusion || drivers[0] || demoRiskReport.summary.oneSentenceConclusion
    },
    llm: {
      provider: payload.narrative_meta?.used_qwen ? 'Qwen / FinanceLM' : '规则增强报告',
      usedQwen: Boolean(payload.narrative_meta?.used_qwen),
      status: narrative ? 'generated' : 'structured',
      text: narrative || fallbackText,
      confidence: Number(payload.narrative_meta?.confidence || payload.confidence || 0.72)
    },
    sections: [
      { title: '核心结论', type: 'conclusion', content: listFrom(payload.narrative_meta?.conclusions, [fallbackText]) },
      { title: '主要风险驱动', type: 'risk_drivers', content: drivers },
      { title: '舆情与事件影响', type: 'sentiment', content: listFrom(payload.news_snapshot?.summary, ['外部新闻与事件数据已纳入风险评分。']) },
      { title: 'VaR / CVaR 解读', type: 'quantile', content: [`VaR 95%：${quantile.var_1d ?? demoRiskReport.metrics.var95}`, `CVaR 95%：${quantile.cvar_1d ?? demoRiskReport.metrics.cvar95}`] },
      { title: '模型融合说明', type: 'model_fusion', content: [`基础风险评分为 ${fusion.risk_score_base ?? payload.risk_score ?? score}。`, `融合后风险评分为 ${fusion.risk_score_fused ?? score}。`, `Shadow 模型评分为 ${fusion.risk_score_shadow ?? '未启用'}。`] },
      { title: '风控动作建议', type: 'actions', content: actions }
    ],
    metrics: {
      var95: -Math.abs(Number(quantile.var_1d ?? demoRiskReport.metrics.var95)),
      cvar95: -Math.abs(Number(quantile.cvar_1d ?? demoRiskReport.metrics.cvar95)),
      riskScoreBase: Number(fusion.risk_score_base ?? payload.risk_score ?? score),
      riskScoreFused: Number(fusion.risk_score_fused ?? score),
      riskScoreShadow: Number(fusion.risk_score_shadow ?? score)
    },
    alerts: Array.isArray(payload.alerts) ? payload.alerts : [],
    raw: payload
  }
}
