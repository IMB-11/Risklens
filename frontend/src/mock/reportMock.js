export const demoRiskReport = {
  stock: { symbol: 'NVDA', name: 'NVIDIA Corporation', riskProfile: 'moderate', updatedAt: '2026-05-24 16:30' },
  summary: {
    title: 'NVDA AI 风控分析报告',
    riskScore: 78,
    riskLevel: 'high',
    riskLabel: '高风险',
    oneSentenceConclusion: '短期波动与估值压力上升，建议控制仓位并等待风险释放。'
  },
  llm: {
    provider: 'Qwen / FinanceLM',
    usedQwen: true,
    status: 'generated',
    text: '综合行情、舆情、VaR/CVaR 与多模型预测结果，NVDA 当前处于高波动风险区间。增长预期仍然支撑中长期叙事，但短期估值压力、监管不确定性与模型分歧使风险预算消耗加快。建议在平衡型风险偏好下控制单票仓位，并等待波动率回落或风险事件明朗后再提高敞口。',
    confidence: 0.76
  },
  sections: [
    { title: '核心结论', type: 'conclusion', content: ['当前综合风险评分处于高风险区间。', '短期预测置信区间扩大，模型对后续走势存在分歧。', '建议降低高仓位敞口。'] },
    { title: '主要风险驱动', type: 'risk_drivers', content: ['市场波动上升', '负面舆情占比提高', 'VaR 超过平衡型风险阈值'] },
    { title: '模型融合说明', type: 'model_fusion', content: ['基础风险评分为 72。', '融合后风险评分为 78。', 'shadow signal 当前不直接覆盖主决策。'] },
    { title: '风控动作建议', type: 'actions', content: ['降低单票暴露。', '设置分层止损。', '等待波动率回落后再考虑加仓。'] }
  ],
  metrics: { var95: -4.8, cvar95: -6.2, riskScoreBase: 72, riskScoreFused: 78, riskScoreShadow: 75 },
  alerts: [{ level: 'high', title: 'VaR 超过阈值', time: '14:32', description: '95% VaR 已达到 -4.8%。' }],
  raw: {}
}
