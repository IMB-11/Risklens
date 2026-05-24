export const demoStockAnalysis = {
  stock: {
    symbol: 'NVDA',
    name: 'NVIDIA Corporation',
    market: 'NASDAQ',
    sector: 'Technology',
    price: 924.18,
    changePercent: -2.34,
    updatedAt: '2026-05-24 16:30'
  },
  risk: {
    score: 78,
    level: 'high',
    label: '高风险',
    var95: -4.8,
    cvar95: -6.2,
    volatility: 36.5,
    maxDrawdown: -11.8,
    summary: '近期波动率上升，舆情分歧扩大，短期预测置信区间变宽。'
  },
  recommendation: {
    action: '谨慎持有',
    confidence: 0.82,
    positionAdvice: '建议降低高仓位敞口，等待波动回落后再考虑加仓。',
    reasons: ['多模型预测显示未来 7 日波动加剧', 'VaR 已超过平衡型风险偏好阈值', '近期负面舆情占比上升']
  },
  prediction: {
    days: ['D+1', 'D+2', 'D+3', 'D+4', 'D+5', 'D+6', 'D+7'],
    prices: [918, 921, 915, 908, 912, 905, 899],
    lowerBound: [902, 899, 890, 884, 878, 872, 866],
    upperBound: [936, 941, 940, 934, 933, 929, 925]
  },
  sentiment: {
    overall: '偏中性，负面上升',
    positive: 34,
    neutral: 41,
    negative: 25,
    keywords: ['AI 芯片', '出口限制', '财报预期', '估值压力'],
    summary: '新闻与社媒讨论热度较高，市场对增长预期仍强，但估值和监管因素带来短期压力。'
  },
  riskFactors: [
    { name: '市场波动', value: 32 },
    { name: '舆情负面', value: 24 },
    { name: '模型分歧', value: 18 },
    { name: '宏观事件', value: 15 },
    { name: '流动性风险', value: 11 }
  ],
  news: [
    { title: 'NVIDIA faces renewed concerns over export restrictions', source: 'Market News', sentiment: 'negative', time: '14:20', impact: 'high' },
    { title: 'AI chip demand remains resilient into next quarter', source: 'Tech Wire', sentiment: 'positive', time: '13:45', impact: 'medium' },
    { title: 'Analysts debate valuation pressure across mega-cap semis', source: 'Finance Daily', sentiment: 'neutral', time: '12:18', impact: 'medium' }
  ],
  alerts: [
    { level: 'high', title: 'VaR 超过平衡型风险阈值', time: '14:32', description: '95% VaR 已达到 -4.8%，建议关注短期回撤风险。' },
    { level: 'medium', title: '模型分歧扩大', time: '13:56', description: 'FinanceLM 与量化模型对 7 日走势置信区间存在分歧。' }
  ],
  models: {
    consensus: 'bearish-neutral',
    confidence: 0.76,
    items: [
      { name: 'FinanceLM', signal: '谨慎', confidence: 0.81 },
      { name: 'LSTM', signal: '下行', confidence: 0.72 },
      { name: 'XGBoost', signal: '震荡', confidence: 0.75 }
    ]
  },
  source: 'demo'
}
