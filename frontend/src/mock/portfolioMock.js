export const defaultPortfolioAssets = [
  { symbol: 'NVDA', weight: 35, sector: 'Technology' },
  { symbol: 'AAPL', weight: 25, sector: 'Technology' },
  { symbol: 'MSFT', weight: 20, sector: 'Technology' },
  { symbol: 'TSLA', weight: 20, sector: 'Consumer Cyclical' }
]

export const demoPortfolio = {
  summary: {
    score: 72,
    level: 'high',
    label: '高风险',
    var95: -3.9,
    cvar95: -5.4,
    volatility: 28.6,
    text: '组合集中于高成长科技资产，NVDA 与 TSLA 对尾部风险贡献较高。'
  },
  holdings: defaultPortfolioAssets,
  contributions: [
    { name: 'NVDA', value: 41 },
    { name: 'TSLA', value: 26 },
    { name: 'AAPL', value: 18 },
    { name: 'MSFT', value: 15 }
  ],
  recommendation: {
    action: '降低高波动敞口',
    items: ['降低 NVDA 和 TSLA 的高波动敞口', '增加低相关性资产以降低组合 CVaR', '当前组合风险主要由 NVDA 贡献']
  },
  alerts: [
    { level: 'high', title: '组合集中度偏高', time: '14:30', description: '科技板块权重超过 80%，建议引入低相关资产。' }
  ]
}
