import { demoPortfolio } from '../mock/portfolioMock'
import { normalizeRiskLevel, riskLabel } from './stockAdapter'

export const adaptPortfolio = (payload = {}, assets = []) => {
  const quantile = payload.quantile_risk || {}
  const score = Number(payload.risk_score_fused ?? payload.risk_score ?? payload.score ?? demoPortfolio.summary.score)
  const level = normalizeRiskLevel(payload.risk_level_fused ?? payload.risk_level, score)
  const exposure = payload.exposure_breakdown || payload.risk_contribution || payload.contributions
  const contributions = Array.isArray(exposure)
    ? exposure.map((item) => ({ name: item.name || item.stock_name || item.symbol, value: Number(item.value ?? item.contribution ?? item.weight ?? 0) }))
    : exposure && typeof exposure === 'object'
      ? Object.entries(exposure).map(([name, value]) => ({ name, value: Number(typeof value === 'object' ? value.value ?? value.contribution ?? value.weight : value) }))
      : demoPortfolio.contributions

  return {
    summary: {
      score,
      level,
      label: payload.risk_label || riskLabel(level),
      var95: Number(quantile.var_1d ?? payload.var95 ?? demoPortfolio.summary.var95),
      cvar95: Number(quantile.cvar_1d ?? payload.cvar95 ?? demoPortfolio.summary.cvar95),
      volatility: Number(payload.volatility ?? demoPortfolio.summary.volatility),
      text: payload.summary || payload.risk_summary || demoPortfolio.summary.text
    },
    holdings: assets.length ? assets : demoPortfolio.holdings,
    contributions: contributions.length ? contributions : demoPortfolio.contributions,
    recommendation: {
      action: payload.control_actions?.action || demoPortfolio.recommendation.action,
      items: payload.control_actions?.next_steps || payload.recommendations || demoPortfolio.recommendation.items
    },
    alerts: payload.alerts || demoPortfolio.alerts
  }
}
