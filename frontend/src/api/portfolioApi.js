import { apiClient, unwrap } from './http'

export const assessPortfolio = (assets, riskLevel = 'moderate') =>
  apiClient.post('/risk/portfolio', {
    assets: assets.map((asset) => ({
      stock_name: asset.symbol || asset.stock_name,
      weight: Number(asset.weight) / (Number(asset.weight) > 1 ? 100 : 1),
      sector: asset.sector || 'unknown',
      style: asset.style || 'unknown',
      leverage: asset.leverage || 1,
      crowding: asset.crowding || 0.5,
      expected_return: asset.expected_return || 0,
      volatility: asset.volatility || 0.02
    })),
    risk_level: riskLevel
  }).then(unwrap)
