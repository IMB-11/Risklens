import { apiClient, unwrap } from './http'

export const getRiskReportApi = (stockName, riskLevel = 'moderate', limit = 20) =>
  apiClient.get(`/risk/report/${encodeURIComponent(stockName)}`, {
    params: { risk_level: riskLevel, limit }
  }).then(unwrap)
