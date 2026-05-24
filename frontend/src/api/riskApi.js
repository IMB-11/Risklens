import { apiClient, unwrap } from './http'

export const assessRisk = (stockName, riskLevel = 'moderate') =>
  apiClient.post('/risk/assess', {
    stock_name: stockName,
    risk_level: riskLevel,
    limit: 8,
    include_social: false,
    enable_alt_data: false
  }).then(unwrap)

export const getRiskAlerts = (stockName, riskLevel = 'moderate') =>
  apiClient.post('/risk/alerts', {
    stock_name: stockName,
    risk_level: riskLevel,
    limit: 8,
    include_social: false,
    enable_alt_data: false
  }).then(unwrap)
