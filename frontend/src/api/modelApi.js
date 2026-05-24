import { apiClient, unwrap } from './http'

export const predictWithModels = (stockName, riskLevel = 'moderate') =>
  apiClient.post('/multi-model/predict', {
    stock_name: stockName,
    risk_level: riskLevel,
    limit: 20
  }).then(unwrap)

export const getModelStats = () => apiClient.get('/model/stats').then(unwrap)
