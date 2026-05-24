import { assessRisk } from '../api/riskApi'
import { predictWithModels } from '../api/modelApi'
import { getNewsSummaryApi } from '../api/newsApi'
import { adaptStockAnalysis } from '../adapters/stockAdapter'
import { createDemoStockAnalysis } from '../mock/stockMock'
import { isDemoModeEnabled } from './configService'

export const getHealth = async () => {
  const { apiClient } = await import('../api/http')
  return apiClient.get('/health').then((response) => response.data)
}

export const getRiskAssessment = (stockName, riskLevel) => assessRisk(stockName, riskLevel)
export const getMultiModelPrediction = (stockName, riskLevel) => predictWithModels(stockName, riskLevel)
export const getNewsSummary = (stockName) => getNewsSummaryApi(stockName)

export const analyzeStock = async (stockName = 'NVDA', riskLevel = 'moderate') => {
  if (isDemoModeEnabled()) return createDemoStockAnalysis(stockName)

  try {
    const [riskPayload, modelPayload, newsPayload] = await Promise.allSettled([
      getRiskAssessment(stockName, riskLevel),
      getMultiModelPrediction(stockName, riskLevel),
      getNewsSummary(stockName)
    ])

    const data = adaptStockAnalysis({
      riskPayload: riskPayload.status === 'fulfilled' ? riskPayload.value : {},
      modelPayload: modelPayload.status === 'fulfilled' ? modelPayload.value : {},
      newsPayload: newsPayload.status === 'fulfilled' ? newsPayload.value : {},
      stockName,
      riskLevel
    })

    return data.source === 'api' ? data : createDemoStockAnalysis(stockName)
  } catch {
    return createDemoStockAnalysis(stockName)
  }
}
