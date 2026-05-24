import { getRiskReportApi } from '../api/reportApi'
import { adaptReport } from '../adapters/reportAdapter'
import { demoRiskReport } from '../mock/reportMock'
import { isDemoModeEnabled } from './configService'

export const getRiskReport = (stockName, riskLevel) => getRiskReportApi(stockName, riskLevel)

export const generateRiskReport = async (stockName = 'NVDA', riskLevel = 'moderate') => {
  if (isDemoModeEnabled()) return demoRiskReport
  try {
    const payload = await getRiskReport(stockName, riskLevel)
    return adaptReport(payload, stockName, riskLevel)
  } catch {
    return demoRiskReport
  }
}
