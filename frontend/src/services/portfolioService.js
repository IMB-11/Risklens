import { assessPortfolio } from '../api/portfolioApi'
import { adaptPortfolio } from '../adapters/portfolioAdapter'
import { demoPortfolio } from '../mock/portfolioMock'
import { isDemoModeEnabled } from './configService'

export const analyzePortfolio = async (assets, riskLevel = 'moderate') => {
  if (isDemoModeEnabled()) return demoPortfolio
  try {
    const payload = await assessPortfolio(assets, riskLevel)
    return adaptPortfolio(payload, assets)
  } catch {
    return demoPortfolio
  }
}
