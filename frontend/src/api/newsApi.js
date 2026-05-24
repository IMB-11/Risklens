import { apiClient, unwrap } from './http'

export const getNewsSummaryApi = (stockName) =>
  apiClient.get(`/news/summary/${encodeURIComponent(stockName)}`).then(unwrap)
