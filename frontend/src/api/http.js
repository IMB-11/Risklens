import axios from 'axios'
import { isDemoModeEnabled } from '../services/configService'

export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  timeout: 45000
})

apiClient.interceptors.request.use((config) => {
  if (isDemoModeEnabled()) return config

  const serpapiKey = localStorage.getItem('risk_terminal_serpapi_key')
  const tavilyKey = localStorage.getItem('risk_terminal_tavily_key')

  if (serpapiKey) config.headers['X-SerpAPI-Key'] = serpapiKey
  if (tavilyKey) config.headers['X-Tavily-API-Key'] = tavilyKey

  return config
})

export const unwrap = (response) => response?.data
