import axios from 'axios'
import { API_KEY_STORAGE, isDemoModeEnabled } from '../services/configService'

export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  timeout: 45000
})

apiClient.interceptors.request.use((config) => {
  if (isDemoModeEnabled()) return config

  const serpapiKey = localStorage.getItem(API_KEY_STORAGE.serpapi)
  const tavilyKey = localStorage.getItem(API_KEY_STORAGE.tavily)
  const qwenKey = localStorage.getItem(API_KEY_STORAGE.qwen)
  const deepseekKey = localStorage.getItem(API_KEY_STORAGE.deepseek)
  const llmProvider = localStorage.getItem(API_KEY_STORAGE.llmProvider)

  if (serpapiKey) config.headers['X-SerpAPI-Key'] = serpapiKey
  if (tavilyKey) config.headers['X-Tavily-API-Key'] = tavilyKey
  if (qwenKey) config.headers['X-Qwen-API-Key'] = qwenKey
  if (deepseekKey) config.headers['X-DeepSeek-API-Key'] = deepseekKey
  if (llmProvider) config.headers['X-LLM-Provider'] = llmProvider

  return config
})

export const unwrap = (response) => response?.data