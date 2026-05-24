import { getConfigStatusApi, saveConfigKeysApi } from '../api/configApi'

const DEMO_MODE_KEY = 'risk_terminal_demo_mode'
const SERPAPI_KEY = 'risk_terminal_serpapi_key'
const TAVILY_KEY = 'risk_terminal_tavily_key'
const QWEN_API_KEY = 'risk_terminal_qwen_api_key'
const DEEPSEEK_API_KEY = 'risk_terminal_deepseek_api_key'
const LLM_PROVIDER_KEY = 'risk_terminal_llm_provider'

export const API_KEY_STORAGE = {
  serpapi: SERPAPI_KEY,
  tavily: TAVILY_KEY,
  qwen: QWEN_API_KEY,
  deepseek: DEEPSEEK_API_KEY,
  llmProvider: LLM_PROVIDER_KEY
}

export const isDemoModeEnabled = () => {
  const stored = localStorage.getItem(DEMO_MODE_KEY)
  if (stored === null) return import.meta.env.VITE_DEFAULT_DEMO_MODE === 'true'
  return stored === 'true'
}

export const setDemoMode = (enabled) => {
  localStorage.setItem(DEMO_MODE_KEY, String(Boolean(enabled)))
  window.dispatchEvent(new CustomEvent('demo-mode-change', { detail: Boolean(enabled) }))
}

const mask = (key) => key ? `${key.slice(0, 4)}****${key.slice(-4)}` : ''
const provider = () => localStorage.getItem(LLM_PROVIDER_KEY) || 'auto'

export const getStoredApiKeyStatus = () => ({
  serpapiConfigured: Boolean(localStorage.getItem(SERPAPI_KEY)),
  tavilyConfigured: Boolean(localStorage.getItem(TAVILY_KEY)),
  qwenApiConfigured: Boolean(localStorage.getItem(QWEN_API_KEY)),
  deepseekConfigured: Boolean(localStorage.getItem(DEEPSEEK_API_KEY)),
  serpapiMasked: mask(localStorage.getItem(SERPAPI_KEY)),
  tavilyMasked: mask(localStorage.getItem(TAVILY_KEY)),
  qwenMasked: mask(localStorage.getItem(QWEN_API_KEY)),
  deepseekMasked: mask(localStorage.getItem(DEEPSEEK_API_KEY)),
  llmProvider: provider()
})

export const getConfigStatus = async () => {
  const local = getStoredApiKeyStatus()
  try {
    const remote = await getConfigStatusApi()
    return { ...remote, local, source: 'api' }
  } catch {
    return {
      serpapi_configured: local.serpapiConfigured,
      tavily_configured: local.tavilyConfigured,
      qwen_api_configured: local.qwenApiConfigured,
      deepseek_configured: local.deepseekConfigured,
      llm_provider: local.llmProvider,
      demo_mode_available: true,
      external_search_enabled: local.serpapiConfigured || local.tavilyConfigured,
      local,
      source: 'local'
    }
  }
}

export const saveUserApiKeys = async ({ serpapiApiKey, tavilyApiKey, qwenApiKey, deepseekApiKey, llmProvider }) => {
  if (serpapiApiKey) localStorage.setItem(SERPAPI_KEY, serpapiApiKey)
  if (tavilyApiKey) localStorage.setItem(TAVILY_KEY, tavilyApiKey)
  if (qwenApiKey) localStorage.setItem(QWEN_API_KEY, qwenApiKey)
  if (deepseekApiKey) localStorage.setItem(DEEPSEEK_API_KEY, deepseekApiKey)
  if (llmProvider) localStorage.setItem(LLM_PROVIDER_KEY, llmProvider)
  if (serpapiApiKey || tavilyApiKey || qwenApiKey || deepseekApiKey || llmProvider) setDemoMode(false)

  try {
    await saveConfigKeysApi({
      serpapi_api_key: serpapiApiKey,
      tavily_api_key: tavilyApiKey,
      qwen_api_key: qwenApiKey,
      deepseek_api_key: deepseekApiKey,
      llm_provider: llmProvider || provider()
    })
  } catch {
    // Local storage is enough for the local demo workflow; headers will apply on the next API call.
  }
  return getStoredApiKeyStatus()
}

export const clearUserApiKeys = () => {
  localStorage.removeItem(SERPAPI_KEY)
  localStorage.removeItem(TAVILY_KEY)
  localStorage.removeItem(QWEN_API_KEY)
  localStorage.removeItem(DEEPSEEK_API_KEY)
  localStorage.removeItem(LLM_PROVIDER_KEY)
  return getStoredApiKeyStatus()
}