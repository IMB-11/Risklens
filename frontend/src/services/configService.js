import { getConfigStatusApi, saveConfigKeysApi } from '../api/configApi'

const DEMO_MODE_KEY = 'risk_terminal_demo_mode'
const SERPAPI_KEY = 'risk_terminal_serpapi_key'
const TAVILY_KEY = 'risk_terminal_tavily_key'

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

export const getStoredApiKeyStatus = () => ({
  serpapiConfigured: Boolean(localStorage.getItem(SERPAPI_KEY)),
  tavilyConfigured: Boolean(localStorage.getItem(TAVILY_KEY)),
  serpapiMasked: mask(localStorage.getItem(SERPAPI_KEY)),
  tavilyMasked: mask(localStorage.getItem(TAVILY_KEY))
})

export const getConfigStatus = async () => {
  try {
    const remote = await getConfigStatusApi()
    return { ...remote, local: getStoredApiKeyStatus(), source: 'api' }
  } catch {
    return {
      serpapi_configured: getStoredApiKeyStatus().serpapiConfigured,
      tavily_configured: getStoredApiKeyStatus().tavilyConfigured,
      demo_mode_available: true,
      external_search_enabled: false,
      local: getStoredApiKeyStatus(),
      source: 'local'
    }
  }
}

export const saveUserApiKeys = async ({ serpapiApiKey, tavilyApiKey }) => {
  if (serpapiApiKey) localStorage.setItem(SERPAPI_KEY, serpapiApiKey)
  if (tavilyApiKey) localStorage.setItem(TAVILY_KEY, tavilyApiKey)
  try {
    await saveConfigKeysApi({ serpapi_api_key: serpapiApiKey, tavily_api_key: tavilyApiKey })
  } catch {
    // Local storage is enough for the demo workflow.
  }
  return getStoredApiKeyStatus()
}

export const clearUserApiKeys = () => {
  localStorage.removeItem(SERPAPI_KEY)
  localStorage.removeItem(TAVILY_KEY)
  return getStoredApiKeyStatus()
}
