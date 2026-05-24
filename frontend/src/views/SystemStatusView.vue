<template>
  <section>
    <div class="page-header">
      <div>
        <h1 class="page-title">系统状态</h1>
        <div class="muted">后端健康检查、外部检索配置与 LLM 报告生成设置。</div>
      </div>
    </div>

    <div class="dashboard-grid">
      <article class="card span-6">
        <div class="card-header">
          <h3 class="card-title">外部检索配置</h3>
          <span class="badge" :class="searchConfigured ? 'low' : 'medium'">
            {{ searchConfigured ? '已配置' : '外部检索未配置' }}
          </span>
        </div>
        <div class="list">
          <input class="field" type="password" v-model="serpapiApiKey" placeholder="SerpAPI API Key" />
          <input class="field" type="password" v-model="tavilyApiKey" placeholder="Tavily API Key" />
          <p class="muted">不填写检索 Key 时，新闻和网页信息会自动降级，系统仍可继续演示。</p>
        </div>
      </article>

      <article class="card span-6">
        <div class="card-header">
          <h3 class="card-title">LLM 报告配置</h3>
          <span class="badge" :class="llmConfigured ? 'low' : 'medium'">
            {{ llmConfigured ? 'LLM 可用' : '规则增强可用' }}
          </span>
        </div>
        <div class="list">
          <select class="select" v-model="llmProvider">
            <option value="auto">自动选择</option>
            <option value="qwen_api">Qwen API</option>
            <option value="deepseek">DeepSeek API</option>
            <option value="local_qwen">本地 Qwen</option>
            <option value="rules">规则增强报告</option>
          </select>
          <input class="field" type="password" v-model="qwenApiKey" placeholder="Qwen / DashScope API Key" />
          <input class="field" type="password" v-model="deepseekApiKey" placeholder="DeepSeek API Key" />
          <div class="row-between" style="justify-content:flex-start; flex-wrap: wrap;">
            <button class="btn" @click="saveKeys">保存配置</button>
            <button class="btn secondary" @click="loadStatus">连接测试</button>
            <button class="btn secondary" @click="clearKeys">清除本地 Key</button>
          </div>
          <p class="muted">保存后会关闭演示模式；之后点击 Analyze 或进入 AI 风控报告页，会由后端按 provider 选择对应 LLM。</p>
        </div>
      </article>

      <article class="card span-12">
        <div class="card-header">
          <h3 class="card-title">配置状态</h3>
          <span class="badge">{{ status?.source || 'local' }}</span>
        </div>
        <div class="kpi-grid status-grid">
          <div class="kpi"><span class="soft">SerpAPI</span><strong>{{ status?.serpapi_configured ? '已配置' : '未配置' }}</strong></div>
          <div class="kpi"><span class="soft">Tavily</span><strong>{{ status?.tavily_configured ? '已配置' : '未配置' }}</strong></div>
          <div class="kpi"><span class="soft">Qwen API</span><strong>{{ status?.qwen_api_configured ? '已配置' : '未配置' }}</strong></div>
          <div class="kpi"><span class="soft">DeepSeek</span><strong>{{ status?.deepseek_configured ? '已配置' : '未配置' }}</strong></div>
          <div class="kpi"><span class="soft">LLM Provider</span><strong>{{ providerLabel(status?.llm_provider || status?.local?.llmProvider) }}</strong></div>
          <div class="kpi"><span class="soft">演示模式</span><strong>{{ status?.demo_mode_available ? '可用' : '不可用' }}</strong></div>
        </div>
        <div class="list" style="margin-top: 12px;">
          <p class="muted" v-if="status?.local?.serpapiMasked">SerpAPI: {{ status.local.serpapiMasked }}</p>
          <p class="muted" v-if="status?.local?.tavilyMasked">Tavily: {{ status.local.tavilyMasked }}</p>
          <p class="muted" v-if="status?.local?.qwenMasked">Qwen API: {{ status.local.qwenMasked }}</p>
          <p class="muted" v-if="status?.local?.deepseekMasked">DeepSeek: {{ status.local.deepseekMasked }}</p>
          <p class="muted">本地黑客松 Demo 会把用户填写的 Key 存在浏览器 localStorage，并通过请求头发给后端；生产环境应改为服务端安全配置。</p>
        </div>
      </article>
    </div>
  </section>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { clearUserApiKeys, getConfigStatus, getStoredApiKeyStatus, saveUserApiKeys } from '../services/configService'

const serpapiApiKey = ref('')
const tavilyApiKey = ref('')
const qwenApiKey = ref('')
const deepseekApiKey = ref('')
const llmProvider = ref(getStoredApiKeyStatus().llmProvider || 'auto')
const status = ref(null)

const searchConfigured = computed(() => status.value?.serpapi_configured || status.value?.tavily_configured)
const llmConfigured = computed(() => status.value?.qwen_api_configured || status.value?.deepseek_configured || status.value?.llm_provider === 'local_qwen')

const providerLabel = (value = 'auto') => ({
  auto: '自动选择',
  qwen_api: 'Qwen API',
  deepseek: 'DeepSeek API',
  local_qwen: '本地 Qwen',
  rules: '规则增强'
}[value] || value)

const loadStatus = async () => {
  status.value = await getConfigStatus()
  llmProvider.value = status.value?.local?.llmProvider || status.value?.llm_provider || 'auto'
}

const saveKeys = async () => {
  await saveUserApiKeys({
    serpapiApiKey: serpapiApiKey.value,
    tavilyApiKey: tavilyApiKey.value,
    qwenApiKey: qwenApiKey.value,
    deepseekApiKey: deepseekApiKey.value,
    llmProvider: llmProvider.value
  })
  serpapiApiKey.value = ''
  tavilyApiKey.value = ''
  qwenApiKey.value = ''
  deepseekApiKey.value = ''
  await loadStatus()
}

const clearKeys = async () => {
  clearUserApiKeys()
  llmProvider.value = 'auto'
  await loadStatus()
}

onMounted(loadStatus)
</script>