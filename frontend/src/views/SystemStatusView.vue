<template>
  <section>
    <div class="page-header"><div><h1 class="page-title">系统状态</h1><div class="muted">后端健康检查、外部检索配置与本机演示设置。</div></div></div>
    <div class="dashboard-grid">
      <article class="card span-6">
        <div class="card-header"><h3 class="card-title">外部检索配置</h3><span class="badge" :class="configured ? 'low' : 'medium'">{{ configured ? '已配置' : '外部检索未配置' }}</span></div>
        <div class="list">
          <input class="field" type="password" v-model="serpapiApiKey" placeholder="SerpAPI API Key" />
          <input class="field" type="password" v-model="tavilyApiKey" placeholder="Tavily API Key" />
          <div class="row-between" style="justify-content:flex-start;"><button class="btn" @click="saveKeys">保存</button><button class="btn secondary" @click="loadStatus">连接测试</button><button class="btn secondary" @click="clearKeys">清除本地 Key</button></div>
          <p class="muted">用户不填写 Key 时，系统仍可使用演示模式与已有后端能力。生产环境不建议把敏感 Key 保存在 localStorage。</p>
        </div>
      </article>
      <article class="card span-6">
        <div class="card-header"><h3 class="card-title">配置状态</h3><span class="badge">{{ status?.source || 'local' }}</span></div>
        <div class="kpi-grid">
          <div class="kpi"><span class="soft">SerpAPI</span><strong>{{ status?.serpapi_configured ? '已配置' : '未配置' }}</strong></div>
          <div class="kpi"><span class="soft">Tavily</span><strong>{{ status?.tavily_configured ? '已配置' : '未配置' }}</strong></div>
          <div class="kpi"><span class="soft">演示模式</span><strong>{{ status?.demo_mode_available ? '可用' : '不可用' }}</strong></div>
          <div class="kpi"><span class="soft">外部检索</span><strong>{{ status?.external_search_enabled ? '可用' : '降级' }}</strong></div>
        </div>
        <p class="muted" v-if="status?.local?.serpapiMasked">SerpAPI: {{ status.local.serpapiMasked }}</p>
        <p class="muted" v-if="status?.local?.tavilyMasked">Tavily: {{ status.local.tavilyMasked }}</p>
      </article>
    </div>
  </section>
</template>
<script setup>
import { computed, onMounted, ref } from 'vue'
import { clearUserApiKeys, getConfigStatus, saveUserApiKeys } from '../services/configService'
const serpapiApiKey = ref(''); const tavilyApiKey = ref(''); const status = ref(null)
const configured = computed(() => status.value?.serpapi_configured || status.value?.tavily_configured)
const loadStatus = async () => { status.value = await getConfigStatus() }
const saveKeys = async () => { await saveUserApiKeys({ serpapiApiKey: serpapiApiKey.value, tavilyApiKey: tavilyApiKey.value }); serpapiApiKey.value = ''; tavilyApiKey.value = ''; await loadStatus() }
const clearKeys = async () => { clearUserApiKeys(); await loadStatus() }
onMounted(loadStatus)
</script>
