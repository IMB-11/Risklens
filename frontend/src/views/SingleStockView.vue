<template>
  <section>
    <div class="page-header">
      <div><h1 class="page-title">单股票风险分析</h1><div class="muted">真实接口优先，接口不可用时自动进入稳定演示数据。</div></div>
      <span class="badge" :class="analysis?.source === 'api' ? 'low' : 'medium'">{{ analysis?.source === 'api' ? '真实数据已更新' : '演示模式' }}</span>
    </div>
    <AnalysisProgressOverlay :show="loading" :active-index="activeStep" />
    <div v-if="analysis" class="dashboard-grid">
      <StockOverviewCard class="span-3" :stock="analysis.stock" />
      <RiskScoreCard class="span-3" :risk="analysis.risk" />
      <VarCvarCard class="span-3" :risk="analysis.risk" />
      <ModelConsensusCard class="span-3" :models="analysis.models" />
      <AiRecommendationCard class="span-4" :recommendation="analysis.recommendation" />
      <PriceForecastChart class="span-8" :prediction="analysis.prediction" />
      <RiskFactorBarChart class="span-5" :factors="analysis.riskFactors" />
      <SentimentPanel class="span-7" :sentiment="analysis.sentiment" />
      <NewsList class="span-6" :news="analysis.news" />
      <AlertTimeline class="span-6" :alerts="analysis.alerts" />
    </div>
  </section>
</template>

<script setup>
import { onMounted, ref, watch } from 'vue'
import { analyzeStock } from '../services/stockAnalysisService'
import { subscribeRealtimeAlerts } from '../services/realtimeService'
import StockOverviewCard from '../components/stock/StockOverviewCard.vue'
import RiskScoreCard from '../components/stock/RiskScoreCard.vue'
import AiRecommendationCard from '../components/stock/AiRecommendationCard.vue'
import PriceForecastChart from '../components/stock/PriceForecastChart.vue'
import RiskFactorBarChart from '../components/stock/RiskFactorBarChart.vue'
import VarCvarCard from '../components/stock/VarCvarCard.vue'
import SentimentPanel from '../components/stock/SentimentPanel.vue'
import NewsList from '../components/stock/NewsList.vue'
import AlertTimeline from '../components/stock/AlertTimeline.vue'
import ModelConsensusCard from '../components/stock/ModelConsensusCard.vue'
import AnalysisProgressOverlay from '../components/stock/AnalysisProgressOverlay.vue'

const props = defineProps({ stockName: String, riskLevel: String, demoMode: Boolean, analysisNonce: Number })
const emit = defineEmits(['status'])
const analysis = ref(null)
const loading = ref(false)
const activeStep = ref(0)

let unsubscribe = null
const run = async () => {
  loading.value = true
  activeStep.value = 0
  const timer = window.setInterval(() => { activeStep.value = Math.min(4, activeStep.value + 1) }, 280)
  const [result] = await Promise.all([
    analyzeStock(props.stockName || 'NVDA', props.riskLevel || 'moderate'),
    new Promise((resolve) => window.setTimeout(resolve, 1400))
  ])
  window.clearInterval(timer)
  analysis.value = result
  loading.value = false
}

onMounted(() => {
  run()
  unsubscribe = subscribeRealtimeAlerts({
    onStatus: (status) => emit('status', status),
    onMessage: (alert) => {
      if (analysis.value?.alerts) analysis.value.alerts = [alert, ...analysis.value.alerts].slice(0, 6)
    }
  })
})
watch(() => [props.analysisNonce, props.demoMode], run)
</script>
