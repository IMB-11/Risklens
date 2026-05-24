<template><section><div class="page-header"><div><h1 class="page-title">舆情监控</h1><div class="muted">新闻摘要、社媒情绪与关键词风险。</div></div></div><div class="dashboard-grid"><SentimentPanel class="span-5" :sentiment="analysis.sentiment" /><NewsList class="span-7" :news="analysis.news" /></div></section></template>
<script setup>
import { onMounted, ref, watch } from 'vue'
import { demoStockAnalysis } from '../mock/stockMock'
import { analyzeStock } from '../services/stockAnalysisService'
import SentimentPanel from '../components/stock/SentimentPanel.vue'
import NewsList from '../components/stock/NewsList.vue'
const props = defineProps({ stockName: String, riskLevel: String, demoMode: Boolean, analysisNonce: Number })
const analysis = ref(demoStockAnalysis)
const run = async () => { analysis.value = await analyzeStock(props.stockName || 'NVDA', props.riskLevel || 'moderate') }
onMounted(run); watch(() => [props.analysisNonce, props.demoMode], run)
</script>
