<template><section><div class="page-header"><div><h1 class="page-title">模型预测</h1><div class="muted">FinanceLM 与可用多模型预测共识。</div></div></div><div class="dashboard-grid"><ModelConsensusCard class="span-4" :models="analysis.models" /><PriceForecastChart class="span-8" :prediction="analysis.prediction" /></div></section></template>
<script setup>
import { onMounted, ref, watch } from 'vue'
import { demoStockAnalysis } from '../mock/stockMock'
import { analyzeStock } from '../services/stockAnalysisService'
import ModelConsensusCard from '../components/stock/ModelConsensusCard.vue'
import PriceForecastChart from '../components/stock/PriceForecastChart.vue'
const props = defineProps({ stockName: String, riskLevel: String, demoMode: Boolean, analysisNonce: Number })
const analysis = ref(demoStockAnalysis)
const run = async () => { analysis.value = await analyzeStock(props.stockName || 'NVDA', props.riskLevel || 'moderate') }
onMounted(run); watch(() => [props.analysisNonce, props.demoMode], run)
</script>
