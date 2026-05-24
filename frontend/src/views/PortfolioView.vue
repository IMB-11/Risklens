<template>
  <section>
    <div class="page-header"><div><h1 class="page-title">组合风控</h1><div class="muted">编辑持仓权重，评估组合级尾部风险与调仓建议。</div></div></div>
    <div class="dashboard-grid" v-if="portfolio">
      <PortfolioEditor class="span-4" :assets="assets" @analyze="run" />
      <PortfolioSummaryCard class="span-4" :summary="portfolio.summary" />
      <PortfolioVarCvarCard class="span-4" :summary="portfolio.summary" />
      <PortfolioRiskContributionChart class="span-7" :contributions="portfolio.contributions" />
      <HoldingsTable class="span-5" :holdings="portfolio.holdings" />
      <PortfolioRecommendationCard class="span-12" :recommendation="portfolio.recommendation" />
    </div>
  </section>
</template>
<script setup>
import { onMounted, ref, watch } from 'vue'
import { analyzePortfolio } from '../services/portfolioService'
import { defaultPortfolioAssets } from '../mock/portfolioMock'
import PortfolioEditor from '../components/portfolio/PortfolioEditor.vue'
import PortfolioSummaryCard from '../components/portfolio/PortfolioSummaryCard.vue'
import HoldingsTable from '../components/portfolio/HoldingsTable.vue'
import PortfolioRiskContributionChart from '../components/portfolio/PortfolioRiskContributionChart.vue'
import PortfolioRecommendationCard from '../components/portfolio/PortfolioRecommendationCard.vue'
import PortfolioVarCvarCard from '../components/portfolio/PortfolioVarCvarCard.vue'
const props = defineProps({ riskLevel: String, demoMode: Boolean, analysisNonce: Number })
const assets = ref(defaultPortfolioAssets)
const portfolio = ref(null)
const run = async (nextAssets = assets.value) => { assets.value = nextAssets; portfolio.value = await analyzePortfolio(assets.value, props.riskLevel) }
onMounted(run)
watch(() => [props.analysisNonce, props.demoMode], () => run())
</script>
