<template>
  <section>
    <div class="page-header">
      <div><h1 class="page-title">AI 风控报告</h1><div class="muted">读取后端 LLM / Qwen / FinanceLM 报告，失败时生成结构化报告。</div></div>
      <ReportExportActions v-if="report" @copy="copyReport" @markdown="downloadMarkdown" @json="downloadJson" />
    </div>
    <ReportGenerationState :show="loading" :active-index="activeStep" />
    <div v-if="report" class="dashboard-grid">
      <ReportHeader class="span-12" :report="report" />
      <LlmNarrativeCard class="span-8" :report="report" />
      <ReportMetricsPanel class="span-4" :report="report" />
      <ReportSectionList class="span-8" :sections="report.sections" />
      <ReportAlertSummary class="span-4" :alerts="report.alerts" />
    </div>
  </section>
</template>
<script setup>
import { onMounted, ref, watch } from 'vue'
import { generateRiskReport } from '../services/reportService'
import ReportHeader from '../components/report/ReportHeader.vue'
import LlmNarrativeCard from '../components/report/LlmNarrativeCard.vue'
import ReportSectionList from '../components/report/ReportSectionList.vue'
import ReportMetricsPanel from '../components/report/ReportMetricsPanel.vue'
import ReportAlertSummary from '../components/report/ReportAlertSummary.vue'
import ReportExportActions from '../components/report/ReportExportActions.vue'
import ReportGenerationState from '../components/report/ReportGenerationState.vue'
const props = defineProps({ stockName: String, riskLevel: String, demoMode: Boolean, analysisNonce: Number })
const report = ref(null); const loading = ref(false); const activeStep = ref(0)
const run = async () => {
  loading.value = true; activeStep.value = 0
  const timer = window.setInterval(() => { activeStep.value = Math.min(4, activeStep.value + 1) }, 260)
  const [result] = await Promise.all([generateRiskReport(props.stockName || 'NVDA', props.riskLevel || 'moderate'), new Promise((r) => setTimeout(r, 1200))])
  clearInterval(timer); report.value = result; loading.value = false
}
const markdown = () => `# ${report.value.summary.title}\n\n${report.value.llm.text}\n\n${report.value.sections.map((s) => `## ${s.title}\n${s.content.map((x) => `- ${x}`).join('\n')}`).join('\n\n')}`
const save = (name, text, type) => { const url = URL.createObjectURL(new Blob([text], { type })); const a = document.createElement('a'); a.href = url; a.download = name; a.click(); URL.revokeObjectURL(url) }
const copyReport = () => navigator.clipboard?.writeText(markdown())
const downloadMarkdown = () => save(`${report.value.stock.symbol}-risk-report.md`, markdown(), 'text/markdown')
const downloadJson = () => save(`${report.value.stock.symbol}-risk-report.json`, JSON.stringify(report.value, null, 2), 'application/json')
onMounted(run); watch(() => [props.analysisNonce, props.demoMode], run)
</script>
