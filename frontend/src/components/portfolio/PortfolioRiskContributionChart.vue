<template><article class="card"><div class="card-header"><h3 class="card-title">个股风险贡献</h3></div><div ref="chartEl" class="chart small"></div></article></template>
<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as echarts from 'echarts'
const props = defineProps({ contributions: { type: Array, required: true } })
const chartEl = ref(null); let chart
const render = () => {
  chart ||= echarts.init(chartEl.value)
  chart.setOption({ grid: { left: 58, right: 18, top: 8, bottom: 18 }, xAxis: { type: 'value', axisLabel: { color: '#94a3b8' }, splitLine: { lineStyle: { color: 'rgba(148,163,184,.12)' } } }, yAxis: { type: 'category', data: props.contributions.map((x) => x.name).reverse(), axisLabel: { color: '#e5e7eb' } }, series: [{ type: 'bar', data: props.contributions.map((x) => x.value).reverse(), barWidth: 14, itemStyle: { color: '#38bdf8', borderRadius: 8 } }] })
}
onMounted(render); watch(() => props.contributions, render, { deep: true }); onBeforeUnmount(() => chart?.dispose())
</script>
