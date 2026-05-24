<template><article class="card"><div class="card-header"><h3 class="card-title">风险因子拆解</h3></div><div ref="chartEl" class="chart small"></div></article></template>
<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as echarts from 'echarts'
const props = defineProps({ factors: { type: Array, required: true } })
const chartEl = ref(null)
let chart
const render = () => {
  chart ||= echarts.init(chartEl.value)
  chart.setOption({
    grid: { left: 72, right: 22, top: 8, bottom: 18 },
    xAxis: { type: 'value', max: 100, axisLabel: { color: '#94a3b8' }, splitLine: { lineStyle: { color: 'rgba(148,163,184,.12)' } } },
    yAxis: { type: 'category', data: props.factors.map((f) => f.name).reverse(), axisLabel: { color: '#e5e7eb' } },
    series: [{ type: 'bar', data: props.factors.map((f) => f.value).reverse(), barWidth: 14, itemStyle: { borderRadius: 8, color: '#a78bfa' } }]
  })
}
onMounted(render); watch(() => props.factors, render, { deep: true }); onBeforeUnmount(() => chart?.dispose())
</script>
