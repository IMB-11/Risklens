<template><article class="card"><div class="card-header"><h3 class="card-title">未来 7 天价格预测</h3><span class="badge">Confidence Band</span></div><div ref="chartEl" class="chart"></div></article></template>
<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as echarts from 'echarts'
const props = defineProps({ prediction: { type: Object, required: true } })
const chartEl = ref(null)
let chart
const render = () => {
  if (!chartEl.value) return
  chart ||= echarts.init(chartEl.value)
  chart.setOption({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis' },
    grid: { left: 38, right: 18, top: 28, bottom: 30 },
    xAxis: { type: 'category', data: props.prediction.days, axisLine: { lineStyle: { color: '#334155' } }, axisLabel: { color: '#94a3b8' } },
    yAxis: { type: 'value', axisLabel: { color: '#94a3b8' }, splitLine: { lineStyle: { color: 'rgba(148,163,184,.12)' } } },
    series: [
      { name: '下界', type: 'line', data: props.prediction.lowerBound, lineStyle: { opacity: 0 }, stack: 'band', symbol: 'none' },
      { name: '区间', type: 'line', data: props.prediction.upperBound.map((v, i) => v - props.prediction.lowerBound[i]), areaStyle: { color: 'rgba(56,189,248,.16)' }, lineStyle: { opacity: 0 }, stack: 'band', symbol: 'none' },
      { name: '预测价格', type: 'line', smooth: true, data: props.prediction.prices, symbolSize: 7, lineStyle: { color: '#38bdf8', width: 3 }, itemStyle: { color: '#38bdf8' } }
    ]
  })
}
onMounted(() => { render(); window.addEventListener('resize', () => chart?.resize()) })
watch(() => props.prediction, render, { deep: true })
onBeforeUnmount(() => chart?.dispose())
</script>
