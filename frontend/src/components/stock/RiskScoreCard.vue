<template>
  <article class="card">
    <div class="card-header"><h3 class="card-title">Risk Score</h3><StatusBadge :label="risk.label" :tone="risk.level" /></div>
    <div class="metric-value" :class="`risk-${risk.level}`">{{ risk.score }}</div>
    <div class="metric-label">综合风险评分 / 100</div>
    <p class="muted">{{ risk.summary }}</p>
  </article>
</template>
<script setup>
import StatusBadge from '../common/StatusBadge.vue'
defineProps({ risk: { type: Object, required: true } })
</script>
