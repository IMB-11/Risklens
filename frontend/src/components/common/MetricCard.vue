<template>
  <article class="card">
    <div class="card-header"><h3 class="card-title">{{ title }}</h3><StatusBadge v-if="badge" :label="badge" :tone="tone" /></div>
    <div class="metric-value" :class="tone ? `risk-${tone}` : ''">{{ value }}</div>
    <div class="metric-label">{{ subtitle }}</div>
  </article>
</template>
<script setup>
import StatusBadge from './StatusBadge.vue'
defineProps({ title: String, value: [String, Number], subtitle: String, badge: String, tone: String })
</script>
