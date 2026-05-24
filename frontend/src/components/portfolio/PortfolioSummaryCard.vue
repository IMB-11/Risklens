<template><MetricCard title="组合总风险" :value="summary.score" :subtitle="summary.text" :badge="summary.label" :tone="summary.level" /></template>
<script setup>
import MetricCard from '../common/MetricCard.vue'
defineProps({ summary: { type: Object, required: true } })
</script>
