<template><article class="card"><div class="card-header"><h3 class="card-title">AI 调仓建议</h3><span class="badge">Portfolio AI</span></div><div class="metric-value">{{ recommendation.action }}</div><ul class="muted"><li v-for="item in recommendation.items" :key="item">{{ item }}</li></ul></article></template>
<script setup>defineProps({ recommendation: { type: Object, required: true } })</script>
