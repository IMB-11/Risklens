<template>
  <article class="card">
    <div class="card-header"><h3 class="card-title">AI 投资建议</h3><span class="badge">FinanceLM</span></div>
    <div class="metric-value">{{ recommendation.action }}</div>
    <p class="muted">{{ recommendation.positionAdvice }}</p>
    <div class="kpi"><span class="soft">置信度</span><strong>{{ Math.round(recommendation.confidence * 100) }}%</strong></div>
    <ul class="muted"><li v-for="reason in recommendation.reasons" :key="reason">{{ reason }}</li></ul>
  </article>
</template>
<script setup>defineProps({ recommendation: { type: Object, required: true } })</script>
