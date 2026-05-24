<template>
  <article class="card">
    <div class="card-header"><h3 class="card-title">舆情分析</h3><span class="badge">{{ sentiment.overall }}</span></div>
    <div class="kpi-grid">
      <div class="kpi"><span class="soft">Positive</span><strong class="positive">{{ sentiment.positive }}%</strong></div>
      <div class="kpi"><span class="soft">Neutral</span><strong>{{ sentiment.neutral }}%</strong></div>
      <div class="kpi"><span class="soft">Negative</span><strong class="negative">{{ sentiment.negative }}%</strong></div>
      <div class="kpi"><span class="soft">关键词</span><strong>{{ sentiment.keywords?.slice(0, 2).join(' / ') }}</strong></div>
    </div>
    <p class="muted">{{ sentiment.summary }}</p>
  </article>
</template>
<script setup>defineProps({ sentiment: { type: Object, required: true } })</script>
