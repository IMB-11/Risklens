<template>
  <article class="card">
    <div class="card-header"><h3 class="card-title">{{ stock.symbol }}</h3><span class="badge">{{ stock.market }}</span></div>
    <div class="metric-value">{{ stock.price?.toFixed ? stock.price.toFixed(2) : stock.price }}</div>
    <p class="muted">{{ stock.name }} · {{ stock.sector }}</p>
    <div class="row-between"><span class="muted">今日涨跌</span><strong :class="stock.changePercent >= 0 ? 'positive' : 'negative'">{{ stock.changePercent }}%</strong></div>
    <div class="soft" style="margin-top: 10px;">更新：{{ stock.updatedAt }}</div>
  </article>
</template>
<script setup>defineProps({ stock: { type: Object, required: true } })</script>
