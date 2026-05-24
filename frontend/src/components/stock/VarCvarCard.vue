<template>
  <article class="card">
    <div class="card-header"><h3 class="card-title">VaR / CVaR</h3><span class="badge" :class="risk.level">95%</span></div>
    <div class="kpi-grid">
      <div class="kpi"><span class="soft">VaR 95%</span><strong>{{ risk.var95 }}%</strong></div>
      <div class="kpi"><span class="soft">CVaR 95%</span><strong>{{ risk.cvar95 }}%</strong></div>
      <div class="kpi"><span class="soft">波动率</span><strong>{{ risk.volatility }}%</strong></div>
      <div class="kpi"><span class="soft">最大回撤</span><strong>{{ risk.maxDrawdown }}%</strong></div>
    </div>
  </article>
</template>
<script setup>defineProps({ risk: { type: Object, required: true } })</script>
