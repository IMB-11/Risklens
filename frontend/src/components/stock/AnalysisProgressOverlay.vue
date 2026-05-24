<template>
  <div v-if="show" class="overlay">
    <article class="card progress-card">
      <div class="card-header"><h3 class="card-title">运行 AI 风控分析</h3><span class="badge">FinanceLM</span></div>
      <div v-for="(step, index) in steps" :key="step" class="progress-step" :class="{ active: index <= activeIndex }">
        <span class="status-dot"></span>{{ step }}
      </div>
    </article>
  </div>
</template>
<script setup>
defineProps({ show: Boolean, activeIndex: Number })
const steps = ['获取行情与新闻', '分析舆情情绪', '运行 FinanceLM / 多模型预测', '计算 VaR / CVaR 与风险因子', '生成 AI 投资建议']
</script>
