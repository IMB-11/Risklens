<template>
  <aside class="sidebar">
    <div class="brand"><span class="brand-mark"><Activity :size="18" /></span><span>AI Risk Terminal</span></div>
    <nav class="nav-list">
      <RouterLink v-for="item in items" :key="item.to" class="nav-item" :to="item.to">
        <component :is="item.icon" :size="18" />
        <span>{{ item.label }}</span>
      </RouterLink>
    </nav>
  </aside>
</template>

<script setup>
import { Activity, Bell, BarChart3, FileText, Gauge, LineChart, RadioTower, Settings } from 'lucide-vue-next'

const items = [
  { to: '/single-stock', label: '单股票分析', icon: Gauge },
  { to: '/portfolio', label: '组合风控', icon: BarChart3 },
  { to: '/alerts', label: '实时告警', icon: Bell },
  { to: '/sentiment', label: '舆情监控', icon: RadioTower },
  { to: '/models', label: '模型预测', icon: LineChart },
  { to: '/risk-report', label: 'AI 风控报告', icon: FileText },
  { to: '/system-status', label: '系统状态', icon: Settings }
]
</script>
