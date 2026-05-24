<template>
  <header class="topbar">
    <input class="field" :value="stockName" placeholder="输入股票代码或名称" @input="$emit('update:stockName', $event.target.value)" @keyup.enter="$emit('analyze')" />
    <select class="select" :value="riskLevel" @change="$emit('update:riskLevel', $event.target.value)">
      <option value="conservative">保守型</option>
      <option value="moderate">平衡型</option>
      <option value="aggressive">进取型</option>
    </select>
    <button class="btn" type="button" @click="$emit('analyze')"><Search :size="18" /> Analyze</button>
    <div class="top-status"><span class="status-dot"></span>{{ connectionLabel }}</div>
    <DemoModeToggle :model-value="demoMode" @update:model-value="$emit('update:demoMode', $event)" />
  </header>
</template>

<script setup>
import { computed } from 'vue'
import { Search } from 'lucide-vue-next'
import DemoModeToggle from '../common/DemoModeToggle.vue'

const props = defineProps({
  stockName: String,
  riskLevel: String,
  demoMode: Boolean,
  connectionStatus: String
})
defineEmits(['update:stockName', 'update:riskLevel', 'update:demoMode', 'analyze'])

const connectionLabel = computed(() => props.connectionStatus === 'connected' ? '实时连接正常' : props.demoMode ? '演示模式稳定' : '连接检测中')
</script>
