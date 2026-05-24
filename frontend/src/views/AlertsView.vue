<template><section><div class="page-header"><div><h1 class="page-title">实时告警</h1><div class="muted">WebSocket 风险推送与本地演示告警。</div></div></div><AlertTimeline :alerts="alerts" /></section></template>
<script setup>
import { onMounted, ref } from 'vue'
import AlertTimeline from '../components/stock/AlertTimeline.vue'
import { demoAlerts } from '../mock/alertsMock'
import { subscribeRealtimeAlerts } from '../services/realtimeService'
defineProps({ demoMode: Boolean })
const emit = defineEmits(['status'])
const alerts = ref(demoAlerts)
onMounted(() => subscribeRealtimeAlerts({ onStatus: (s) => emit('status', s), onMessage: (a) => { alerts.value = [a, ...alerts.value].slice(0, 10) } }))
</script>
