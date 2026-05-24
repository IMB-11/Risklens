<template><AnalysisProgressOverlay :show="show" :active-index="activeIndex" /></template>
<script setup>
import AnalysisProgressOverlay from '../stock/AnalysisProgressOverlay.vue'
defineProps({ show: Boolean, activeIndex: Number })
</script>
