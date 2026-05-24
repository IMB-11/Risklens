<template><span class="badge" :class="tone">{{ label }}</span></template>
<script setup>
defineProps({ label: { type: String, default: '正常' }, tone: { type: String, default: '' } })
</script>
