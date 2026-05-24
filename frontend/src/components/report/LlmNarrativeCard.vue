<template><article class="card"><div class="card-header"><h3 class="card-title">LLM 风控报告正文</h3><span class="badge">{{ report.llm.provider }}</span></div><p class="muted" style="white-space: pre-line; line-height: 1.8;">{{ report.llm.text }}</p></article></template>
<script setup>defineProps({ report: { type: Object, required: true } })</script>
