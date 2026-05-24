<template><div class="row-between" style="justify-content:flex-start;"><button class="btn secondary" @click="$emit('copy')">复制报告</button><button class="btn secondary" @click="$emit('markdown')">下载 Markdown</button><button class="btn secondary" @click="$emit('json')">下载 JSON</button></div></template>
<script setup>defineEmits(['copy', 'markdown', 'json'])</script>
