<template><div class="card"><h3 class="card-title">演示模式已接管</h3><p class="muted">{{ message || '真实接口暂不可用，当前页面仍可继续演示。' }}</p></div></template>
<script setup>defineProps({ message: String })</script>
