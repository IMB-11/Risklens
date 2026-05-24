<template><div class="card"><h3 class="card-title">{{ title || '暂无数据' }}</h3><p class="muted">{{ description || '当前没有可展示的数据。' }}</p></div></template>
<script setup>defineProps({ title: String, description: String })</script>
