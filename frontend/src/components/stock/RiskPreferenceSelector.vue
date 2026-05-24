<template><select class="select" :value="modelValue" @change="$emit('update:modelValue', $event.target.value)"><option value="conservative">保守型</option><option value="moderate">平衡型</option><option value="aggressive">进取型</option></select></template>
<script setup>defineProps({ modelValue: String }); defineEmits(['update:modelValue'])</script>
