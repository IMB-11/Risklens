<template>
  <button class="btn secondary" type="button" @click="$emit('update:modelValue', !modelValue)">
    <ToggleRight v-if="modelValue" :size="18" />
    <ToggleLeft v-else :size="18" />
    {{ modelValue ? '演示模式' : '真实接口' }}
  </button>
</template>

<script setup>
import { ToggleLeft, ToggleRight } from 'lucide-vue-next'
defineProps({ modelValue: Boolean })
defineEmits(['update:modelValue'])
</script>
