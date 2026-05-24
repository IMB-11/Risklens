<template>
  <article class="card">
    <div class="card-header"><h3 class="card-title">组合持仓编辑</h3><button class="btn secondary" type="button" @click="add">添加资产</button></div>
    <div class="list">
      <div v-for="(asset, index) in localAssets" :key="index" class="row-between">
        <input class="field" style="max-width: 160px;" v-model="asset.symbol" />
        <input class="field" style="max-width: 120px;" type="number" v-model.number="asset.weight" />
        <button class="btn secondary" type="button" @click="remove(index)">删除</button>
      </div>
    </div>
    <button class="btn" style="margin-top: 14px;" type="button" @click="$emit('analyze', localAssets)">分析组合</button>
  </article>
</template>
<script setup>
import { ref, watch } from 'vue'
const props = defineProps({ assets: { type: Array, required: true } })
defineEmits(['analyze'])
const localAssets = ref(props.assets.map((item) => ({ ...item })))
watch(() => props.assets, (value) => { localAssets.value = value.map((item) => ({ ...item })) })
const add = () => localAssets.value.push({ symbol: 'AMD', weight: 10, sector: 'Technology' })
const remove = (index) => localAssets.value.splice(index, 1)
</script>
