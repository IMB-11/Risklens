<template>
  <div class="app-shell">
    <Sidebar />
    <section class="main-pane">
      <Topbar
        v-model:stock-name="state.stockName"
        v-model:risk-level="state.riskLevel"
        v-model:demo-mode="state.demoMode"
        :connection-status="state.connectionStatus"
        @analyze="runAnalyze"
      />
      <main class="content">
        <RouterView v-slot="{ Component }">
          <component
            :is="Component"
            :stock-name="state.stockName"
            :risk-level="state.riskLevel"
            :demo-mode="state.demoMode"
            :analysis-nonce="state.analysisNonce"
            @status="state.connectionStatus = $event"
          />
        </RouterView>
      </main>
    </section>
  </div>
</template>

<script setup>
import { reactive, watch } from 'vue'
import Sidebar from './Sidebar.vue'
import Topbar from './Topbar.vue'
import { isDemoModeEnabled, setDemoMode } from '../../services/configService'

const state = reactive({
  stockName: 'NVDA',
  riskLevel: 'moderate',
  demoMode: isDemoModeEnabled(),
  connectionStatus: 'checking',
  analysisNonce: 0
})

watch(() => state.demoMode, (value) => setDemoMode(value))

const runAnalyze = () => {
  state.stockName = (state.stockName || 'NVDA').trim().toUpperCase()
  state.analysisNonce += 1
}
</script>
