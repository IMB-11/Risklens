import { createRouter, createWebHistory } from 'vue-router'
import SingleStockView from '../views/SingleStockView.vue'
import PortfolioView from '../views/PortfolioView.vue'
import AlertsView from '../views/AlertsView.vue'
import SentimentView from '../views/SentimentView.vue'
import ModelView from '../views/ModelView.vue'
import RiskReportView from '../views/RiskReportView.vue'
import SystemStatusView from '../views/SystemStatusView.vue'

const routes = [
  { path: '/', redirect: '/single-stock' },
  { path: '/single-stock', name: 'single-stock', component: SingleStockView },
  { path: '/portfolio', name: 'portfolio', component: PortfolioView },
  { path: '/alerts', name: 'alerts', component: AlertsView },
  { path: '/sentiment', name: 'sentiment', component: SentimentView },
  { path: '/models', name: 'models', component: ModelView },
  { path: '/risk-report', name: 'risk-report', component: RiskReportView },
  { path: '/system-status', name: 'system-status', component: SystemStatusView }
]

export default createRouter({
  history: createWebHistory(),
  routes
})
