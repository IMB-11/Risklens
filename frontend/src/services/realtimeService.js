import { createRealtimeSocket } from '../api/realtimeApi'
import { demoAlerts } from '../mock/alertsMock'
import { isDemoModeEnabled } from './configService'

export const subscribeRealtimeAlerts = ({ onMessage, onStatus } = {}) => {
  if (isDemoModeEnabled()) {
    onStatus?.('demo')
    const timer = window.setInterval(() => onMessage?.(demoAlerts[Math.floor(Math.random() * demoAlerts.length)]), 8000)
    return () => window.clearInterval(timer)
  }

  try {
    const socket = createRealtimeSocket()
    socket.onopen = () => onStatus?.('connected')
    socket.onmessage = (event) => {
      try {
        onMessage?.(JSON.parse(event.data))
      } catch {
        onMessage?.({ level: 'low', title: '实时消息', time: '刚刚', description: event.data })
      }
    }
    socket.onerror = () => onStatus?.('degraded')
    socket.onclose = () => onStatus?.('closed')
    return () => socket.close()
  } catch {
    onStatus?.('degraded')
    return () => {}
  }
}
