export const createRealtimeSocket = () => {
  const base = import.meta.env.VITE_WS_BASE_URL || '/ws'
  const url = base.startsWith('ws')
    ? `${base}/realtime`
    : `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}${base}/realtime`
  return new WebSocket(url)
}
