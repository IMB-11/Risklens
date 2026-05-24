import { apiClient, unwrap } from './http'

export const getConfigStatusApi = () => apiClient.get('/config/status').then(unwrap)

export const saveConfigKeysApi = (payload) => apiClient.post('/config/keys', payload).then(unwrap)
