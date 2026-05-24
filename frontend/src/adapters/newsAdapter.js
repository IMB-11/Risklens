export const adaptNewsItem = (item) => ({
  title: item.title || item.headline || '市场新闻',
  source: item.source || 'External News',
  time: item.time || item.publish_time || item.published_at || '刚刚',
  sentiment: item.sentiment || item.sentiment_type || 'neutral',
  impact: Number(item.weight || 0) > 0.7 ? 'high' : 'medium'
})
