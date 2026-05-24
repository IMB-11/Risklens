# China Alternative Data Layer (No-Training)

This module provides a no-training path for domestic alternative data:

1. Real-time collection from registered domestic platforms
2. Historical window replay from local cache snapshots
3. Algorithmic fusion into:
   - `event_impacts` (risk event chain input)
   - `macro_factors` (market-regime macro input)

## Covered platform catalog

- `cninfo`
- `sse`
- `szse`
- `pbc`
- `nbs`
- `safe`
- `customs`
- `cma`
- `cnemc`
- `baidu_index`
- `weibo_index`
- `chinamoney`
- `shibor`
- `baidu_hot` (互联网热搜热度)
- `eastmoney_hot_rank` (个股人气/热度)
- `eastmoney_fund_flow` (资金流向)
- `eastmoney_margin` (融资融券)

Some platforms may require login or captcha. The engine keeps them in catalog and reports source status.

## API integration

Main risk pipeline (`backend/api/main.py`):

1. Collects standard multi-source news
2. Collects domestic alternative data (realtime + cached history)
3. Merges alt events into `inference_result.event_impacts`
4. Merges alt macro factors into `multi_model_result.macro_factors`
5. Runs `risk_control_engine.assess(...)`

## API endpoint

`POST /api/alt-data/collect`

Request body:

```json
{
  "stock_name": "贵州茅台",
  "lookback_days": 30,
  "limit_per_source": 8,
  "include_realtime": true,
  "include_history": true
}
```

## Quick run (API)

```bash
curl -X POST http://127.0.0.1:8000/api/alt-data/collect \
  -H "Content-Type: application/json" \
  -d "{\"stock_name\":\"贵州茅台\",\"lookback_days\":30,\"limit_per_source\":8,\"include_realtime\":true,\"include_history\":true}"
```

## Anti-crawl safeguards

The collector includes:

1. Domain-level pacing (`base interval + jitter`)
2. User-Agent rotation per request
3. Exponential backoff retries
4. `403` / `429` handling with dynamic domain penalty
5. Bounded async concurrency to avoid burst traffic

Runtime anti-crawl diagnostics are returned in:

- `anti_crawl.request_stats`
- `anti_crawl.active_domain_penalty`
- `anti_crawl.source_refresh_seconds`

## Low-frequency fallback policy

For harder endpoints (auth/captcha or frequent rate limits), the collector now:

1. Uses **source-level refresh intervals** (e.g., 10-30 minutes for flow/heat, longer for auth sources)
2. Returns **cached source snapshots** if live crawl is not due yet
3. Falls back to **stale source cache** if live fetch fails

`source_status.status` may return:

- `ok`
- `cached`
- `cached_stale`
- `auth_required`
- `error`
