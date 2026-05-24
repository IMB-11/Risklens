"""Risk data provider: fetch real market data from akshare for risk metrics."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import akshare as ak
import math


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# Stock code mapping (same as in api/main.py)
STOCK_CODE_MAPPING = {
    '宁德时代': '300750', '贵州茅台': '600519', '比亚迪': '002594',
    '腾讯控股': '00700', '阿里巴巴': 'BABA', '招商银行': '600036',
    '万科': '000002', '格力电器': '000651', '五粮液': '000858',
    '美团': '03690', '京东': 'JD', '网易': 'NTES',
    '海康威视': '002415', '药明康德': '603259', '隆基绿能': '601012',
}


class RiskDataProvider:
    """Fetch real market and financial data for risk metrics via akshare."""

    def __init__(self) -> None:
        print("[OK] RiskDataProvider initialized")

    def get_stock_code(self, stock_name: str) -> str:
        return STOCK_CODE_MAPPING.get(stock_name, stock_name)

    def _to_tencent_symbol(self, code: str) -> str:
        """Convert 6-digit code to Tencent format (e.g., '600519' -> 'sh600519')."""
        if code.startswith('6'):
            return f'sh{code}'
        else:
            return f'sz{code}'

    def fetch_kline(
        self, stock_name: str, days: int = 120
    ) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
        """Fetch real OHLCV + returns from akshare.

        Tries Tencent source first (more reliable), then 东方财富 as fallback.
        Returns:
            (closes, highs, lows, volumes, returns, dates)
            All lists have the same length. returns[0] = 0.0 (no prior day).
        """
        code = self.get_stock_code(stock_name)
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        df = None

        # Try 东方财富 first (has volume column, more complete data)
        try:
            df = ak.stock_zh_a_hist(symbol=code, period='daily', start_date=start_date, end_date=end_date, adjust="qfq")
            if df is not None and not df.empty:
                print(f"[OK] fetch_kline via 东方财富 for {stock_name}")
        except Exception as e:
            print(f"[WARN] 东方财富 source failed for {stock_name}: {e}")

        # Fallback to Tencent source (may lack volume column)
        if df is None or df.empty:
            try:
                tc_symbol = self._to_tencent_symbol(code)
                df = ak.stock_zh_a_hist_tx(symbol=tc_symbol, start_date=start_date, end_date=end_date, adjust="qfq")
                if df is not None and not df.empty:
                    print(f"[OK] fetch_kline via Tencent for {stock_name}")
            except Exception as e:
                print(f"[WARN] Tencent source also failed for {stock_name}: {e}")
                return [], [], [], [], [], []

        if df is None or df.empty:
            return [], [], [], [], [], []

        # Normalize column names
        col_map = {}
        for col in df.columns:
            c = str(col).lower()
            if '日期' in c or 'date' in c:
                col_map[col] = 'date'
            elif '开盘' in c or 'open' in c:
                col_map[col] = 'open'
            elif '最高' in c or 'high' in c:
                col_map[col] = 'high'
            elif '最低' in c or 'low' in c:
                col_map[col] = 'low'
            elif '收盘' in c or 'close' in c:
                col_map[col] = 'close'
            elif '成交量' in c or 'volume' in c:
                col_map[col] = 'volume'
            elif c == 'amount':
                # Tencent source uses 'amount' for volume (in lots)
                col_map[col] = 'volume'
        df = df.rename(columns=col_map)

        required = {'close', 'high', 'low', 'volume'}
        if not required.issubset(set(df.columns)):
            print(f"[WARN] Missing columns: {required - set(df.columns)}")
            return [], [], [], [], [], []

        closes = df['close'].astype(float).tolist()
        highs = df['high'].astype(float).tolist()
        lows = df['low'].astype(float).tolist()
        volumes = df['volume'].astype(float).tolist()
        dates = df['date'].astype(str).tolist() if 'date' in df.columns else [''] * len(closes)

        # Compute daily returns
        returns = [0.0]
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                returns.append((closes[i] - closes[i - 1]) / closes[i - 1])
            else:
                returns.append(0.0)

        return closes, highs, lows, volumes, returns, dates

    def fetch_financial_data(self, stock_name: str) -> Dict[str, Any]:
        """Fetch financial data for Z-score and credit risk computation.

        Uses Sina balance sheet + income statement for accurate totals,
        falls back to analysis indicators for ratios.
        Returns dict with keys: total_assets, total_liabilities, working_capital,
        retained_earnings, ebit, current_assets, current_liabilities, revenue,
        net_profit, debt_ratio, current_ratio, operating_margin.
        """
        code = self.get_stock_code(stock_name)
        result: Dict[str, Any] = {}

        # 1) Sina balance sheet — accurate total values
        try:
            bs = ak.stock_financial_report_sina(stock=code, symbol='资产负债表')
            if bs is not None and not bs.empty:
                row = bs.iloc[0]
                # Banks use '资产总计', non-banks use '负债和所有者权益(或股东权益)总计'
                ta = _to_float(row.get('负债和所有者权益(或股东权益)总计'), 0.0)
                if ta <= 0:
                    ta = _to_float(row.get('资产总计'), 0.0)
                result['total_assets'] = ta
                result['total_liabilities'] = _to_float(row.get('负债合计'), 0.0)
                ca = _to_float(row.get('流动资产合计'), 0.0)
                cl = _to_float(row.get('流动负债合计'), 0.0)
                if ca > 0 and cl > 0:
                    result['current_assets'] = ca
                    result['current_liabilities'] = cl
                    result['working_capital'] = ca - cl
                    result['liquid_assets'] = ca
                result['retained_earnings'] = _to_float(row.get('未分配利润'), 0.0)
                if result['total_assets'] > 0:
                    result['debt_ratio'] = result['total_liabilities'] / result['total_assets']
                print(f"[OK] fetch_financial_data (Sina BS) for {stock_name}: TA={result['total_assets']/1e8:.0f}亿, RE={result['retained_earnings']/1e8:.0f}亿")
        except Exception as e:
            print(f"[WARN] Sina balance sheet failed for {stock_name}: {e}")

        # 2) Sina income statement — EBIT and revenue
        try:
            inc = ak.stock_financial_report_sina(stock=code, symbol='利润表')
            if inc is not None and not inc.empty:
                row = inc.iloc[0]
                result['revenue'] = _to_float(row.get('营业总收入'), 0.0)
                result['ebit'] = _to_float(row.get('营业利润'), 0.0)
                result['net_profit'] = _to_float(row.get('净利润'), 0.0)
                if result['revenue'] > 0:
                    result['operating_margin'] = result['ebit'] / result['revenue']
                print(f"[OK] fetch_financial_data (Sina IS) for {stock_name}: EBIT={result['ebit']/1e8:.0f}亿, Rev={result['revenue']/1e8:.0f}亿")
        except Exception as e:
            print(f"[WARN] Sina income statement failed for {stock_name}: {e}")

        # 3) Fallback: analysis indicators for ratios (current_ratio, etc.)
        if 'current_ratio' not in result:
            try:
                df = ak.stock_financial_analysis_indicator(symbol=code, start_year='2024')
                if df is not None and not df.empty:
                    latest = df.iloc[0]
                    for col in df.columns:
                        c = str(col)
                        val = _to_float(latest[col], 0.0)
                        if '流动比率' in c and 'current_ratio' not in result:
                            result['current_ratio'] = val
                        elif '速动比率' in c and 'quick_ratio' not in result:
                            result['quick_ratio'] = val
                        elif '总资产周转率' in c and 'asset_turnover' not in result:
                            result['asset_turnover'] = val
                        elif c == '总资产(元)' and 'total_assets' not in result:
                            result['total_assets'] = val
                        elif c == '资产负债率(%)' and 'debt_ratio' not in result:
                            result['debt_ratio'] = val / 100.0
            except Exception as e:
                print(f"[WARN] Analysis indicator fallback failed for {stock_name}: {e}")

        return result

    def fetch_market_data(self, stock_name: str, financial_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fetch market-level data for credit risk metrics."""
        closes, highs, lows, volumes, returns, dates = self.fetch_kline(stock_name, days=120)
        equity_vol = 0.0
        if len(returns) > 5:
            import statistics
            equity_vol = statistics.stdev(returns)

        # Try to get market cap from spot data
        market_cap = 0.0
        try:
            code = self.get_stock_code(stock_name)
            if len(code) == 6:
                df = ak.stock_zh_a_spot_em()
                if df is not None and not df.empty:
                    row = df[df['代码'] == code]
                    if not row.empty:
                        market_cap = _to_float(row['总市值'].iloc[0], 0.0)
        except Exception as e:
            print(f"[WARN] fetch_market_data spot query failed: {e}")

        # Fallback: estimate market cap from book value * P/B ratio
        if market_cap <= 0 and financial_data:
            ta = _to_float(financial_data.get('total_assets'), 0.0)
            tl = _to_float(financial_data.get('total_liabilities'), 0.0)
            if ta > 0 and tl > 0:
                book_equity = ta - tl
                # P/B heuristic by industry
                pb_ratio = 2.0  # default
                if stock_name in ('招商银行',):
                    pb_ratio = 0.7  # banks
                elif stock_name in ('贵州茅台',):
                    pb_ratio = 6.0  # premium consumer
                elif stock_name in ('宁德时代', '比亚迪'):
                    pb_ratio = 3.0  # growth
                market_cap = book_equity * pb_ratio
                print(f"[OK] Estimated market cap for {stock_name}: {market_cap/1e8:.0f}亿 (book={book_equity/1e8:.0f}亿, P/B={pb_ratio}x)")

        # Fetch index data as market benchmark
        index_returns = []
        try:
            idx_df = ak.stock_zh_index_daily(symbol='sh000001')
            if idx_df is not None and not idx_df.empty and 'close' in idx_df.columns:
                idx_closes = idx_df['close'].astype(float).tolist()
                idx_closes = idx_closes[-min(120, len(idx_closes)):]
                for i in range(1, len(idx_closes)):
                    if idx_closes[i-1] > 0:
                        index_returns.append((idx_closes[i] - idx_closes[i-1]) / idx_closes[i-1])
                print(f"[OK] Fetched Shanghai index data: {len(index_returns)} returns")
        except Exception as e:
            print(f"[WARN] Index data fetch failed: {e}")

        return {
            'stock_returns': returns,
            'equity_volatility': equity_vol,
            'market_cap': market_cap,
            'prices': closes,
            'volumes': volumes,
            'index_returns': index_returns,
        }


# Singleton
risk_data_provider = RiskDataProvider()
