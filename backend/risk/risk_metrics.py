"""Enhanced risk metrics: tail risk, liquidity risk, credit risk, concentration risk."""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional, Tuple


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_mean(values: List[float], default: float = 0.0) -> float:
    return statistics.mean(values) if values else default


def _safe_stdev(values: List[float], default: float = 0.0) -> float:
    return statistics.stdev(values) if len(values) > 1 else default


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    q = _clip(q, 0.0, 1.0)
    idx = (len(sorted_vals) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = idx - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


# ---------------------------------------------------------------------------
# 1. Tail Risk Metrics
# ---------------------------------------------------------------------------

class TailRiskMetrics:
    """Tail risk indicators: ES, multi-timeframe CVaR, extreme loss probability."""

    def compute(self, returns: List[float], alpha: float = 0.95) -> Dict[str, Any]:
        clean = [r for r in returns if math.isfinite(r)]
        if not clean:
            return self._empty()

        losses = [-r for r in clean]

        # VaR at given alpha
        var_val = max(0.0, _quantile(losses, alpha))

        # ES (Expected Shortfall) = mean of losses beyond VaR
        tail_losses = [x for x in losses if x >= var_val]
        es_val = max(var_val, _safe_mean(tail_losses, var_val))

        # Multi-timeframe CVaR
        cvar_1d = self._cvar(losses, alpha)
        weekly_returns = self._aggregate_returns(clean, 5)
        monthly_returns = self._aggregate_returns(clean, 22)
        cvar_1w = self._cvar([-r for r in weekly_returns], alpha) if weekly_returns else cvar_1d
        cvar_1m = self._cvar([-r for r in monthly_returns], alpha) if monthly_returns else cvar_1w

        # Extreme loss probability: P(L > 3*sigma)
        mu_loss = _safe_mean(losses)
        sigma_loss = _safe_stdev(losses, 0.001)
        if sigma_loss > 0:
            z_3sigma = (3 * sigma_loss - mu_loss) / sigma_loss
            extreme_prob = max(0.0, 1.0 - self._normal_cdf(z_3sigma))
        else:
            extreme_prob = 0.0

        # Count actual 3-sigma losses in sample
        threshold_3sigma = mu_loss + 3 * sigma_loss
        actual_extreme_count = sum(1 for x in losses if x > threshold_3sigma)
        actual_extreme_ratio = actual_extreme_count / max(len(losses), 1)

        return {
            "expected_shortfall": round(es_val, 6),
            "es_alpha": alpha,
            "cvar_1d": round(cvar_1d, 6),
            "cvar_1w": round(cvar_1w, 6),
            "cvar_1m": round(cvar_1m, 6),
            "extreme_loss_probability": round(extreme_prob, 6),
            "actual_extreme_ratio": round(actual_extreme_ratio, 6),
            "actual_extreme_count": actual_extreme_count,
            "threshold_3sigma": round(threshold_3sigma, 6),
            "loss_mean": round(mu_loss, 6),
            "loss_std": round(sigma_loss, 6),
            "sample_size": len(clean),
            "data_quality": "estimated",
        }

    def _cvar(self, losses: List[float], alpha: float) -> float:
        if not losses:
            return 0.0
        var_val = _quantile(losses, alpha)
        tail = [x for x in losses if x >= var_val]
        return max(var_val, _safe_mean(tail, var_val))

    def _aggregate_returns(self, returns: List[float], window: int) -> List[float]:
        if len(returns) < window:
            return []
        aggregated = []
        for i in range(0, len(returns) - window + 1, window):
            chunk = returns[i:i + window]
            prod = 1.0
            for r in chunk:
                prod *= (1.0 + r)
            aggregated.append(prod - 1.0)
        return aggregated

    def _normal_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _empty(self) -> Dict[str, Any]:
        return {
            "expected_shortfall": 0.0,
            "es_alpha": 0.95,
            "cvar_1d": 0.0,
            "cvar_1w": 0.0,
            "cvar_1m": 0.0,
            "extreme_loss_probability": 0.0,
            "actual_extreme_ratio": 0.0,
            "actual_extreme_count": 0,
            "threshold_3sigma": 0.0,
            "loss_mean": 0.0,
            "loss_std": 0.0,
            "sample_size": 0,
            "data_quality": "no_data",
        }


# ---------------------------------------------------------------------------
# 2. Liquidity Risk Metrics
# ---------------------------------------------------------------------------

class LiquidityRiskMetrics:
    """Liquidity risk: bid-ask proxy, Kyle's Lambda, LCR approximation."""

    def compute(
        self,
        volume_data: Optional[List[float]] = None,
        price_data: Optional[List[float]] = None,
        high_data: Optional[List[float]] = None,
        low_data: Optional[List[float]] = None,
        financial_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        volume = [_to_float(v) for v in (volume_data or []) if _to_float(v) > 0]
        prices = [_to_float(p) for p in (price_data or []) if _to_float(p) > 0]
        highs = [_to_float(h) for h in (high_data or []) if _to_float(h) > 0]
        lows = [_to_float(l) for l in (low_data or []) if _to_float(l) > 0]

        has_ohlc = bool(highs and lows and prices)

        # Bid-ask spread proxy: (high - low) / close
        if has_ohlc and len(highs) == len(lows) and len(lows) == len(prices):
            spreads = []
            for h, l, c in zip(highs, lows, prices):
                if c > 0:
                    spreads.append((h - l) / c)
            bid_ask_proxy = _safe_mean(spreads)
            spread_quality = "estimated"
        elif prices and len(prices) >= 2:
            # Fallback: use daily return range as spread proxy
            returns = [abs(prices[i] - prices[i - 1]) / prices[i - 1]
                       for i in range(1, len(prices)) if prices[i - 1] > 0]
            bid_ask_proxy = _safe_mean(returns) * 0.5
            spread_quality = "rough_estimate"
        else:
            bid_ask_proxy = 0.02
            spread_quality = "default"

        # Kyle's Lambda: cov(return, volume) / var(volume)
        if volume and prices and len(volume) == len(prices) and len(volume) >= 3:
            returns = []
            for i in range(1, len(prices)):
                if prices[i - 1] > 0:
                    returns.append((prices[i] - prices[i - 1]) / prices[i - 1])
            vol_aligned = volume[1:]
            min_len = min(len(returns), len(vol_aligned))
            if min_len >= 3:
                r_slice = returns[:min_len]
                v_slice = vol_aligned[:min_len]
                mean_r = _safe_mean(r_slice)
                mean_v = _safe_mean(v_slice)
                cov_rv = sum((r_slice[i] - mean_r) * (v_slice[i] - mean_v) for i in range(min_len)) / min_len
                var_v = sum((v_slice[i] - mean_v) ** 2 for i in range(min_len)) / min_len
                kyle_lambda = cov_rv / var_v if var_v > 1e-12 else 0.0
                impact_quality = "estimated"
            else:
                kyle_lambda = 0.0
                impact_quality = "insufficient_data"
        else:
            kyle_lambda = 0.0
            impact_quality = "no_data"

        # LCR approximation: liquid_assets / 30day_outflow
        financial = financial_data or {}
        liquid_assets = _to_float(financial.get("liquid_assets"))
        total_assets = _to_float(financial.get("total_assets"))
        current_liabilities = _to_float(financial.get("current_liabilities"))
        if liquid_assets > 0 and current_liabilities > 0:
            lcr = liquid_assets / current_liabilities
            lcr_quality = "estimated"
        elif total_assets > 0:
            # Rough proxy: current assets / current liabilities
            current_assets = _to_float(financial.get("current_assets"), total_assets * 0.5)
            lcr = current_assets / max(current_liabilities, total_assets * 0.3)
            lcr_quality = "rough_estimate"
        else:
            lcr = 1.0
            lcr_quality = "default"

        # Amihud illiquidity ratio: mean(|return| / volume)
        if volume and prices and len(volume) == len(prices) and len(volume) >= 2:
            amihud_values = []
            for i in range(1, len(prices)):
                if prices[i - 1] > 0 and volume[i] > 0:
                    amihud_values.append(abs((prices[i] - prices[i - 1]) / prices[i - 1]) / volume[i])
            amihud_illiq = _safe_mean(amihud_values, 0.0)
        else:
            amihud_illiq = 0.0

        # Composite liquidity score (0=liquid, 1=illiquid)
        spread_score = _clip(bid_ask_proxy / 0.05)
        impact_score = _clip(abs(kyle_lambda) * 1000)
        lcr_score = _clip(1.0 - lcr / 3.0)
        amihud_score = _clip(amihud_illiq * 1e6)
        composite = _clip(
            0.30 * spread_score +
            0.25 * impact_score +
            0.25 * lcr_score +
            0.20 * amihud_score
        )

        return {
            "bid_ask_spread_proxy": round(bid_ask_proxy, 6),
            "bid_ask_quality": spread_quality,
            "kyle_lambda": round(kyle_lambda, 8),
            "kyle_lambda_quality": impact_quality,
            "lcr_approximation": round(lcr, 4),
            "lcr_quality": lcr_quality,
            "amihud_illiquidity": round(amihud_illiq, 8),
            "composite_liquidity_score": round(composite, 4),
            "volume_count": len(volume),
            "data_quality": "mixed",
        }


# ---------------------------------------------------------------------------
# 3. Credit Risk Metrics
# ---------------------------------------------------------------------------

class CreditRiskMetrics:
    """Credit risk: Altman Z-score, PD estimation, credit spread proxy."""

    def compute(
        self,
        financial_data: Optional[Dict[str, Any]] = None,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        financial = financial_data or {}
        market = market_data or {}

        # Altman Z-score: 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
        total_assets = _to_float(financial.get("total_assets"))
        working_capital = _to_float(financial.get("working_capital"))
        retained_earnings = _to_float(financial.get("retained_earnings"))
        ebit = _to_float(financial.get("ebit"))
        market_cap = _to_float(market.get("market_cap"))
        total_liabilities = _to_float(financial.get("total_liabilities"))
        sales = _to_float(financial.get("revenue") or financial.get("sales"))

        z_score = None
        z_quality = "no_data"
        if total_assets > 0:
            x1 = working_capital / total_assets
            x2 = retained_earnings / total_assets
            x3 = ebit / total_assets if ebit != 0 else 0.0
            x4 = market_cap / max(total_liabilities, 1.0) if market_cap > 0 else 0.0
            x5 = sales / total_assets if sales > 0 else 0.0
            z_score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
            z_quality = "estimated"

        # PD estimation via simplified Merton model
        pd = None
        pd_quality = "no_data"
        equity_vol = _to_float(market.get("equity_volatility"))
        asset_vol = _to_float(market.get("asset_volatility"), equity_vol)
        leverage = _to_float(market.get("leverage_ratio"))
        if total_assets > 0 and total_liabilities > 0:
            leverage = total_liabilities / total_assets
        if asset_vol > 0 and leverage > 0:
            # Simplified Merton: PD = N(-DD) where DD = (ln(A/D) + (mu - 0.5*sigma^2)*T) / (sigma*sqrt(T))
            debt = max(total_liabilities, 1.0)
            asset_value = max(total_assets, debt)
            T = 1.0
            mu = 0.05  # Expected asset return assumption
            sigma = min(asset_vol, 1.0)
            dd = (math.log(asset_value / debt) + (mu - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            pd = max(0.0, min(1.0, 0.5 * (1.0 + math.erf(-dd / math.sqrt(2.0)))))
            pd_quality = "estimated"

        # Credit spread proxy: use equity volatility as proxy
        equity_vol = _to_float(market.get("equity_volatility"))
        stock_returns = market.get("stock_returns")
        if isinstance(stock_returns, list) and len(stock_returns) >= 5:
            equity_vol = _safe_stdev(stock_returns)
        if equity_vol > 0:
            credit_spread = equity_vol * 0.3  # Rough mapping
            spread_quality = "estimated"
        else:
            credit_spread = 0.02
            spread_quality = "default"

        # Composite credit score (0=safe, 1=distressed)
        if z_score is not None:
            z_score_norm = _clip(1.0 - (z_score + 3.0) / 5.0)  # Z>2.99=safe, Z<1.81=distressed
        else:
            z_score_norm = 0.5
        pd_norm = _clip(pd) if pd is not None else 0.5
        spread_norm = _clip(credit_spread / 0.10)

        composite = _clip(0.35 * z_score_norm + 0.40 * pd_norm + 0.25 * spread_norm)

        return {
            "altman_z_score": round(z_score, 4) if z_score is not None else None,
            "z_score_quality": z_quality,
            "z_score_zone": self._z_score_zone(z_score),
            "probability_of_default": round(pd, 6) if pd is not None else None,
            "pd_quality": pd_quality,
            "credit_spread_proxy": round(credit_spread, 6),
            "spread_quality": spread_quality,
            "composite_credit_score": round(composite, 4),
            "data_quality": "mixed",
        }

    def _z_score_zone(self, z: Optional[float]) -> str:
        if z is None:
            return "unknown"
        if z > 2.99:
            return "safe"
        if z > 1.81:
            return "gray"
        return "distressed"


# ---------------------------------------------------------------------------
# 4. Concentration Risk Metrics
# ---------------------------------------------------------------------------

class ConcentrationRiskMetrics:
    """Concentration risk: HHI, single-asset limit breach, sector exposure limits."""

    def compute(
        self,
        portfolio_weights: Optional[List[float]] = None,
        asset_sectors: Optional[List[str]] = None,
        asset_names: Optional[List[str]] = None,
        single_asset_limit: float = 0.10,
        sector_limit: float = 0.30,
    ) -> Dict[str, Any]:
        weights = portfolio_weights or []
        sectors = asset_sectors or []
        names = asset_names or []

        if not weights:
            return self._empty()

        abs_weights = [abs(w) for w in weights]
        n = len(abs_weights)

        # HHI (Herfindahl-Hirschman Index)
        hhi = sum(w ** 2 for w in abs_weights)
        if n > 1:
            normalized_hhi = _clip((hhi - 1.0 / n) / (1.0 - 1.0 / n))
        else:
            normalized_hhi = 1.0

        # Single asset limit breach
        breaches = []
        for i, w in enumerate(abs_weights):
            if w > single_asset_limit:
                name = names[i] if i < len(names) else f"asset_{i}"
                breaches.append({
                    "asset": name,
                    "weight": round(w, 4),
                    "limit": single_asset_limit,
                    "breach_ratio": round(w / single_asset_limit, 2),
                })

        # Sector concentration
        sector_weights: Dict[str, float] = {}
        for i, w in enumerate(abs_weights):
            sector = sectors[i] if i < len(sectors) else "unknown"
            sector_weights[sector] = sector_weights.get(sector, 0.0) + w

        sector_breaches = []
        for sector, sw in sector_weights.items():
            if sw > sector_limit:
                sector_breaches.append({
                    "sector": sector,
                    "weight": round(sw, 4),
                    "limit": sector_limit,
                    "breach_ratio": round(sw / sector_limit, 2),
                })

        max_single = max(abs_weights) if abs_weights else 0.0
        max_sector = max(sector_weights.values()) if sector_weights else 0.0

        # Effective number of positions (1/HHI)
        effective_n = 1.0 / hhi if hhi > 1e-12 else n

        # Composite concentration score
        hhi_score = _clip(normalized_hhi)
        single_breach_score = _clip(len(breaches) / max(n, 1))
        sector_breach_score = _clip(len(sector_breaches) / max(len(set(sectors)) if sectors else 1, 1))
        composite = _clip(0.40 * hhi_score + 0.30 * single_breach_score + 0.30 * sector_breach_score)

        return {
            "hhi": round(hhi, 6),
            "normalized_hhi": round(normalized_hhi, 6),
            "effective_positions": round(effective_n, 2),
            "largest_weight": round(max_single, 6),
            "single_asset_breaches": breaches,
            "single_asset_breach_count": len(breaches),
            "sector_weights": {k: round(v, 4) for k, v in sector_weights.items()},
            "sector_breaches": sector_breaches,
            "sector_breach_count": len(sector_breaches),
            "max_sector_exposure": round(max_sector, 4),
            "composite_concentration_score": round(composite, 4),
            "total_positions": n,
        }

    def _empty(self) -> Dict[str, Any]:
        return {
            "hhi": 0.0,
            "normalized_hhi": 0.0,
            "effective_positions": 0.0,
            "largest_weight": 0.0,
            "single_asset_breaches": [],
            "single_asset_breach_count": 0,
            "sector_weights": {},
            "sector_breaches": [],
            "sector_breach_count": 0,
            "max_sector_exposure": 0.0,
            "composite_concentration_score": 0.0,
            "total_positions": 0,
        }


# ---------------------------------------------------------------------------
# Singleton instances
# ---------------------------------------------------------------------------

tail_risk_metrics = TailRiskMetrics()
liquidity_risk_metrics = LiquidityRiskMetrics()
credit_risk_metrics = CreditRiskMetrics()
concentration_risk_metrics = ConcentrationRiskMetrics()
