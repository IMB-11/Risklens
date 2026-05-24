"""Stress testing engine: historical scenarios, hypothetical, Monte Carlo, reverse stress test."""

from __future__ import annotations

import math
import random
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
    q = max(0.0, min(1.0, q))
    idx = (len(sorted_vals) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = idx - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


class StressTestingEngine:
    """Comprehensive stress testing with historical, hypothetical, Monte Carlo, and reverse tests."""

    HISTORICAL_SCENARIOS: Dict[str, Dict[str, Any]] = {
        "2015_crash": {
            "name": "2015 A-Share Crash",
            "description": "2015年A股股灾：杠杆资金清理引发的市场崩盘",
            "market_shock": -0.45,
            "vol_multiplier": 3.0,
            "correlation_shift": 0.30,
            "liquidity_shrink": 0.60,
            "duration_days": 45,
            "sector_impacts": {
                "tech": -0.52, "consumer": -0.38, "financials": -0.42,
                "industrials": -0.48, "healthcare": -0.35, "materials": -0.50,
            },
        },
        "2018_trade_war": {
            "name": "2018 US-China Trade War",
            "description": "2018年中美贸易战：关税升级与科技制裁",
            "market_shock": -0.25,
            "vol_multiplier": 2.0,
            "correlation_shift": 0.20,
            "liquidity_shrink": 0.30,
            "duration_days": 90,
            "sector_impacts": {
                "tech": -0.35, "consumer": -0.18, "financials": -0.22,
                "industrials": -0.30, "healthcare": -0.15, "materials": -0.28,
            },
        },
        "2020_covid": {
            "name": "2020 COVID-19 Pandemic",
            "description": "2020年新冠疫情冲击：全球经济停摆",
            "market_shock": -0.30,
            "vol_multiplier": 2.5,
            "correlation_shift": 0.35,
            "liquidity_shrink": 0.50,
            "duration_days": 30,
            "sector_impacts": {
                "tech": -0.15, "consumer": -0.35, "financials": -0.28,
                "industrials": -0.32, "healthcare": +0.10, "materials": -0.25,
            },
        },
        "2022_rate_hike": {
            "name": "2022 Fed Rate Hike Cycle",
            "description": "2022年美联储加息周期：全球流动性收紧",
            "market_shock": -0.20,
            "vol_multiplier": 1.8,
            "correlation_shift": 0.15,
            "liquidity_shrink": 0.25,
            "duration_days": 120,
            "sector_impacts": {
                "tech": -0.30, "consumer": -0.15, "financials": +0.05,
                "industrials": -0.18, "healthcare": -0.12, "materials": -0.20,
            },
        },
        "2024_ai_bubble": {
            "name": "2024 AI Bubble Burst",
            "description": "假设AI泡沫破裂：科技股估值崩塌",
            "market_shock": -0.35,
            "vol_multiplier": 2.8,
            "correlation_shift": 0.25,
            "liquidity_shrink": 0.40,
            "duration_days": 60,
            "sector_impacts": {
                "tech": -0.55, "consumer": -0.20, "financials": -0.25,
                "industrials": -0.15, "healthcare": -0.10, "materials": -0.18,
            },
        },
    }

    def run_full_stress_test(
        self,
        portfolio_returns: List[float],
        weights: Optional[List[float]] = None,
        asset_sectors: Optional[List[str]] = None,
        historical_scenarios: Optional[List[str]] = None,
        hypothetical_scenarios: Optional[List[Dict[str, Any]]] = None,
        monte_carlo_sims: int = 10000,
        horizon_days: int = 30,
        reverse_targets: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        clean = [r for r in portfolio_returns if math.isfinite(r)]
        if not clean:
            clean = [0.0]

        results: Dict[str, Any] = {
            "input_summary": {
                "return_count": len(clean),
                "mean_return": round(_safe_mean(clean), 6),
                "volatility": round(_safe_stdev(clean, 0.02), 6),
                "worst_return": round(min(clean), 6),
                "best_return": round(max(clean), 6),
            },
        }

        # 1. Historical scenarios
        scenario_keys = historical_scenarios or list(self.HISTORICAL_SCENARIOS.keys())
        results["historical_results"] = self._run_historical_scenarios(
            clean, weights, asset_sectors, scenario_keys
        )

        # 2. Hypothetical scenarios
        results["hypothetical_results"] = self._run_hypothetical_scenarios(
            clean, hypothetical_scenarios or []
        )

        # 3. Monte Carlo stress test
        results["monte_carlo_results"] = self._run_monte_carlo_stress(
            clean, monte_carlo_sims, horizon_days
        )

        # 4. Reverse stress test
        targets = reverse_targets or [-0.10, -0.15, -0.20, -0.30]
        results["reverse_stress_results"] = self._run_reverse_stress(clean, targets)

        # 5. Aggregate risk score
        results["aggregate_stress_risk"] = self._compute_aggregate_stress_risk(results)

        return results

    # ------------------------------------------------------------------
    # 1. Historical Scenario Stress Test
    # ------------------------------------------------------------------

    def _run_historical_scenarios(
        self,
        returns: List[float],
        weights: Optional[List[float]],
        asset_sectors: Optional[List[str]],
        scenario_keys: List[str],
    ) -> List[Dict[str, Any]]:
        results = []
        mu = _safe_mean(returns)
        sigma = max(_safe_stdev(returns, 0.02), 1e-6)

        for key in scenario_keys:
            scenario = self.HISTORICAL_SCENARIOS.get(key)
            if not scenario:
                continue

            shock = scenario["market_shock"]
            vol_mult = scenario["vol_multiplier"]
            corr_shift = scenario["correlation_shift"]
            liq_shrink = scenario["liquidity_shrink"]
            duration = scenario["duration_days"]

            # Apply shock to returns
            stressed_mu = mu + shock / max(duration, 1)
            stressed_sigma = sigma * vol_mult
            stressed_returns = [stressed_mu + stressed_sigma * random.gauss(0, 1)
                                for _ in range(duration)]

            # Portfolio loss under stress
            if weights:
                # Sector-adjusted shock
                sector_shocks = scenario.get("sector_impacts", {})
                portfolio_shock = 0.0
                for i, w in enumerate(weights):
                    sector = asset_sectors[i] if asset_sectors and i < len(asset_sectors) else "unknown"
                    sector_shock = sector_shocks.get(sector, shock)
                    portfolio_shock += abs(w) * sector_shock
                total_loss = portfolio_shock
            else:
                total_loss = sum(stressed_returns)

            # Liquidity impact
            liquidity_cost = abs(total_loss) * liq_shrink * 0.1

            results.append({
                "scenario_id": key,
                "name": scenario["name"],
                "description": scenario["description"],
                "market_shock": round(shock, 4),
                "portfolio_loss": round(total_loss, 6),
                "liquidity_cost": round(liquidity_cost, 6),
                "total_impact": round(total_loss - liquidity_cost, 6),
                "volatility_regime": round(stressed_sigma, 6),
                "correlation_spike": round(corr_shift, 4),
                "duration_days": duration,
                "recovery_days_estimate": int(duration * 2.5),
            })

        return results

    # ------------------------------------------------------------------
    # 2. Hypothetical Scenario Stress Test
    # ------------------------------------------------------------------

    def _run_hypothetical_scenarios(
        self,
        returns: List[float],
        scenarios: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not scenarios:
            return []

        mu = _safe_mean(returns)
        sigma = max(_safe_stdev(returns, 0.02), 1e-6)
        results = []

        for i, scenario in enumerate(scenarios):
            market_drop = _to_float(scenario.get("market_drop", -0.10))
            vol_increase = _to_float(scenario.get("vol_increase", 1.5))
            rate_change = _to_float(scenario.get("rate_change_bp", 0)) / 10000.0
            fx_shock = _to_float(scenario.get("fx_shock", 0.0))
            duration = int(_to_float(scenario.get("duration_days", 20)))
            name = scenario.get("name", f"Hypothetical_{i+1}")

            stressed_mu = mu + market_drop / max(duration, 1) - rate_change / max(duration, 1)
            stressed_sigma = sigma * vol_increase
            stressed_returns = [stressed_mu + stressed_sigma * random.gauss(0, 1)
                                for _ in range(duration)]

            total_loss = sum(stressed_returns) + fx_shock
            scenario_shock = market_drop + rate_change + fx_shock

            results.append({
                "scenario_id": f"hypothetical_{i+1}",
                "name": name,
                "parameters": {
                    "market_drop": round(market_drop, 4),
                    "vol_increase": round(vol_increase, 2),
                    "rate_change_bp": round(rate_change * 10000, 1),
                    "fx_shock": round(fx_shock, 4),
                    "duration_days": duration,
                },
                "portfolio_loss": round(total_loss, 6),
                "total_shock": round(scenario_shock, 6),
                "volatility_regime": round(stressed_sigma, 6),
            })

        return results

    # ------------------------------------------------------------------
    # 3. Monte Carlo Stress Test
    # ------------------------------------------------------------------

    def _run_monte_carlo_stress(
        self,
        returns: List[float],
        n_sims: int,
        horizon_days: int,
    ) -> Dict[str, Any]:
        mu = _safe_mean(returns)
        sigma = max(_safe_stdev(returns, 0.02), 1e-6)

        # Generate simulation paths
        sim_losses = []
        sim_max_drawdowns = []
        sim_final_values = []

        for _ in range(n_sims):
            path = [1.0]
            current = 1.0
            peak = 1.0
            max_dd = 0.0
            for _ in range(horizon_days):
                daily_return = mu + sigma * random.gauss(0, 1)
                current *= (1.0 + daily_return)
                path.append(current)
                peak = max(peak, current)
                dd = (current - peak) / peak
                max_dd = min(max_dd, dd)
            sim_losses.append(1.0 - current)
            sim_max_drawdowns.append(max_dd)
            sim_final_values.append(current)

        # VaR and CVaR from simulations
        var_95 = max(0.0, _quantile(sim_losses, 0.95))
        var_99 = max(0.0, _quantile(sim_losses, 0.99))
        cvar_95 = max(var_95, _safe_mean([x for x in sim_losses if x >= var_95], var_95))
        cvar_99 = max(var_99, _safe_mean([x for x in sim_losses if x >= var_99], var_99))
        es_95 = cvar_95  # ES = CVaR

        # Max drawdown statistics
        worst_dd = min(sim_max_drawdowns)
        avg_dd = _safe_mean(sim_max_drawdowns)
        dd_95 = _quantile(sim_max_drawdowns, 0.05)  # 5th percentile of drawdowns

        # Loss distribution percentiles
        loss_distribution = {
            "p5": round(_quantile(sim_losses, 0.05), 6),
            "p10": round(_quantile(sim_losses, 0.10), 6),
            "p25": round(_quantile(sim_losses, 0.25), 6),
            "p50": round(_quantile(sim_losses, 0.50), 6),
            "p75": round(_quantile(sim_losses, 0.75), 6),
            "p90": round(_quantile(sim_losses, 0.90), 6),
            "p95": round(_quantile(sim_losses, 0.95), 6),
            "p99": round(_quantile(sim_losses, 0.99), 6),
        }

        # Probability of loss
        prob_loss = sum(1 for x in sim_losses if x > 0) / n_sims
        prob_loss_5pct = sum(1 for x in sim_losses if x > 0.05) / n_sims
        prob_loss_10pct = sum(1 for x in sim_losses if x > 0.10) / n_sims

        return {
            "n_simulations": n_sims,
            "horizon_days": horizon_days,
            "var_95": round(var_95, 6),
            "var_99": round(var_99, 6),
            "cvar_95": round(cvar_95, 6),
            "cvar_99": round(cvar_99, 6),
            "es_95": round(es_95, 6),
            "max_drawdown_worst": round(worst_dd, 6),
            "max_drawdown_avg": round(avg_dd, 6),
            "max_drawdown_95": round(dd_95, 6),
            "loss_distribution": loss_distribution,
            "probability_of_loss": round(prob_loss, 4),
            "probability_loss_5pct": round(prob_loss_5pct, 4),
            "probability_loss_10pct": round(prob_loss_10pct, 4),
            "mean_loss": round(_safe_mean(sim_losses), 6),
            "std_loss": round(_safe_stdev(sim_losses, 0.0), 6),
        }

    # ------------------------------------------------------------------
    # 4. Reverse Stress Test
    # ------------------------------------------------------------------

    def _run_reverse_stress(
        self,
        returns: List[float],
        target_losses: List[float],
    ) -> List[Dict[str, Any]]:
        mu = _safe_mean(returns)
        sigma = max(_safe_stdev(returns, 0.02), 1e-6)
        results = []

        for target in target_losses:
            target_abs = abs(target)
            # Estimate required shock: need to shift distribution so that
            # the median outcome equals the target loss
            # target = mu_stressed * horizon + z * sigma_stressed * sqrt(horizon)
            # Simplified: required_market_shock ~= target - current_expected
            horizon = 20
            current_expected = mu * horizon
            required_shock = target - current_expected

            # Scenario description
            if target_abs <= 0.10:
                severity = "mild"
                description = "温和下跌情景"
            elif target_abs <= 0.20:
                severity = "moderate"
                description = "中度下跌情景"
            elif target_abs <= 0.30:
                severity = "severe"
                description = "严重下跌情景"
            else:
                severity = "extreme"
                description = "极端下跌情景"

            # Required conditions
            vol_multiplier = max(1.0, target_abs / max(sigma * math.sqrt(horizon), 0.01))
            required_daily_shock = required_shock / horizon

            results.append({
                "target_loss": round(target, 4),
                "severity": severity,
                "description": description,
                "required_market_shock": round(required_shock, 6),
                "required_daily_shock": round(required_daily_shock, 6),
                "required_vol_multiplier": round(vol_multiplier, 2),
                "estimated_trigger": self._estimate_trigger(target_abs),
                "probability_estimate": round(self._estimate_probability(target_abs, mu, sigma), 6),
            })

        return results

    def _estimate_trigger(self, target_loss: float) -> str:
        if target_loss <= 0.05:
            return "正常市场波动"
        if target_loss <= 0.10:
            return "局部利空事件或行业调整"
        if target_loss <= 0.15:
            return "宏观经济数据恶化或政策收紧"
        if target_loss <= 0.20:
            return "系统性风险事件（如贸易战升级、信用危机）"
        if target_loss <= 0.30:
            return "重大黑天鹅事件（如金融危机、地缘冲突）"
        return "极端尾部事件（如全球性危机、市场崩盘）"

    def _estimate_probability(self, target_loss: float, mu: float, sigma: float) -> float:
        if sigma <= 0:
            return 0.0
        # Rough estimate using normal distribution
        z = (target_loss + mu * 20) / (sigma * math.sqrt(20))
        return max(0.0, min(1.0, 0.5 * (1.0 + math.erf(-z / math.sqrt(2.0)))))

    # ------------------------------------------------------------------
    # 5. Aggregate Stress Risk
    # ------------------------------------------------------------------

    def _compute_aggregate_stress_risk(self, results: Dict[str, Any]) -> Dict[str, Any]:
        # Historical scenario worst case
        hist_results = results.get("historical_results", [])
        worst_hist_loss = min((r["total_impact"] for r in hist_results), default=0.0) if hist_results else 0.0

        # Monte Carlo VaR
        mc = results.get("monte_carlo_results", {})
        mc_var_95 = _to_float(mc.get("var_95"), 0.0)
        mc_cvar_95 = _to_float(mc.get("cvar_95"), 0.0)
        mc_worst_dd = _to_float(mc.get("max_drawdown_worst"), 0.0)

        # Reverse stress test
        reverse = results.get("reverse_stress_results", [])
        min_trigger = min((abs(r["target_loss"]) for r in reverse), default=0.3) if reverse else 0.3

        # Composite score (0=safe, 1=extremely stressed)
        hist_score = _clip(abs(worst_hist_loss) / 0.45)
        mc_score = _clip(mc_cvar_95 / 0.15)
        dd_score = _clip(abs(mc_worst_dd) / 0.30)
        reverse_score = _clip(1.0 - min_trigger / 0.30)

        composite = _clip(
            0.30 * hist_score +
            0.35 * mc_score +
            0.20 * dd_score +
            0.15 * reverse_score
        )

        if composite >= 0.75:
            stress_level = "critical"
        elif composite >= 0.55:
            stress_level = "high"
        elif composite >= 0.35:
            stress_level = "medium"
        else:
            stress_level = "low"

        return {
            "composite_stress_score": round(composite, 4),
            "stress_level": stress_level,
            "worst_historical_loss": round(worst_hist_loss, 6),
            "mc_var_95": round(mc_var_95, 6),
            "mc_cvar_95": round(mc_cvar_95, 6),
            "mc_max_drawdown": round(mc_worst_dd, 6),
            "min_trigger_loss": round(min_trigger, 4),
            "component_scores": {
                "historical": round(hist_score, 4),
                "monte_carlo": round(mc_score, 4),
                "drawdown": round(dd_score, 4),
                "reverse": round(reverse_score, 4),
            },
        }

    def get_available_scenarios(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": key,
                "name": s["name"],
                "description": s["description"],
                "market_shock": s["market_shock"],
                "duration_days": s["duration_days"],
            }
            for key, s in self.HISTORICAL_SCENARIOS.items()
        ]


# Singleton
stress_testing_engine = StressTestingEngine()
