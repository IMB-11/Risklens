"""Risk weight optimizer: backtest-driven weight calibration for multi-factor risk engine."""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Dict, List, Optional, Tuple

from .market_state_engine import market_state_engine_2


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


def _rank_correlation(xs: List[float], ys: List[float]) -> float:
    """Spearman rank correlation."""
    n = len(xs)
    if n < 3:
        return 0.0

    def _rank(vals: List[float]) -> List[float]:
        indexed = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * n
        for rank, idx in enumerate(indexed):
            ranks[idx] = float(rank + 1)
        return ranks

    rx = _rank(xs)
    ry = _rank(ys)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    std_x = math.sqrt(sum((r - mean_rx) ** 2 for r in rx))
    std_y = math.sqrt(sum((r - mean_ry) ** 2 for r in ry))
    if std_x < 1e-12 or std_y < 1e-12:
        return 0.0
    return cov / (std_x * std_y)


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
# Factor Signal Simulation
# ---------------------------------------------------------------------------

class FactorSimulator:
    """Simulate 6 risk factor signals from historical OHLCV + news data."""

    # Regime multipliers for factor adjustment
    REGIME_MULTIPLIERS = {
        "bull": {       # regime < 0.4
            "sentiment": 1.2,    # High attention = momentum in bull
            "trend": 1.1,        # Strong trend = follow
            "volatility": 0.8,   # Low weight: volatility = opportunity
            "event": 1.1,
            "uncertainty": 0.7,  # Low weight: uncertainty = potential upside
        },
        "sideways": {   # 0.4 <= regime <= 0.6
            "sentiment": 1.0,
            "trend": 1.0,
            "volatility": 1.0,
            "event": 1.0,
            "uncertainty": 1.0,
        },
        "bear": {       # regime > 0.6
            "sentiment": 0.8,    # Low weight: attention = panic
            "trend": 0.9,
            "volatility": 1.2,   # High weight: volatility = risk
            "event": 0.9,
            "uncertainty": 1.3,  # High weight: uncertainty = danger
        },
    }

    def _get_regime_multipliers(self, regime: float) -> Dict[str, float]:
        """Get regime-adjusted multipliers for factors."""
        if regime < 0.4:
            return self.REGIME_MULTIPLIERS["bull"]
        elif regime > 0.6:
            return self.REGIME_MULTIPLIERS["bear"]
        else:
            return self.REGIME_MULTIPLIERS["sideways"]

    def compute_factors(
        self,
        returns: List[float],
        rsi_values: List[float],
        ma5: List[float],
        ma20: List[float],
        ma60: List[float],
        news_counts: List[float],
        window: int = 20,
        regime_stride: int = 10,
    ) -> Dict[str, List[float]]:
        n = len(returns)
        factors: Dict[str, List[float]] = {
            "sentiment": [],
            "trend": [],
            "volatility": [],
            "event": [],
            "uncertainty": [],
            "market_regime": [],
        }

        # Pre-compute regime signals at stride intervals for speed
        regime_cache: Dict[int, float] = {}
        min_train = 120
        for t in range(min_train, n, regime_stride):
            regime_cache[t] = self._market_regime_signal(returns, t)

        for t in range(n):
            start = max(0, t - window + 1)
            ret_slice = returns[start:t + 1]
            rsi_slice = rsi_values[start:t + 1]
            news_slice = news_counts[start:t + 1]

            # Get current regime
            if t < min_train:
                regime = 0.50
            else:
                key = (t // regime_stride) * regime_stride
                if key in regime_cache:
                    regime = regime_cache[key]
                else:
                    k_lo = (t // regime_stride) * regime_stride
                    k_hi = k_lo + regime_stride
                    v_lo = regime_cache.get(k_lo, 0.50)
                    v_hi = regime_cache.get(k_hi, v_lo)
                    frac = (t - k_lo) / regime_stride if k_hi > k_lo else 0.0
                    regime = v_lo + frac * (v_hi - v_lo)

            # Get regime multipliers
            multipliers = self._get_regime_multipliers(regime)

            # Compute base factors
            sentiment_base = self._sentiment_signal(news_slice, ret_slice)
            trend_base = self._trend_signal(
                rsi_slice, ma5[t] if t < len(ma5) else 0.0,
                ma20[t] if t < len(ma20) else 0.0,
                ma60[t] if t < len(ma60) else 0.0,
                returns[t] if t < len(returns) else 0.0,
            )
            volatility_base = self._volatility_signal(ret_slice)
            event_base = self._event_signal(news_slice, ret_slice)
            uncertainty_base = self._uncertainty_signal(ret_slice)

            # Apply regime multipliers
            factors["sentiment"].append(_clip(sentiment_base * multipliers["sentiment"]))
            factors["trend"].append(_clip(trend_base * multipliers["trend"]))
            factors["volatility"].append(_clip(volatility_base * multipliers["volatility"]))
            factors["event"].append(_clip(event_base * multipliers["event"]))
            factors["uncertainty"].append(_clip(uncertainty_base * multipliers["uncertainty"]))
            factors["market_regime"].append(regime)

        return factors

    def _sentiment_signal(self, news: List[float], returns: List[float]) -> float:
        if len(news) < 2:
            return 0.50
        avg_news = _safe_mean(news[:-1], 1.0)
        latest = news[-1]
        change = (latest - avg_news) / max(avg_news, 1.0)
        ret_mean = abs(_safe_mean(returns, 0.0))
        news_vol = _safe_stdev(news, 0.0) / max(_safe_mean(news, 1.0), 1.0)
        raw = 0.45 * _clip(0.5 + change * 0.5) + 0.30 * _clip(0.5 + ret_mean * 10) + 0.25 * _clip(news_vol)
        return _clip(raw)

    def _trend_signal(
        self, rsi: List[float], ma5: float, ma20: float, ma60: float, ret: float
    ) -> float:
        rsi_val = rsi[-1] if rsi else 50.0
        rsi_norm = _clip((rsi_val - 30) / 40.0)
        ma_score = 0.0
        if ma20 > 0 and ma60 > 0:
            if ma5 > ma20 > ma60:
                ma_score = 0.80
            elif ma5 > ma20:
                ma_score = 0.65
            elif ma5 < ma20 < ma60:
                ma_score = 0.20
            elif ma5 < ma20:
                ma_score = 0.35
            else:
                ma_score = 0.50
        else:
            ma_score = 0.50
        momentum = _clip(0.5 + ret * 5.0)
        return _clip(0.35 * rsi_norm + 0.40 * ma_score + 0.25 * momentum)

    def _volatility_signal(self, returns: List[float]) -> float:
        if len(returns) < 5:
            return 0.50
        vol = _safe_stdev(returns, 0.02)
        return _clip(vol / 0.04)

    def _event_signal(self, news: List[float], returns: List[float]) -> float:
        if len(news) < 3:
            return 0.50
        avg_news = _safe_mean(news[:-1], 1.0)
        latest_news = news[-1]
        news_spike = max(0.0, (latest_news - avg_news) / max(avg_news, 1.0))
        ret = returns[-1] if returns else 0.0
        ret_anomaly = abs(ret) / 0.05
        return _clip(0.45 * _clip(news_spike) + 0.55 * _clip(ret_anomaly))

    def _uncertainty_signal(self, returns: List[float]) -> float:
        if len(returns) < 8:
            return 0.50
        vol = _safe_stdev(returns, 0.02)
        mean_ret = abs(_safe_mean(returns, 0.0))
        kurtosis_proxy = vol / max(mean_ret + 1e-6, 1e-6)
        return _clip(kurtosis_proxy / 5.0)

    def _market_regime_signal(self, all_returns: List[float], t: int) -> float:
        start = max(0, t - 59)
        window_returns = all_returns[start:t + 1]
        if len(window_returns) < 5:
            return 0.50
        macro_factors = {"macro_stress": 0.5}
        regime_info = market_state_engine_2.infer_regime(
            window_returns, macro_factors, previous_regime="sideways"
        )
        return _clip(_to_float(regime_info.get("state_stress"), 0.5))


# ---------------------------------------------------------------------------
# Backtest Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Rolling-window backtest for risk factor weights."""

    def __init__(
        self,
        lookback: int = 60,
        min_train: int = 120,
        eval_window: int = 20,
    ):
        self.lookback = lookback
        self.min_train = min_train
        self.eval_window = eval_window

    def run(
        self,
        stock_data: Dict[str, List[float]],
        weights: Dict[str, float],
    ) -> Dict[str, Any]:
        returns = stock_data["return_1d"]
        rsi = stock_data["rsi"]
        ma5 = stock_data["ma5"]
        ma20 = stock_data["ma20"]
        ma60 = stock_data["ma60"]
        news = stock_data["news_count"]
        n = len(returns)

        if n < self.min_train + self.eval_window:
            return self._empty_result()

        simulator = FactorSimulator()
        factors = simulator.compute_factors(returns, rsi, ma5, ma20, ma60, news)

        risk_scores = []
        actual_returns = []
        regime_scores = []

        for t in range(self.min_train, n - self.eval_window):
            score = sum(factors[f][t] * weights.get(f, 0.0) for f in weights)
            future_returns = returns[t + 1:t + 1 + self.eval_window]
            avg_future = _safe_mean(future_returns, 0.0) if future_returns else 0.0
            risk_scores.append(score)
            actual_returns.append(avg_future)
            regime_scores.append(factors["market_regime"][t])

        if len(risk_scores) < 10:
            return self._empty_result()

        return self._evaluate(risk_scores, actual_returns, regime_scores)

    def _evaluate(
        self, risk_scores: List[float], actual_returns: List[float],
        regime_scores: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        n = len(risk_scores)

        # 1. Rank IC (negative expected: high risk_score -> low return)
        ic = _rank_correlation(risk_scores, actual_returns)

        # 2. Sharpe-like signal performance with regime-aware logic
        median_score = _quantile(risk_scores, 0.50)
        signal_returns = []

        for i in range(n):
            # Get regime state (0=bull, 0.5=sideways, 1=crisis)
            regime = regime_scores[i] if regime_scores else 0.5

            # Regime-aware signal direction
            if regime > 0.6:  # Bear/Crisis: high risk -> short (original logic)
                if risk_scores[i] > median_score:
                    signal_returns.append(-actual_returns[i])
                else:
                    signal_returns.append(actual_returns[i])
            elif regime < 0.4:  # Bull: high risk -> long (reversed logic)
                if risk_scores[i] > median_score:
                    signal_returns.append(actual_returns[i])
                else:
                    signal_returns.append(-actual_returns[i])
            else:  # Sideways: use absolute IC direction
                if ic < 0:  # Normal: high risk -> low return
                    if risk_scores[i] > median_score:
                        signal_returns.append(-actual_returns[i])
                    else:
                        signal_returns.append(actual_returns[i])
                else:  # Reversed: high risk -> high return
                    if risk_scores[i] > median_score:
                        signal_returns.append(actual_returns[i])
                    else:
                        signal_returns.append(-actual_returns[i])

        sig_mean = _safe_mean(signal_returns, 0.0)
        sig_std = _safe_stdev(signal_returns, 0.01)
        sharpe = sig_mean / max(sig_std, 1e-6) * math.sqrt(252.0)

        # 3. CVaR ratio: CVaR of high-risk periods vs overall
        high_risk_rets = [actual_returns[i] for i in range(n) if risk_scores[i] > _quantile(risk_scores, 0.75)]
        all_cvar = _quantile([-r for r in actual_returns], 0.95)
        high_cvar = _quantile([-r for r in high_risk_rets], 0.95) if high_risk_rets else all_cvar
        cvar_ratio = high_cvar / max(all_cvar, 1e-6)

        # 4. Max drawdown of signal
        cumulative = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in signal_returns:
            cumulative *= (1.0 + r)
            peak = max(peak, cumulative)
            dd = (cumulative - peak) / peak
            max_dd = min(max_dd, dd)

        return {
            "rank_ic": round(ic, 6),
            "abs_ic": round(abs(ic), 6),
            "sharpe": round(sharpe, 6),
            "cvar_ratio": round(cvar_ratio, 6),
            "max_drawdown": round(max_dd, 6),
            "n_samples": n,
            "signal_mean": round(sig_mean, 6),
            "signal_std": round(sig_std, 6),
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "rank_ic": 0.0, "abs_ic": 0.0, "sharpe": 0.0,
            "cvar_ratio": 1.0, "max_drawdown": 0.0,
            "n_samples": 0, "signal_mean": 0.0, "signal_std": 0.0,
        }


# ---------------------------------------------------------------------------
# Weight Optimizer
# ---------------------------------------------------------------------------

FACTOR_NAMES = ["sentiment", "trend", "volatility", "event", "uncertainty", "market_regime"]

PROFILE_CONFIG = {
    "aggressive": {
        "weights": {"sentiment": 0.22, "trend": 0.18, "volatility": 0.22, "event": 0.16, "uncertainty": 0.12, "market_regime": 0.10},
        "constraints": {"trend": (0.10, 0.30), "volatility": (0.15, 0.30)},
    },
    "moderate": {
        "weights": {"sentiment": 0.24, "trend": 0.18, "volatility": 0.20, "event": 0.16, "uncertainty": 0.12, "market_regime": 0.10},
        "constraints": {},
    },
    "conservative": {
        "weights": {"sentiment": 0.26, "trend": 0.18, "volatility": 0.20, "event": 0.18, "uncertainty": 0.10, "market_regime": 0.10},
        "constraints": {"sentiment": (0.15, 0.35), "event": (0.12, 0.28)},
    },
}


class WeightOptimizer:
    """Optimize risk factor weights via backtesting on historical data."""

    def __init__(
        self,
        data_path: str,
        n_stocks: int = 200,
        n_trials: int = 200,
        ic_w: float = 0.40,
        sharpe_w: float = 0.35,
        cvar_w: float = 0.25,
        seed: int = 42,
    ):
        self.data_path = data_path
        self.n_stocks = n_stocks
        self.n_trials = n_trials
        self.ic_w = ic_w
        self.sharpe_w = sharpe_w
        self.cvar_w = cvar_w
        self.seed = seed
        self.backtest = BacktestEngine()
        self._stock_data_cache: Optional[Dict[str, Dict[str, List[float]]]] = None

    def load_data(self) -> Dict[str, Dict[str, List[float]]]:
        if self._stock_data_cache is not None:
            return self._stock_data_cache

        import pandas as pd
        print("Loading CSV with pandas...")
        df = pd.read_csv(self.data_path, usecols=["symbol", "return_1d", "rsi", "ma5", "ma20", "ma60", "news_count"])
        print(f"  Raw rows: {len(df)}")

        # Fill NaN
        df["rsi"] = df["rsi"].fillna(50.0)
        for col in ["ma5", "ma20", "ma60", "return_1d", "news_count"]:
            df[col] = df[col].fillna(0.0)

        # Filter stocks with >= 200 rows
        counts = df.groupby("symbol").size()
        valid_symbols = counts[counts >= 200].index.tolist()
        print(f"  Valid stocks (>=200 rows): {len(valid_symbols)}")

        # Sample
        random.seed(self.seed)
        if len(valid_symbols) > self.n_stocks:
            valid_symbols = random.sample(valid_symbols, self.n_stocks)

        df = df[df["symbol"].isin(valid_symbols)]

        by_stock: Dict[str, Dict[str, List[float]]] = {}
        for sym, group in df.groupby("symbol"):
            by_stock[sym] = {
                "return_1d": group["return_1d"].tolist(),
                "rsi": group["rsi"].tolist(),
                "ma5": group["ma5"].tolist(),
                "ma20": group["ma20"].tolist(),
                "ma60": group["ma60"].tolist(),
                "news_count": group["news_count"].tolist(),
            }

        self._stock_data_cache = by_stock
        print(f"  Loaded {len(by_stock)} stocks")
        return self._stock_data_cache

    def _objective(
        self, weights: List[float], stock_data: Dict[str, Dict[str, List[float]]]
    ) -> float:
        """Negative composite score (minimize -> maximize original)."""
        w_dict = {FACTOR_NAMES[i]: weights[i] for i in range(6)}

        all_ic = []
        all_sharpe = []
        all_cvar = []

        for sym, data in stock_data.items():
            result = self.backtest.run(data, w_dict)
            if result["n_samples"] < 10:
                continue
            all_ic.append(result["abs_ic"])
            all_sharpe.append(result["sharpe"])
            all_cvar.append(result["cvar_ratio"])

        if not all_ic:
            return 0.0

        avg_ic = _safe_mean(all_ic, 0.0)
        avg_sharpe = _safe_mean(all_sharpe, 0.0)
        avg_cvar = _safe_mean(all_cvar, 1.0)

        # Normalize sharpe to [0,1] range
        sharpe_norm = _clip((avg_sharpe + 2.0) / 4.0)
        cvar_score = _clip(1.0 - avg_cvar / 2.0)

        composite = self.ic_w * avg_ic + self.sharpe_w * sharpe_norm + self.cvar_w * cvar_score
        return -composite  # negative for minimizer

    def optimize(self, profile: str = "moderate") -> Dict[str, Any]:
        stock_data = self.load_data()
        config = PROFILE_CONFIG.get(profile, PROFILE_CONFIG["moderate"])
        constraints = config.get("constraints", {})

        print(f"\n=== Optimizing '{profile}' profile ===")
        print(f"Stocks: {len(stock_data)}, Trials: {self.n_trials}")

        best_score = float("inf")
        best_weights = list(config["weights"].values())

        # Phase 1: Grid search over coarse space
        print("Phase 1: Grid search...")
        grid_results = self._grid_search(stock_data, profile)
        if grid_results:
            best_score = grid_results[0]["score"]
            best_weights = grid_results[0]["weights"]
            print(f"  Grid best: score={-best_score:.4f}, weights={grid_results[0]['weights']}")

        # Phase 2: Local refinement via random perturbation
        print("Phase 2: Local refinement...")
        for trial in range(self.n_trials):
            perturbed = self._perturb(best_weights, step=0.02 * max(0.3, 1.0 - trial / self.n_trials))
            normalized = self._normalize(perturbed)

            if not self._check_constraints(normalized, constraints):
                continue

            score = self._objective(normalized, stock_data)
            if score < best_score:
                best_score = score
                best_weights = normalized[:]
                if (trial + 1) % 50 == 0:
                    print(f"  Trial {trial+1}: improved to {-best_score:.4f}")

        # Phase 3: Final evaluation
        w_dict = {FACTOR_NAMES[i]: best_weights[i] for i in range(6)}
        final_metrics = self._full_eval(stock_data, w_dict)
        current_metrics = self._full_eval(stock_data, config["weights"])

        result = {
            "profile": profile,
            "current_weights": config["weights"],
            "optimized_weights": {FACTOR_NAMES[i]: round(best_weights[i], 4) for i in range(6)},
            "current_metrics": current_metrics,
            "optimized_metrics": final_metrics,
            "improvement": {
                "ic_delta": round(final_metrics["avg_abs_ic"] - current_metrics["avg_abs_ic"], 6),
                "sharpe_delta": round(final_metrics["avg_sharpe"] - current_metrics["avg_sharpe"], 6),
                "cvar_delta": round(final_metrics["avg_cvar_ratio"] - current_metrics["avg_cvar_ratio"], 6),
            },
        }

        self._print_result(result)
        return result

    def _grid_search(
        self, stock_data: Dict[str, Dict[str, List[float]]], profile: str
    ) -> List[Dict[str, Any]]:
        results = []
        n_grid = 5  # points per dimension

        # Sample a subset for grid search (too expensive for all)
        grid_samples = {s: d for i, (s, d) in enumerate(stock_data.items()) if i < 30}

        for _ in range(min(self.n_trials, 100)):
            raw = [random.random() for _ in range(6)]
            normalized = self._normalize(raw)
            score = self._objective(normalized, grid_samples)
            results.append({"weights": normalized, "score": score})

        results.sort(key=lambda x: x["score"])
        return results[:5]

    def _perturb(self, weights: List[float], step: float = 0.02) -> List[float]:
        perturbed = []
        for w in weights:
            delta = random.uniform(-step, step)
            perturbed.append(max(0.04, w + delta))
        return perturbed

    def _normalize(self, weights: List[float]) -> List[float]:
        total = sum(abs(w) for w in weights)
        if total < 1e-12:
            n = len(weights)
            return [1.0 / n] * n
        return [abs(w) / total for w in weights]

    def _check_constraints(
        self, weights: List[float], constraints: Dict[str, Tuple[float, float]]
    ) -> bool:
        for name, (lo, hi) in constraints.items():
            idx = FACTOR_NAMES.index(name) if name in FACTOR_NAMES else -1
            if idx >= 0 and not (lo <= weights[idx] <= hi):
                return False
        return True

    def _full_eval(
        self, stock_data: Dict[str, Dict[str, List[float]]], weights: Dict[str, float]
    ) -> Dict[str, float]:
        all_ic = []
        all_sharpe = []
        all_cvar = []
        all_dd = []

        for sym, data in stock_data.items():
            result = self.backtest.run(data, weights)
            if result["n_samples"] < 10:
                continue
            all_ic.append(result["abs_ic"])
            all_sharpe.append(result["sharpe"])
            all_cvar.append(result["cvar_ratio"])
            all_dd.append(result["max_drawdown"])

        return {
            "avg_abs_ic": round(_safe_mean(all_ic, 0.0), 6),
            "avg_sharpe": round(_safe_mean(all_sharpe, 0.0), 6),
            "avg_cvar_ratio": round(_safe_mean(all_cvar, 1.0), 6),
            "avg_max_drawdown": round(_safe_mean(all_dd, 0.0), 6),
            "n_evaluated": len(all_ic),
        }

    def _print_result(self, result: Dict[str, Any]) -> None:
        print("\n" + "=" * 60)
        print(f"Optimization Result: {result['profile']}")
        print("=" * 60)
        print(f"\n{'Factor':<20} {'Current':>10} {'Optimized':>10} {'Delta':>10}")
        print("-" * 55)
        for name in FACTOR_NAMES:
            cur = result["current_weights"][name]
            opt = result["optimized_weights"][name]
            delta = opt - cur
            print(f"{name:<20} {cur:>10.4f} {opt:>10.4f} {delta:>+10.4f}")

        print(f"\n{'Metric':<25} {'Current':>10} {'Optimized':>10} {'Delta':>10}")
        print("-" * 60)
        cm = result["current_metrics"]
        om = result["optimized_metrics"]
        imp = result["improvement"]
        print(f"{'Rank IC (abs)':<25} {cm['avg_abs_ic']:>10.4f} {om['avg_abs_ic']:>10.4f} {imp['ic_delta']:>+10.4f}")
        print(f"{'Sharpe':<25} {cm['avg_sharpe']:>10.4f} {om['avg_sharpe']:>10.4f} {imp['sharpe_delta']:>+10.4f}")
        print(f"{'CVaR Ratio':<25} {cm['avg_cvar_ratio']:>10.4f} {om['avg_cvar_ratio']:>10.4f} {imp['cvar_delta']:>+10.4f}")
        print(f"{'Max Drawdown':<25} {cm['avg_max_drawdown']:>10.4f} {om['avg_max_drawdown']:>10.4f}")
        print(f"{'Stocks Evaluated':<25} {cm['n_evaluated']:>10d} {om['n_evaluated']:>10d}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    import os
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "combined_dataset", "training_data.csv"
    )
    data_path = os.path.normpath(data_path)

    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    optimizer = WeightOptimizer(data_path=data_path, n_stocks=200, n_trials=200)

    results = {}
    for profile in ("aggressive", "moderate", "conservative"):
        results[profile] = optimizer.optimize(profile)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Recommended Weight Changes")
    print("=" * 70)
    for profile, res in results.items():
        print(f"\n--- {profile.upper()} ---")
        for name in FACTOR_NAMES:
            cur = res["current_weights"][name]
            opt = res["optimized_weights"][name]
            if abs(opt - cur) > 0.01:
                print(f"  {name}: {cur:.4f} -> {opt:.4f} ({opt - cur:+.4f})")


if __name__ == "__main__":
    main()
