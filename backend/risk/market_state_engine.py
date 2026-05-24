"""Market State Engine 2.0: regime-switching + macro-factor driver."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _softmax(logits: List[float]) -> List[float]:
    if not logits:
        return []
    top = max(logits)
    exps = [math.exp(x - top) for x in logits]
    total = sum(exps)
    if total <= 1e-12:
        n = len(logits)
        return [1.0 / n for _ in logits]
    return [x / total for x in exps]


class MarketStateEngine2:
    """Regime inference with lightweight HMM-style filtering."""

    REGIMES = ("bull", "sideways", "bear", "crisis")

    REGIME_PARAMS = {
        "bull": {"mu": 0.0012, "sigma": 0.010, "downside": 0.42, "stress": 0.20},
        "sideways": {"mu": 0.0002, "sigma": 0.013, "downside": 0.50, "stress": 0.42},
        "bear": {"mu": -0.0010, "sigma": 0.018, "downside": 0.62, "stress": 0.70},
        "crisis": {"mu": -0.0030, "sigma": 0.028, "downside": 0.78, "stress": 0.93},
    }

    # Row: previous regime, col: next regime.
    TRANSITION = {
        "bull": {"bull": 0.77, "sideways": 0.18, "bear": 0.04, "crisis": 0.01},
        "sideways": {"bull": 0.24, "sideways": 0.55, "bear": 0.17, "crisis": 0.04},
        "bear": {"bull": 0.08, "sideways": 0.24, "bear": 0.56, "crisis": 0.12},
        "crisis": {"bull": 0.03, "sideways": 0.12, "bear": 0.34, "crisis": 0.51},
    }

    REGIME_LABELS = {
        "bull": "Bull Expansion",
        "sideways": "Range / Transition",
        "bear": "Bear Compression",
        "crisis": "Crisis / Liquidity Stress",
    }

    REGIME_WEIGHT_ADJUST = {
        "bull": {
            "sentiment": +0.03,
            "trend": +0.02,
            "volatility": -0.03,
            "event": -0.01,
            "uncertainty": -0.02,
            "market_regime": +0.01,
            "reasoning": -0.01,
        },
        "sideways": {
            "sentiment": +0.00,
            "trend": -0.01,
            "volatility": +0.01,
            "event": +0.00,
            "uncertainty": +0.01,
            "market_regime": -0.01,
            "reasoning": +0.00,
        },
        "bear": {
            "sentiment": -0.01,
            "trend": +0.03,
            "volatility": +0.04,
            "event": +0.02,
            "uncertainty": +0.02,
            "market_regime": +0.00,
            "reasoning": +0.02,
        },
        "crisis": {
            "sentiment": -0.02,
            "trend": +0.03,
            "volatility": +0.08,
            "event": +0.03,
            "uncertainty": +0.05,
            "market_regime": -0.01,
            "reasoning": +0.04,
        },
    }

    def extract_macro_factors(
        self,
        inference_result: Optional[Dict[str, Any]] = None,
        multi_model_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        inference_result = inference_result or {}
        multi_model_result = multi_model_result or {}

        raw = {}
        for key in ("macro_factors", "macro", "macro_inputs"):
            payload = multi_model_result.get(key)
            if isinstance(payload, dict):
                raw.update(payload)

        market_env = multi_model_result.get("market_environment", {})
        if isinstance(market_env, dict):
            raw.update(market_env)

        market_state = multi_model_result.get("market_state", {})
        if isinstance(market_state, dict):
            raw["market_confidence"] = market_state.get("confidence", raw.get("market_confidence"))

        summary_conf = _to_float((inference_result.get("summary", {}) or {}).get("confidence"), 0.5)

        # Normalize to [0,1], where larger means more macro stress.
        rate_stress = _clip(abs(_to_float(raw.get("rate_change", raw.get("rate_delta", 0.0)))) / 0.007)
        credit_stress = _clip(abs(_to_float(raw.get("credit_spread", raw.get("credit_spread_change", 0.0)))) / 0.03)
        inflation_stress = _clip(abs(_to_float(raw.get("inflation_surprise", 0.0))) / 0.015)
        fx_stress = _clip(abs(_to_float(raw.get("usd_cny_change", raw.get("fx_change", 0.0)))) / 0.015)
        vix_stress = _clip(_to_float(raw.get("vix", raw.get("vol_index", 18.0)), 18.0) / 45.0)
        liquidity_stress = _clip(_to_float(raw.get("liquidity_stress", 0.0)))
        policy_uncertainty = _clip(_to_float(raw.get("policy_uncertainty", 0.35)))

        # If there is no explicit macro payload, back off to a mild prior.
        explicit_macro_keys = (
            "rate_change", "rate_delta", "credit_spread", "credit_spread_change",
            "inflation_surprise", "usd_cny_change", "fx_change", "vix", "vol_index",
            "liquidity_stress", "policy_uncertainty",
        )
        explicit_count = sum(1 for k in explicit_macro_keys if k in raw)
        if explicit_count == 0:
            base = _clip(0.45 + 0.20 * (1.0 - summary_conf))
            rate_stress = base
            credit_stress = base
            inflation_stress = base
            fx_stress = base
            vix_stress = base
            liquidity_stress = base
            policy_uncertainty = base

        out = {
            "rate_stress": round(rate_stress, 6),
            "credit_stress": round(credit_stress, 6),
            "inflation_stress": round(inflation_stress, 6),
            "fx_stress": round(fx_stress, 6),
            "vix_stress": round(vix_stress, 6),
            "liquidity_stress": round(liquidity_stress, 6),
            "policy_uncertainty": round(policy_uncertainty, 6),
            "inference_confidence": round(summary_conf, 6),
            "explicit_macro_count": explicit_count,
        }
        out["macro_stress"] = round(self._macro_stress(out), 6)
        return out

    def infer_regime(
        self,
        returns: List[float],
        macro_factors: Dict[str, float],
        previous_regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        clean = [r for r in returns if math.isfinite(_to_float(r))]
        if not clean:
            clean = [0.0]

        mu_obs = sum(clean) / len(clean)
        downside_prob = len([r for r in clean if r < 0]) / max(len(clean), 1)
        sigma_obs = math.sqrt(sum((r - mu_obs) ** 2 for r in clean) / max(len(clean) - 1, 1))
        macro_stress = self._macro_stress(macro_factors)

        prev = previous_regime if previous_regime in self.REGIMES else "sideways"
        priors = self.TRANSITION.get(prev, {})
        prior_vec = [max(1e-8, _to_float(priors.get(regime), 1.0 / len(self.REGIMES))) for regime in self.REGIMES]
        prior_log = [math.log(p) for p in prior_vec]

        logits: List[float] = []
        for regime in self.REGIMES:
            params = self.REGIME_PARAMS[regime]
            stress_k = params["stress"]
            mu = params["mu"] - (macro_stress - 0.45) * 0.0018 * (1.0 + stress_k)
            sigma = max(1e-4, params["sigma"] * (1.0 + 0.8 * macro_stress))

            ll = 0.0
            for r in clean:
                z = (r - mu) / sigma
                ll += -0.5 * (z ** 2) - math.log(sigma)

            # Match downside profile + observed sigma profile.
            downside_gap = abs(downside_prob - params["downside"])
            sigma_gap = abs(sigma_obs - sigma) / max(sigma, 1e-6)
            ll -= 8.0 * downside_gap + 1.2 * sigma_gap

            # Macro stress alignment: risk-off regimes preferred under high stress.
            ll -= 2.5 * abs(macro_stress - stress_k)
            logits.append(ll)

        post = _softmax([prior_log[i] + logits[i] for i in range(len(self.REGIMES))])
        idx = max(range(len(post)), key=lambda i: post[i])
        regime = self.REGIMES[idx]
        confidence = post[idx]

        stress_score = 0.0
        for i, reg in enumerate(self.REGIMES):
            stress_score += post[i] * self.REGIME_PARAMS[reg]["stress"]
        stress_score = _clip(0.72 * stress_score + 0.28 * macro_stress)

        return {
            "regime": regime,
            "regime_label": self.REGIME_LABELS.get(regime, regime),
            "confidence": round(confidence, 6),
            "posterior": {self.REGIMES[i]: round(post[i], 6) for i in range(len(self.REGIMES))},
            "macro_stress": round(macro_stress, 6),
            "state_stress": round(stress_score, 6),
            "observations": {
                "mu_obs": round(mu_obs, 6),
                "sigma_obs": round(sigma_obs, 6),
                "downside_prob": round(downside_prob, 6),
                "sample_size": len(clean),
            },
            "macro_factors": macro_factors,
        }

    def adjusted_factor_weights(self, base_weights: Dict[str, float], regime_info: Dict[str, Any]) -> Dict[str, Any]:
        regime = str(regime_info.get("regime", "sideways"))
        regime_adj = self.REGIME_WEIGHT_ADJUST.get(regime, self.REGIME_WEIGHT_ADJUST["sideways"])
        macro_stress = _to_float(regime_info.get("macro_stress"), 0.5)
        state_conf = _to_float(regime_info.get("confidence"), 0.5)
        blend = _clip(0.45 + 0.40 * state_conf + 0.25 * abs(macro_stress - 0.5))

        out = {}
        for factor, base in base_weights.items():
            delta = _to_float(regime_adj.get(factor), 0.0) * blend
            # Under high stress, force extra focus on volatility/uncertainty/event.
            if factor in {"volatility", "uncertainty", "event", "reasoning"}:
                delta += 0.02 * max(0.0, macro_stress - 0.62)
            if factor in {"sentiment", "trend"}:
                delta -= 0.01 * max(0.0, macro_stress - 0.72)
            out[factor] = max(0.04, _to_float(base) + delta)

        total = sum(out.values())
        if total <= 1e-12:
            n = len(out)
            out = {k: 1.0 / n for k in out}
        else:
            out = {k: v / total for k, v in out.items()}

        return {
            "weights": {k: round(v, 6) for k, v in out.items()},
            "regime": regime,
            "blend": round(blend, 6),
            "macro_stress": round(macro_stress, 6),
        }

    def threshold_shift_points(self, regime_info: Dict[str, Any]) -> Dict[str, float]:
        regime = str(regime_info.get("regime", "sideways"))
        macro_stress = _to_float(regime_info.get("macro_stress"), 0.5)
        confidence = _to_float(regime_info.get("confidence"), 0.5)

        base_shift = {
            "bull": +3.0,
            "sideways": 0.0,
            "bear": -4.5,
            "crisis": -8.0,
        }.get(regime, 0.0)

        stress_shift = (macro_stress - 0.5) * 10.0
        conf_boost = 0.6 + 0.8 * confidence
        shift = (base_shift - stress_shift) * conf_boost

        return {
            "low": round(shift * 0.90, 4),
            "medium": round(shift * 1.00, 4),
            "high": round(shift * 1.08, 4),
        }

    def regime_risk_score(self, regime_info: Dict[str, Any], fallback: float = 0.58) -> float:
        regime = str(regime_info.get("regime", "sideways"))
        confidence = _to_float(regime_info.get("confidence"), 0.5)
        state_stress = _clip(_to_float(regime_info.get("state_stress"), fallback))
        base = {
            "bull": 0.22,
            "sideways": 0.52,
            "bear": 0.78,
            "crisis": 0.94,
        }.get(regime, fallback)
        return _clip(0.55 * base + 0.35 * state_stress + 0.10 * confidence)

    def _macro_stress(self, macro_factors: Dict[str, float]) -> float:
        return _clip(
            0.18 * _clip(_to_float(macro_factors.get("rate_stress"), 0.5)) +
            0.18 * _clip(_to_float(macro_factors.get("credit_stress"), 0.5)) +
            0.14 * _clip(_to_float(macro_factors.get("inflation_stress"), 0.5)) +
            0.10 * _clip(_to_float(macro_factors.get("fx_stress"), 0.5)) +
            0.16 * _clip(_to_float(macro_factors.get("vix_stress"), 0.5)) +
            0.14 * _clip(_to_float(macro_factors.get("liquidity_stress"), 0.5)) +
            0.10 * _clip(_to_float(macro_factors.get("policy_uncertainty"), 0.5))
        )


market_state_engine_2 = MarketStateEngine2()
