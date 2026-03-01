"""
Price Prediction Agent (CV-PP)
==============================
Hybrid price forecasting using rule-based seasonal averages,
trend analysis, and optional LLM contextual reasoning.

Prediction flow:
  1. Fetch 90-day price history from AgmarknetTool
  2. Extract features (moving averages, momentum, seasonal multiplier)
  3. Rule-based prediction: 7d_avg × seasonal_mult × momentum_factor
  4. Trend classification via linear regression slope
  5. Sell/hold recommendation from predicted-vs-current delta
  6. LLM fallback when no historical data is available
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Optional

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from src.agents.base_agent import AgentConfig, AgentResponse, BaseAgent
from src.memory.state_manager import AgentExecutionState
from src.orchestrator.llm_provider import BaseLLMProvider
from src.tools.agmarknet import AgmarknetTool, AgmarknetPrice


# * ═══════════════════════════════════════════════════════════════
# * CONSTANTS
# * ═══════════════════════════════════════════════════════════════

# * Karnataka seasonal price multipliers by crop and month (1=Jan … 12=Dec)
SEASONAL_CALENDAR: dict[str, dict[int, float]] = {
    "tomato": {
        1: 1.0, 2: 0.9, 3: 0.8, 4: 0.7, 5: 1.3, 6: 1.4,
        7: 1.2, 8: 1.0, 9: 0.9, 10: 0.85, 11: 0.8, 12: 0.9,
    },
    "onion": {
        1: 1.0, 2: 1.1, 3: 1.2, 4: 1.3, 5: 1.1, 6: 0.9,
        7: 0.8, 8: 0.7, 9: 0.8, 10: 1.0, 11: 0.7, 12: 0.8,
    },
    "potato": {
        1: 1.0, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.9, 6: 1.0,
        7: 1.1, 8: 1.2, 9: 1.1, 10: 1.0, 11: 0.95, 12: 1.0,
    },
    "cauliflower": {
        1: 0.9, 2: 0.8, 3: 0.7, 4: 0.8, 5: 1.1, 6: 1.2,
        7: 1.1, 8: 1.0, 9: 1.0, 10: 0.9, 11: 0.85, 12: 0.9,
    },
    "carrot": {
        1: 1.0, 2: 1.0, 3: 0.9, 4: 0.8, 5: 0.9, 6: 1.0,
        7: 1.0, 8: 1.1, 9: 1.2, 10: 1.1, 11: 1.0, 12: 1.0,
    },
    "okra": {
        1: 1.1, 2: 1.0, 3: 0.9, 4: 0.85, 5: 1.0, 6: 1.1,
        7: 1.2, 8: 1.1, 9: 1.0, 10: 0.95, 11: 0.9, 12: 1.0,
    },
}

# * Labels describing the seasonal position of prices
SEASONAL_LABEL_MAP = {
    "peak_harvest": "Peak harvest season — high supply depresses prices.",
    "off_season": "Off-season — reduced supply supports higher prices.",
    "normal": "Normal seasonal conditions.",
}

# * Recommendation thresholds
SELL_NOW_DELTA = 0.05    # predicted ≥ 5% above current → sell now
HOLD_3D_DELTA  = -0.03   # predicted 0–5% above current → hold 3 days
HOLD_7D_DELTA  = -0.08   # predicted < −3% → hold 7 days


# * ═══════════════════════════════════════════════════════════════
# * DATA MODELS
# * ═══════════════════════════════════════════════════════════════

class PricePrediction(BaseModel):
    """Complete price prediction result for a commodity."""
    commodity: str
    district: str
    current_price: float                # ₹/kg — latest observed
    predicted_price_7d: float           # ₹/kg — 7-day forward estimate
    predicted_price_30d: float          # ₹/kg — 30-day forward estimate (trend-adjusted)
    confidence: float                   # 0.0–1.0
    trend: str                          # 'rising' | 'falling' | 'stable'
    trend_strength: float               # 0.0–1.0
    seasonal_factor: str                # 'peak_harvest' | 'off_season' | 'normal'
    factors: list[str] = Field(default_factory=list)   # human-readable drivers
    recommendation: str                 # 'sell_now' | 'hold_3d' | 'hold_7d' | 'hold_30d'
    data_source: str                    # 'historical' | 'model' | 'llm_estimate'
    predicted_at: datetime = Field(default_factory=datetime.now)


# * ═══════════════════════════════════════════════════════════════
# * AGENT
# * ═══════════════════════════════════════════════════════════════

class PricePredictionAgent(BaseAgent):
    """
    Hybrid price forecasting agent for Karnataka mandi commodities.

    Approach:
      1. Rule-based: seasonal avg × moving avg × momentum (always available)
      2. LLM analysis: narrative reasoning about price trend (contextual)

    Usage:
        agent = PricePredictionAgent(llm=provider, agmarknet_tool=tool)
        await agent.initialize()
        prediction = await agent.predict("tomato", district="Kolar")
    """

    def __init__(
        self,
        llm: Optional[BaseLLMProvider] = None,
        agmarknet_tool: Optional[AgmarknetTool] = None,
        **kwargs: Any,
    ):
        config = AgentConfig(
            name="price_prediction",
            description="Hybrid price forecasting using seasonal rules + trend analysis + LLM reasoning",
            max_retries=1,
            temperature=0.3,
            max_tokens=300,
            kb_categories=["agronomy", "market"],
        )
        super().__init__(config=config, llm=llm, **kwargs)
        self.agmarknet = agmarknet_tool or AgmarknetTool()

    def _get_system_prompt(self, context: Optional[dict] = None) -> str:
        return (
            "You are CropFresh's Price Prediction Agent for Karnataka agricultural markets. "
            "Analyse the provided market context and give a concise JSON price forecast."
        )

    # * ───────────────────────────────────────────────────────────
    # * PUBLIC INTERFACE
    # * ───────────────────────────────────────────────────────────

    async def predict(
        self,
        commodity: str,
        district: str = "Kolar",
        days_ahead: int = 7,
    ) -> PricePrediction:
        """
        Forecast price for a commodity.

        Args:
            commodity: Crop name (e.g. 'tomato', 'onion').
            district:  Karnataka district (default 'Kolar').
            days_ahead: Forecast horizon in days (default 7).

        Returns:
            PricePrediction with trend, recommendation, and factors.
        """
        history = await self._get_price_history(commodity, district, days=90)

        if not history:
            logger.warning("No price history for {} in {} — using LLM fallback", commodity, district)
            return await self._llm_based_prediction(commodity, district)

        features = self._extract_features(history, commodity)
        rule_pred = self._rule_based_predict(features, days_ahead)

        # * No ML model yet — rule-based only until XGBoost is trained
        predicted_7d = rule_pred
        confidence = 0.65
        source = "historical"

        trend, strength = self._analyze_trend(history)
        seasonal = self._get_seasonal_factor(commodity, datetime.now().month)
        current = history[-1].modal_price_per_kg

        recommendation = self._generate_recommendation(
            current=current,
            predicted=predicted_7d,
            trend=trend,
            seasonal=seasonal,
        )
        factors = self._explain_factors(features, trend, seasonal, commodity)

        # * 30-day estimate projects current trend forward
        trend_sign = 1 if trend == "rising" else (-1 if trend == "falling" else 0)
        predicted_30d = round(predicted_7d * (1 + trend_sign * strength * 0.15), 2)

        return PricePrediction(
            commodity=commodity,
            district=district,
            current_price=round(current, 2),
            predicted_price_7d=round(predicted_7d, 2),
            predicted_price_30d=predicted_30d,
            confidence=confidence,
            trend=trend,
            trend_strength=round(strength, 3),
            seasonal_factor=seasonal,
            factors=factors,
            recommendation=recommendation,
            data_source=source,
        )

    async def process(
        self,
        query: str,
        context: Optional[dict] = None,
        execution: Optional[AgentExecutionState] = None,
    ) -> AgentResponse:
        """Process a natural-language price prediction query via the supervisor."""
        ctx = context or {}
        commodity = ctx.get("commodity", self._extract_commodity(query))
        district = ctx.get("district", "Kolar")

        try:
            pred = await self.predict(commodity, district)
            content = (
                f"{pred.commodity.title()} price in {pred.district}: "
                f"₹{pred.current_price:.1f}/kg now, "
                f"₹{pred.predicted_price_7d:.1f}/kg in 7 days "
                f"({pred.trend}, confidence {pred.confidence:.0%}). "
                f"Recommendation: {pred.recommendation.replace('_', ' ')}. "
                f"Factors: {'; '.join(pred.factors[:2])}."
            )
            return AgentResponse(
                content=content,
                agent_name=self.name,
                confidence=pred.confidence,
                steps=["price_history_fetch", "feature_extraction", "rule_based_predict"],
            )
        except Exception as exc:
            logger.error("PricePredictionAgent.process failed: {}", exc)
            return AgentResponse(
                content=f"Unable to predict price for {commodity} right now.",
                agent_name=self.name,
                confidence=0.2,
                error=str(exc),
            )

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute structured prediction from API / orchestrator."""
        commodity = input_data.get("commodity", "tomato")
        district = input_data.get("district", "Kolar")
        days_ahead = int(input_data.get("days_ahead", 7))

        pred = await self.predict(commodity, district, days_ahead)
        return {
            "commodity": pred.commodity,
            "district": pred.district,
            "current_price": pred.current_price,
            "predicted_price_7d": pred.predicted_price_7d,
            "predicted_price_30d": pred.predicted_price_30d,
            "confidence": pred.confidence,
            "trend": pred.trend,
            "trend_strength": pred.trend_strength,
            "seasonal_factor": pred.seasonal_factor,
            "factors": pred.factors,
            "recommendation": pred.recommendation,
            "data_source": pred.data_source,
        }

    # * ───────────────────────────────────────────────────────────
    # * PRIVATE HELPERS
    # * ───────────────────────────────────────────────────────────

    async def _get_price_history(
        self,
        commodity: str,
        district: str,
        days: int = 90,
    ) -> list[AgmarknetPrice]:
        """Fetch price history via AgmarknetTool with graceful empty-list fallback."""
        try:
            history = await self.agmarknet.get_historical_prices(
                commodity=commodity.title(),
                state="Karnataka",
                district=district,
                days=days,
            )
            return history
        except Exception as exc:
            logger.warning("Price history fetch failed for {}: {}", commodity, exc)
            return []

    def _extract_features(
        self,
        history: list[AgmarknetPrice],
        commodity: str,
    ) -> dict[str, float]:
        """
        Build feature dict from price history.

        Keys returned:
          avg_7d, avg_14d, avg_30d, momentum_7d, momentum_30d,
          seasonal_multiplier, latest_price.
        """
        prices = [p.modal_price_per_kg for p in history]
        n = len(prices)

        avg_7d = float(np.mean(prices[-7:])) if n >= 7 else float(np.mean(prices))
        avg_14d = float(np.mean(prices[-14:])) if n >= 14 else avg_7d
        avg_30d = float(np.mean(prices[-30:])) if n >= 30 else avg_14d

        # * Momentum = relative price change over window
        momentum_7d = (prices[-1] - prices[-7]) / prices[-7] if n >= 7 and prices[-7] > 0 else 0.0
        momentum_30d = (prices[-1] - prices[-30]) / prices[-30] if n >= 30 and prices[-30] > 0 else 0.0

        month = datetime.now().month
        seasonal_multiplier = self._get_seasonal_multiplier(commodity, month)

        return {
            "avg_7d": avg_7d,
            "avg_14d": avg_14d,
            "avg_30d": avg_30d,
            "momentum_7d": momentum_7d,
            "momentum_30d": momentum_30d,
            "seasonal_multiplier": seasonal_multiplier,
            "latest_price": prices[-1],
        }

    def _rule_based_predict(self, features: dict[str, float], days_ahead: int) -> float:
        """
        Weighted rule-based prediction formula:
          predicted = 7d_avg × (1 + momentum × days/7) × seasonal_multiplier
        """
        avg_7d = features["avg_7d"]
        momentum = features["momentum_7d"]
        seasonal_mult = features["seasonal_multiplier"]

        raw = avg_7d * (1 + momentum * (days_ahead / 7.0))
        predicted = raw * seasonal_mult
        return max(predicted, 0.5)

    def _analyze_trend(self, history: list[AgmarknetPrice]) -> tuple[str, float]:
        """
        Linear regression on the last 14 days of modal prices.

        Returns (direction, strength) where direction ∈ {rising, falling, stable}
        and strength ∈ [0.0, 1.0].
        """
        prices = [p.modal_price_per_kg for p in history[-14:]]
        if len(prices) < 3:
            return "stable", 0.0

        x = np.arange(len(prices), dtype=float)
        slope = float(np.polyfit(x, prices, 1)[0])
        avg = float(np.mean(prices))
        rel_slope = slope / avg if avg > 0 else 0.0

        if rel_slope > 0.02:
            return "rising", min(1.0, rel_slope * 10)
        if rel_slope < -0.02:
            return "falling", min(1.0, abs(rel_slope) * 10)
        return "stable", 0.0

    def _get_seasonal_factor(self, commodity: str, month: int) -> str:
        """Return human-readable seasonal label for a crop in a given month."""
        multiplier = self._get_seasonal_multiplier(commodity, month)
        if multiplier >= 1.15:
            return "off_season"
        if multiplier <= 0.85:
            return "peak_harvest"
        return "normal"

    def _get_seasonal_multiplier(self, commodity: str, month: int) -> float:
        """Return numeric seasonal multiplier (default 1.0 when crop not in calendar)."""
        crop_calendar = SEASONAL_CALENDAR.get(commodity.lower(), {})
        return crop_calendar.get(month, 1.0)

    def _generate_recommendation(
        self,
        current: float,
        predicted: float,
        trend: str,
        seasonal: str,
    ) -> str:
        """
        Determine sell/hold advice from price delta and trend context.

        Logic:
          - predicted ≥ +5% above current  → sell_now
          - predicted +0–5%               → hold_3d (slight upside still expected)
          - trend rising + off_season      → hold_7d (strong upward momentum)
          - otherwise                     → hold_30d (wait for seasonal recovery)
        """
        if current <= 0:
            return "hold_7d"

        delta = (predicted - current) / current

        if delta >= SELL_NOW_DELTA:
            return "sell_now"
        if delta >= HOLD_3D_DELTA:
            return "hold_3d"
        if trend == "rising" and seasonal == "off_season":
            return "hold_7d"
        return "hold_30d"

    def _explain_factors(
        self,
        features: dict[str, float],
        trend: str,
        seasonal: str,
        commodity: str,
    ) -> list[str]:
        """Generate human-readable list of price-driving factors."""
        factors: list[str] = []

        momentum = features.get("momentum_7d", 0.0)
        if momentum > 0.05:
            factors.append(f"Strong upward momentum (+{momentum * 100:.1f}% over 7 days).")
        elif momentum < -0.05:
            factors.append(f"Downward price pressure ({momentum * 100:.1f}% over 7 days).")
        else:
            factors.append("Price stable over last 7 days.")

        if trend == "rising":
            factors.append("14-day trend is rising — buyers increasing demand.")
        elif trend == "falling":
            factors.append("14-day trend is falling — market oversupply likely.")

        if seasonal == "off_season":
            factors.append(f"{commodity.title()} is in off-season — supply constrained, prices elevated.")
        elif seasonal == "peak_harvest":
            factors.append(f"{commodity.title()} is at peak harvest — high supply depresses prices.")

        seasonal_mult = features.get("seasonal_multiplier", 1.0)
        if seasonal_mult > 1.1:
            factors.append(f"Seasonal multiplier {seasonal_mult:.2f}× above baseline.")
        elif seasonal_mult < 0.9:
            factors.append(f"Seasonal multiplier {seasonal_mult:.2f}× below baseline.")

        return factors[:4]

    async def _llm_based_prediction(
        self,
        commodity: str,
        district: str,
    ) -> PricePrediction:
        """
        LLM-based fallback when no historical data is available.

        Returns a PricePrediction with data_source='llm_estimate'.
        """
        month = datetime.now().month
        seasonal = self._get_seasonal_factor(commodity, month)
        seasonal_mult = self._get_seasonal_multiplier(commodity, month)

        if self.llm:
            prompt = (
                f"You are a Karnataka agri-market expert. "
                f"Estimate the current modal price (₹/kg) for {commodity.title()} "
                f"in {district} district. Month: {month}, seasonal factor: {seasonal}. "
                f"Reply with JSON: {{\"price_per_kg\": <float>, \"trend\": \"rising|falling|stable\", "
                f"\"factors\": [\"...\", \"...\"]}}"
            )
            try:
                import json, re
                raw = await self.llm.generate(prompt, max_tokens=150)
                match = re.search(r"\{[\s\S]*\}", raw)
                if match:
                    data = json.loads(match.group())
                    base_price = float(data.get("price_per_kg", 25.0))
                    trend = data.get("trend", "stable")
                    llm_factors = data.get("factors", [])
                    predicted_7d = round(base_price * seasonal_mult, 2)
                    return PricePrediction(
                        commodity=commodity,
                        district=district,
                        current_price=round(base_price, 2),
                        predicted_price_7d=predicted_7d,
                        predicted_price_30d=predicted_7d,
                        confidence=0.45,
                        trend=trend,
                        trend_strength=0.3,
                        seasonal_factor=seasonal,
                        factors=llm_factors[:4] or [f"LLM estimate for {seasonal} season."],
                        recommendation="hold_7d",
                        data_source="llm_estimate",
                    )
            except Exception as exc:
                logger.warning("LLM price estimation failed: {}", exc)

        # * Hard fallback: seasonal-adjusted generic price
        base_price = 25.0
        predicted = round(base_price * seasonal_mult, 2)
        return PricePrediction(
            commodity=commodity,
            district=district,
            current_price=base_price,
            predicted_price_7d=predicted,
            predicted_price_30d=predicted,
            confidence=0.3,
            trend="stable",
            trend_strength=0.0,
            seasonal_factor=seasonal,
            factors=[f"No historical data available. Seasonal estimate ({seasonal})."],
            recommendation="hold_7d",
            data_source="llm_estimate",
        )

    def _extract_commodity(self, query: str) -> str:
        """Extract commodity name from a free-text query string."""
        known = [
            "tomato", "onion", "potato", "carrot", "cauliflower",
            "okra", "capsicum", "cabbage", "beans", "brinjal",
        ]
        query_lower = query.lower()
        for crop in known:
            if crop in query_lower:
                return crop
        return "tomato"
