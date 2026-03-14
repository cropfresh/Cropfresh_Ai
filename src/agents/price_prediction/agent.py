"""
Price Prediction Agent Core
===========================
Hybrid price forecasting using rule-based seasonal averages,
trend analysis, and optional LLM contextual reasoning.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Optional

from loguru import logger

from src.agents.base_agent import AgentConfig, AgentResponse, BaseAgent
from src.memory.state_manager import AgentExecutionState
from src.orchestrator.llm_provider import BaseLLMProvider
from src.tools.agmarknet import AgmarknetPrice, AgmarknetTool

from .analysis import AnalysisMixin
from .models import PricePrediction


class PricePredictionAgent(BaseAgent, AnalysisMixin):
    """
    Hybrid price forecasting agent for Karnataka mandi commodities.
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
        """Forecast price for a commodity."""
        history = await self._get_price_history(commodity, district, days=90)

        if not history:
            logger.warning("No price history for {} in {} — using LLM fallback", commodity, district)
            return await self._llm_based_prediction(commodity, district)

        features = self._extract_features(history, commodity)
        rule_pred = self._rule_based_predict(features, days_ahead)

        # No ML model yet — rule-based only until XGBoost is trained
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

        # 30-day estimate projects current trend forward
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

    async def _llm_based_prediction(
        self,
        commodity: str,
        district: str,
    ) -> PricePrediction:
        """LLM-based fallback when no historical data is available."""
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

        # Hard fallback
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
