"""
Pricing Agent
=============
Dynamic Pricing Engine (DPLE) — real-time market recommendations for farmers.

Signal pipeline (all fetched concurrently):
  ① Agmarknet live mandi prices          → current price anchor
  ② ML ensemble forecast (7-day)         → trend + forecasted price
  ③ OpenWeatherMap current + 7-day forecast → supply-side weather risk
  ④ DuckDuckGo news sentiment            → market sentiment signal

All four signals are injected into an LLM prompt for a final, personalised
recommendation (sell / hold / wait) in plain language.

Canonical usage::

    agent = PricingAgent(
        llm=llm_provider,
        agmarknet_api_key="<data.gov.in key>",
        weather_api_key="<OWM key>",
    )
    rec = await agent.get_recommendation("Tomato", "Kolar", quantity_kg=200)

Author: CropFresh AI Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from src.agents.aisp_calculator import AISPCalculation, calculate_aisp
from src.orchestrator.llm_provider import BaseLLMProvider, LLMMessage
from src.tools.agmarknet import AgmarknetPrice, AgmarknetTool
from src.tools.ml_forecaster import ForecastResult, PriceForecaster
from src.tools.news_sentiment import CommoditySentiment, NewsSentimentScraper
from src.tools.weather import WeatherForecast, WeatherTool

# ── Domain models ─────────────────────────────────────────────────────────────

class FarmerContext(BaseModel):
    """Optional farmer-specific context that personalises the recommendation."""
    has_cold_storage: bool = False
    financial_urgency: str = "normal"   # "urgent" | "normal" | "flexible"
    target_market: Optional[str] = None


class PriceRecommendation(BaseModel):
    """Full price recommendation result returned to callers."""

    commodity: str
    location: str
    current_price: float          # ₹/kg
    current_price_quintal: float  # ₹/quintal

    # Market context
    market_min: float
    market_max: float

    # ML forecast
    forecasted_price_7d: Optional[float] = None
    forecast_trend: Optional[str] = None     # "rising" | "falling" | "stable"
    forecast_confidence: Optional[float] = None

    # External signals
    weather_advisory: Optional[str] = None
    news_sentiment: Optional[str] = None     # "bullish" | "bearish" | "neutral"
    news_score: Optional[float] = None

    # Recommendation
    recommended_action: str                  # "sell" | "hold" | "wait"
    confidence: float
    reason: str
    llm_analysis: Optional[str] = None      # Natural language explanation from LLM

    # AISP (for buyers)
    aisp_per_kg: Optional[float] = None
    aisp_breakdown: Optional[dict] = None

    # Metadata
    data_source: str
    timestamp: datetime


# ── Main agent ────────────────────────────────────────────────────────────────

class PricingAgent:
    """
    Advanced Dynamic Pricing Engine (DPLE) — wires real-time signals into LLM.

    Concurrently fetches: Agmarknet prices, ML forecast, weather, and news.
    Falls back gracefully if any individual signal fails.
    """

    def __init__(
        self,
        llm: Optional[BaseLLMProvider] = None,
        agmarknet_api_key: str = "",
        weather_api_key: str = "",
        # TODO: add e-NAM API key here once access is approved
        use_mock: bool = False,
    ) -> None:
        self.llm = llm
        self.agmarknet  = AgmarknetTool(api_key=agmarknet_api_key)
        self.weather    = WeatherTool(api_key=weather_api_key, use_mock=not weather_api_key)
        self.forecaster = PriceForecaster()
        self.news       = NewsSentimentScraper()
        #! use_mock overrides live APIs — only True in unit tests / CI
        self.use_mock   = use_mock

    async def initialize(self) -> bool:
        """Initialize agent resources."""
        return True

    # ── Public interface ──────────────────────────────────────────────────────

    async def get_recommendation(
        self,
        commodity: str,
        location: str,
        quantity_kg: float = 100,
        asking_price: Optional[float] = None,
        farmer_context: Optional[FarmerContext] = None,
        distance_km: float = 30,
    ) -> PriceRecommendation:
        """
        Full recommendation pipeline — concurrently collects all signals.

        Args:
            commodity:       Crop name (e.g. "Tomato").
            location:        District / market name.
            quantity_kg:     Quantity to sell.
            asking_price:    Farmer's target price — used in heuristic analysis.
            farmer_context:  Optional personalisation (cold storage, urgency).
            distance_km:     Distance to nearest mandi / buyer.

        Returns:
            PriceRecommendation with action, reason, and full signal metadata.
        """
        ctx = farmer_context or FarmerContext()

        # ── Concurrent signal fetch ──────────────────────────────────────────
        prices_task   = self._fetch_prices(commodity, location)
        weather_task  = self.weather.get_forecast(location, days=7)
        news_task     = self.news.get_sentiment(commodity, location)

        prices, weather, sentiment = await asyncio.gather(
            prices_task, weather_task, news_task,
            return_exceptions=True,
        )

        # Guard: treat exceptions as missing data
        prices   = prices   if isinstance(prices,   list)          else []
        weather  = weather  if isinstance(weather,  WeatherForecast) else None
        sentiment= sentiment if isinstance(sentiment, CommoditySentiment) else None

        if not prices:
            return self._empty_recommendation(commodity, location)

        price = prices[0]

        # ── ML forecast (needs historical prices — run after primary fetch) ──
        history = await self._fetch_history(commodity, location)
        forecast: Optional[ForecastResult] = None
        if history:
            raw = [p.modal_price_per_kg for p in history]
            forecast = self.forecaster.forecast_from_raw(raw, commodity, location)

        # ── Heuristic action (fallback / supplement to LLM) ─────────────────
        action, heuristic_conf, heuristic_reason = self._analyze_signals(
            price, quantity_kg, asking_price, forecast, sentiment, ctx
        )

        # ── AISP for buyers ──────────────────────────────────────────────────
        aisp = calculate_aisp(
            farmer_price_per_kg=price.modal_price_per_kg,
            quantity_kg=quantity_kg,
            distance_km=distance_km,
            mandi_modal_per_kg=price.modal_price_per_kg,
            cold_chain=ctx.has_cold_storage,
        )

        # ── LLM natural-language analysis ────────────────────────────────────
        llm_text: Optional[str] = None
        if self.llm:
            llm_text = await self._llm_analysis(
                commodity, location, price, forecast, weather, sentiment, ctx
            )

        return PriceRecommendation(
            commodity=commodity,
            location=location,
            current_price=price.modal_price_per_kg,
            current_price_quintal=price.modal_price,
            market_min=price.min_price / 100,
            market_max=price.max_price / 100,
            forecasted_price_7d=forecast.forecasted_prices[-1] if forecast else None,
            forecast_trend=forecast.trend_direction if forecast else None,
            forecast_confidence=forecast.confidence if forecast else None,
            weather_advisory=weather.planting_advisory if weather else None,
            news_sentiment=sentiment.label if sentiment else None,
            news_score=sentiment.score if sentiment else None,
            recommended_action=action,
            confidence=heuristic_conf,
            reason=heuristic_reason,
            llm_analysis=llm_text,
            aisp_per_kg=aisp.aisp_per_kg,
            aisp_breakdown=aisp.model_dump(),
            data_source="agmarknet_live" if not self.use_mock else "mock",
            timestamp=datetime.now(),
        )

    async def get_price_trend(
        self,
        commodity: str,
        district: str = "Bangalore",
        days: int = 30,
    ) -> dict:
        """
        Analyse historical trend (delegates to forecaster after fetching history).
        Keeps the same return shape for backwards compatibility.
        """
        if days <= 0:
            raise ValueError("days must be > 0")

        history = await self.agmarknet.get_historical_prices(
            commodity=commodity, state="Karnataka", district=district, days=max(days, 30)
        )
        if not history:
            return {"trend": "stable", "volatility_index": 0.0,
                    "7d_avg": 0.0, "30d_avg": 0.0, "recommendation": "hold_3_days"}

        raw = [p.modal_price_per_kg for p in sorted(history, key=lambda x: x.date)]
        forecast = self.forecaster.forecast_from_raw(raw, commodity, district, horizon=7)

        # Keep backward-compat dict shape
        avg_7d  = sum(raw[-7:])  / len(raw[-7:])
        avg_30d = sum(raw[-30:]) / len(raw[-30:])

        rec_map = {"rising": "sell_now", "falling": "hold_7_days", "stable": "hold_3_days"}
        return {
            "trend":            forecast.trend_direction,
            "volatility_index": forecast.volatility_index,
            "7d_avg":           round(avg_7d,  2),
            "30d_avg":          round(avg_30d, 2),
            "forecasted_7d":    forecast.forecasted_prices[-1],
            "recommendation":   rec_map[forecast.trend_direction],
        }

    # Keep the standalone AISP method for direct buyer usage
    def calculate_aisp(self, **kwargs) -> AISPCalculation:
        """Direct AISP calculation — delegates to aisp_calculator module."""
        return calculate_aisp(**kwargs)

    def get_seasonal_adjustment(self, commodity: str, month: int) -> float:
        """Return seasonal multiplier. Delegates to ml_forecaster's table."""
        from datetime import datetime

        from src.tools.ml_forecaster import _seasonal_factor
        if month < 1 or month > 12:
            raise ValueError("month must be 1–12")
        return _seasonal_factor(commodity, datetime(2024, month, 1))

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _fetch_prices(
        self, commodity: str, location: str
    ) -> list[AgmarknetPrice]:
        """Fetch live prices, fall back to mock only if use_mock is set."""
        if self.use_mock or not self.agmarknet.api_key:
            logger.info(f"Using mock prices for {commodity}")
            return self.agmarknet.get_mock_prices(commodity, district=location)
        return await self.agmarknet.get_prices(commodity, "Karnataka", location)

    async def _fetch_history(
        self, commodity: str, location: str, days: int = 30
    ) -> list[AgmarknetPrice]:
        try:
            return await self.agmarknet.get_historical_prices(
                commodity=commodity, state="Karnataka", district=location, days=days
            )
        except Exception as e:
            logger.warning(f"Historical price fetch failed: {e}")
            return []

    def _analyze_signals(
        self,
        price: AgmarknetPrice,
        quantity_kg: float,
        asking_price: Optional[float],
        forecast: Optional[ForecastResult],
        sentiment: Optional[CommoditySentiment],
        ctx: FarmerContext,
    ) -> tuple[str, float, str]:
        """
        Combine market position, ML trend, news sentiment, and farmer context
        into a heuristic (action, confidence, reason) tuple.

        This is intentionally simple — the LLM does the nuanced reasoning.
        """
        modal   = price.modal_price_per_kg
        min_p   = price.min_price / 100
        max_p   = price.max_price / 100
        rng     = max_p - min_p if max_p > min_p else 1
        position = (modal - min_p) / rng             # 0 = at min, 1 = at max

        # Base score from market position (0 = bearish, 1 = bullish)
        score = position

        # Boost from ML trend
        if forecast:
            if forecast.trend_direction == "rising":
                score += 0.2
            elif forecast.trend_direction == "falling":
                score -= 0.2

        # Boost from news sentiment
        if sentiment:
            score += sentiment.score * 0.15

        # Adjust for perishability risk when quantity is large
        if quantity_kg > 500 and not ctx.has_cold_storage:
            score += 0.15   # nudge towards selling to avoid spoilage

        # Urgent cash need → override to sell
        if ctx.financial_urgency == "urgent":
            score += 0.3

        # Clamp
        score = min(max(score, 0.0), 1.0)

        if score > 0.6:
            action = "sell"
            confidence = round(0.6 + score * 0.35, 2)
            reason = (
                f"Price ₹{modal:.1f}/kg is favourable"
                + (f" with a {forecast.trend_direction} ML forecast" if forecast else "")
                + (f" and {sentiment.label} market news" if sentiment else "")
                + "."
            )
        elif score < 0.35:
            action = "hold"
            confidence = round(0.55 + (1 - score) * 0.25, 2)
            reason = (
                f"Price ₹{modal:.1f}/kg is near market lows"
                + (f"; ML forecasts prices are {forecast.trend_direction}" if forecast else "")
                + ". Consider waiting for a better window."
            )
        else:
            action = "wait"
            confidence = 0.55
            reason = (
                f"Price ₹{modal:.1f}/kg is in mid-range. "
                "Monitor market for the next 2-3 days before deciding."
            )

        return action, min(confidence, 0.95), reason

    async def _llm_analysis(
        self,
        commodity: str,
        location: str,
        price: AgmarknetPrice,
        forecast: Optional[ForecastResult],
        weather: Optional[WeatherForecast],
        sentiment: Optional[CommoditySentiment],
        ctx: FarmerContext,
    ) -> str:
        """Build a rich, signal-aware LLM prompt and return the response."""
        forecast_section = ""
        if forecast:
            forecast_section = (
                f"\nML Price Forecast (7-day): ₹{forecast.forecasted_prices[-1]:.1f}/kg "
                f"| Trend: {forecast.trend_direction} ({forecast.trend_pct_change:+.1f}%) "
                f"| Confidence: {forecast.confidence:.0%}"
            )

        weather_section = ""
        if weather:
            c = weather.current
            weather_section = (
                f"\nWeather ({location}): {c.condition}, {c.temperature_c:.1f}°C, "
                f"Humidity {c.humidity_pct:.0f}%, Rainfall {c.rainfall_mm:.1f}mm"
                f"\nForecast advisory: {weather.planting_advisory}"
            )

        news_section = ""
        if sentiment and sentiment.top_headlines:
            headlines = "; ".join(sentiment.top_headlines[:3])
            news_section = (
                f"\nNews Sentiment: {sentiment.label} (score={sentiment.score:+.2f})"
                f"\nTop headlines: {headlines}"
            )

        farmer_section = (
            f"\nFarmer context: cold_storage={ctx.has_cold_storage}, "
            f"urgency={ctx.financial_urgency}"
        )

        messages = [
            LLMMessage(
                role="system",
                content=(
                    "You are CropFresh AI's senior Pricing Agent. "
                    "You receive real-time market data, ML price forecasts, "
                    "weather signals, and news sentiment. "
                    "Give a concise, data-driven sell/hold/wait recommendation "
                    "in 2-3 sentences. Use ₹ for prices. Be specific."
                ),
            ),
            LLMMessage(
                role="user",
                content=(
                    f"Commodity: {commodity} | Location: {location}\n"
                    f"Current mandi price: ₹{price.modal_price:.0f}/quintal "
                    f"(₹{price.modal_price_per_kg:.1f}/kg)"
                    f"\nMin: ₹{price.min_price:.0f} | Max: ₹{price.max_price:.0f}"
                    f" | Date: {price.date.strftime('%d %b %Y')}"
                    f"{forecast_section}{weather_section}{news_section}{farmer_section}"
                    "\n\nShould the farmer sell now, hold, or wait?"
                ),
            ),
        ]

        try:
            response = await self.llm.generate(messages, temperature=0.5, max_tokens=250)
            return response.content
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return ""

    @staticmethod
    def _empty_recommendation(commodity: str, location: str) -> PriceRecommendation:
        return PriceRecommendation(
            commodity=commodity,
            location=location,
            current_price=0,
            current_price_quintal=0,
            market_min=0,
            market_max=0,
            recommended_action="unknown",
            confidence=0,
            reason="Could not fetch market prices. Please check API connectivity.",
            data_source="none",
            timestamp=datetime.now(),
        )
