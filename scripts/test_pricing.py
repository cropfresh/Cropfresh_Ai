"""
Live Integration Test — Advanced Pricing Agent
===============================================
End-to-end validation against real APIs (Agmarknet, OpenWeatherMap, DuckDuckGo).
Run with:  uv run python scripts/test_pricing.py

Requires the following env vars (or change the constants below):
    AGMARKNET_API_KEY  — data.gov.in Variety-wise Daily Market Prices API key
    OPENWEATHER_API_KEY — OpenWeatherMap free-tier API key

The test does NOT require an LLM — the heuristic engine will produce results
even without one. Set GROQ_API_KEY to enable LLM analysis.
"""

import asyncio
import os

from loguru import logger

from src.agents.pricing_agent import FarmerContext, PricingAgent
from src.tools.ml_forecaster import PriceForecaster
from src.tools.news_sentiment import NewsSentimentScraper
from src.tools.weather import WeatherTool

# ── Config ────────────────────────────────────────────────────────────────────
# Keys are loaded from .env automatically by the OS / python-dotenv.
# Set AGMARKNET_API_KEY and WEATHER_API_KEY in your .env file.

AGMARKNET_KEY   = os.getenv("AGMARKNET_API_KEY", "")
OPENWEATHER_KEY = os.getenv("WEATHER_API_KEY", "")

TEST_COMMODITY = "Tomato"
TEST_LOCATION  = "Kolar"
TEST_QTY_KG    = 250


# ── Individual signal tests ───────────────────────────────────────────────────

async def test_weather_api() -> None:
    print("\n── Weather API ──────────────────────────────────────────────────────")
    tool    = WeatherTool(api_key=OPENWEATHER_KEY, use_mock=False)
    forecast = await tool.get_forecast(TEST_LOCATION, days=7)
    c = forecast.current
    print(f"  Location:     {forecast.location}")
    print(f"  Condition:    {c.condition}")
    print(f"  Temperature:  {c.temperature_c:.1f}°C")
    print(f"  Humidity:     {c.humidity_pct:.0f}%")
    print(f"  Rain (1h):    {c.rainfall_mm:.1f} mm")
    print(f"  7-day advisory: {forecast.planting_advisory}")
    assert forecast.location, "Weather location should not be empty"
    print("  ✓ WeatherTool live test passed")


async def test_news_sentiment() -> None:
    print("\n── News Sentiment ───────────────────────────────────────────────────")
    scraper   = NewsSentimentScraper()
    sentiment = await scraper.get_sentiment(TEST_COMMODITY, TEST_LOCATION)
    print(f"  Commodity:  {sentiment.commodity}")
    print(f"  Label:      {sentiment.label}")
    print(f"  Score:      {sentiment.score:+.3f}")
    print(f"  Articles:   {sentiment.sample_count}")
    if sentiment.top_headlines:
        print(f"  Headlines:")
        for h in sentiment.top_headlines:
            print(f"    • {h}")
    assert sentiment.label in ("bullish", "bearish", "neutral")
    print("  ✓ News Sentiment test passed")


async def test_ml_forecaster() -> None:
    print("\n── ML Forecaster ────────────────────────────────────────────────────")
    # Synthetic 30-day rising prices
    prices = [20.0 + i * 0.3 for i in range(30)]
    forecaster = PriceForecaster()
    result = forecaster.forecast_from_raw(prices, TEST_COMMODITY, TEST_LOCATION, horizon=7)
    print(f"  Commodity:      {result.commodity}")
    print(f"  Current avg:    ₹{result.current_avg_price:.2f}/kg")
    print(f"  Trend:          {result.trend_direction} ({result.trend_pct_change:+.1f}%)")
    print(f"  7-day forecast: ₹{result.forecasted_prices[-1]:.2f}/kg")
    print(f"  Confidence:     {result.confidence:.0%}")
    print(f"  Models used:    {', '.join(result.models_used)}")
    assert result.trend_direction in ("rising", "stable", "falling")
    print("  ✓ ML Forecaster test passed")


async def test_full_pipeline() -> None:
    print("\n── Full Pricing Agent Pipeline ──────────────────────────────────────")
    agent = PricingAgent(
        llm=None,                               # No LLM key needed for heuristic mode
        agmarknet_api_key=AGMARKNET_KEY,
        weather_api_key=OPENWEATHER_KEY,
        use_mock=False,
    )

    ctx = FarmerContext(has_cold_storage=False, financial_urgency="normal")
    rec = await agent.get_recommendation(
        commodity=TEST_COMMODITY,
        location=TEST_LOCATION,
        quantity_kg=TEST_QTY_KG,
        farmer_context=ctx,
    )

    print(f"  Commodity:        {rec.commodity} @ {rec.location}")
    print(f"  Current Price:    ₹{rec.current_price:.2f}/kg")
    print(f"  Market Range:     ₹{rec.market_min:.1f} – ₹{rec.market_max:.1f}/kg")
    print(f"  Action:           {rec.recommended_action.upper()} (conf={rec.confidence:.0%})")
    print(f"  Reason:           {rec.reason}")
    print(f"  Forecast (7d):    ₹{rec.forecasted_price_7d:.2f}/kg [{rec.forecast_trend}]" if rec.forecasted_price_7d else "  Forecast: unavailable")
    print(f"  News sentiment:   {rec.news_sentiment} (score={rec.news_score:+.2f})" if rec.news_sentiment else "  News: unavailable")
    print(f"  Weather advisory: {rec.weather_advisory}" if rec.weather_advisory else "  Weather: unavailable")
    print(f"  AISP for buyer:   ₹{rec.aisp_per_kg:.2f}/kg")
    print(f"  Data source:      {rec.data_source}")
    print(f"  Timestamp:        {rec.timestamp}")
    assert rec.commodity == TEST_COMMODITY
    assert rec.recommended_action in ("sell", "hold", "wait", "unknown")
    print("  ✓ Full pipeline test passed")


# ── Trend test ────────────────────────────────────────────────────────────────

async def test_trend() -> None:
    print("\n── Price Trend (30-day) ─────────────────────────────────────────────")
    agent = PricingAgent(agmarknet_api_key=AGMARKNET_KEY, use_mock=False)
    trend = await agent.get_price_trend(TEST_COMMODITY, district=TEST_LOCATION, days=30)
    print(f"  Trend:        {trend['trend']}")
    print(f"  Volatility:   {trend['volatility_index']:.4f}")
    print(f"  7d avg:       ₹{trend['7d_avg']:.2f}/kg")
    print(f"  30d avg:      ₹{trend['30d_avg']:.2f}/kg")
    print(f"  Forecast 7d:  ₹{trend.get('forecasted_7d', 'N/A')}")
    print(f"  Recommendation: {trend['recommendation']}")
    print("  ✓ Trend test passed")


# ── Runner ────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("=" * 70)
    print("  CropFresh Advanced Pricing Agent — Live Integration Test")
    print("=" * 70)

    await test_weather_api()
    await test_news_sentiment()
    await test_ml_forecaster()
    await test_full_pipeline()
    await test_trend()

    print("\n" + "=" * 70)
    print("  All integration tests completed successfully ✓")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
