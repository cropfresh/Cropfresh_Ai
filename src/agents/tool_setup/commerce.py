"""Commerce tool registration for agent-facing tool registries."""

from __future__ import annotations

from loguru import logger

from src.tools.agmarknet import AgmarknetTool
from src.tools.ml_forecaster import PriceForecaster
from src.tools.news_sentiment import NewsSentimentScraper
from src.tools.registry import ToolRegistry


async def _agmarknet(
    commodity: str,
    state: str = "Karnataka",
    district: str | None = None,
    market: str | None = None,
    limit: int = 20,
) -> dict:
    prices = await AgmarknetTool().get_prices(commodity, state, district, market, limit)
    return {"records": [price.model_dump(mode="json") for price in prices]}


def _ml_forecaster(
    commodity: str,
    location: str,
    historical_prices: list[float],
    horizon: int = 7,
) -> dict:
    forecast = PriceForecaster().forecast_from_raw(historical_prices, commodity, location, horizon)
    return forecast.__dict__


async def _news_sentiment(
    commodity: str,
    location: str = "India",
    max_articles: int = 10,
) -> dict:
    sentiment = await NewsSentimentScraper().get_sentiment(commodity, location, max_articles)
    return sentiment.__dict__


def register_commerce_tools(registry: ToolRegistry) -> None:
    """Register commerce and market-data tools."""
    try:
        registry.add_tool(
            _agmarknet,
            name="agmarknet",
            description="Fetch mandi prices from Agmarknet for a commodity and market.",
            category="commerce",
        )
        registry.add_tool(
            _ml_forecaster,
            name="ml_forecaster",
            description="Forecast commodity prices from a recent historical series.",
            category="commerce",
        )
        registry.add_tool(
            _news_sentiment,
            name="news_sentiment",
            description="Estimate bullish or bearish commodity sentiment from recent news.",
            category="commerce",
        )
    except Exception as exc:
        logger.debug("Commerce tool registration skipped: {}", exc)
