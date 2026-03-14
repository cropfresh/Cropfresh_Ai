"""
eNAM Client Main Module
"""

from typing import Any, Optional

from loguru import logger

from .cache import ENAMCacheManager
from .mock_data import get_mock_prices_data, get_mock_trend_data
from .models import MandiPrice, MarketSummary, PriceTrend


class ENAMClient:
    """
    eNAM API Client for real-time mandi prices.

    Connects to the Electronic National Agriculture Market platform
    for live commodity prices across 1,000+ mandis in India.
    """

    def __init__(
        self,
        api_key: str = "",
        cache_ttl: int = 300,
        use_mock: bool = True,
    ):
        self.api_key = api_key
        self.use_mock = use_mock or not api_key
        self.cache_manager = ENAMCacheManager(ttl=cache_ttl)

        if self.use_mock:
            logger.info("ENAMClient initialized in MOCK mode")
        else:
            logger.info("ENAMClient initialized with live API")

    async def get_live_prices(
        self,
        commodity: str,
        state: str,
        district: Optional[str] = None,
        market: Optional[str] = None,
        limit: int = 20,
    ) -> list[MandiPrice]:
        """Fetch live prices from eNAM or mock data."""
        cache_key = self.cache_manager.get_cache_key("prices", commodity, state, district, market)
        cached = self.cache_manager.get(cache_key)
        if cached:
            return cached

        if self.use_mock:
            prices = get_mock_prices_data(commodity, state, district, limit)
        else:
            # We delay import to avoid circular dependencies if api_fetch relies on this
            from .api_fetch import fetch_live_prices
            prices = await fetch_live_prices(
                self.api_key, commodity, state, district, market, limit
            )

        self.cache_manager.set(cache_key, prices)
        return prices

    async def get_price_trends(
        self,
        commodity: str,
        state: str,
        market: Optional[str] = None,
        days: int = 30,
    ) -> PriceTrend:
        """Get price trends over time."""
        cache_key = self.cache_manager.get_cache_key("trend", commodity, state, market, days)
        cached = self.cache_manager.get(cache_key)
        if cached:
            return cached

        if self.use_mock:
            trend = get_mock_trend_data(commodity, state, market)
        else:
            from .trends import fetch_price_trends
            trend = await fetch_price_trends(
                self, commodity, state, market, days
            )

        self.cache_manager.set(cache_key, trend)
        return trend

    async def get_market_summary(
        self,
        commodity: str,
        state: str,
    ) -> MarketSummary:
        """Get market summary for a commodity across all mandis in a state."""
        from .trends import get_market_summary
        return await get_market_summary(self, commodity, state)

    def get_data_freshness(self) -> dict[str, Any]:
        """Get data freshness indicators."""
        stats = self.cache_manager.get_freshness_stats()

        from datetime import datetime
        stats["mode"] = "mock" if self.use_mock else "live"
        stats["checked_at"] = datetime.now().isoformat()
        return stats


# Singleton instance
_enam_client: Optional[ENAMClient] = None

def get_enam_client(api_key: str = "", use_mock: bool = True) -> ENAMClient:
    """Get or create singleton eNAM client instance."""
    global _enam_client
    if _enam_client is None:
        _enam_client = ENAMClient(api_key=api_key, use_mock=use_mock)
    return _enam_client
