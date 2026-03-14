"""
Agricultural Data API
=====================
Unified API for all agricultural data sources (Production-Grade).
"""

from typing import Any, Optional

from loguru import logger

from src.scrapers.agmarknet.client import AgmarknetScraper
from src.scrapers.base_scraper import ScrapeResult
from src.scrapers.state_portals import StatePortalScraper

from .enam import ENAMScraper
from .imd import IMDWeatherScraper
from .models import NewsArticle
from .rss import RSSNewsScraper


class AgriculturalDataAPI:
    """
    Unified API for all agricultural data sources (Production-Grade).

    Uses Scrapling-powered scrapers with circuit breakers,
    retry logic, and automatic fallback chains.
    """

    def __init__(self, llm_provider: Optional[Any] = None):
        self.agmarknet = AgmarknetScraper(llm_provider=llm_provider)
        self.enam = ENAMScraper(llm_provider=llm_provider)
        self.state_portals = StatePortalScraper(llm_provider=llm_provider)
        self.imd = IMDWeatherScraper()
        self.news = RSSNewsScraper()

    async def get_mandi_prices(
        self,
        commodity: str,
        state: Optional[str] = None,
        district: Optional[str] = None,
        source: str = "agmarknet",
    ) -> ScrapeResult:
        """Get mandi prices with automatic fallback."""
        if source == "agmarknet":
            result = await self.agmarknet.scrape(commodity=commodity, state=state, district=district)
            if not result.success:
                logger.warning("Agmarknet failed — falling back to eNAM")
                result = await self.enam.scrape(commodity=commodity, state=state, district=district)
            return result
        elif source == "enam":
            result = await self.enam.scrape(commodity=commodity, state=state, district=district)
            if not result.success:
                logger.warning("eNAM failed — falling back to Agmarknet")
                result = await self.agmarknet.scrape(commodity=commodity, state=state, district=district)
            return result
        else:
            logger.warning(f"Unknown price source: {source}")
            return ScrapeResult(source=source, url="", error=f"Unknown source: {source}")

    async def get_weather(
        self,
        state: str,
        district: Optional[str] = None,
        include_advisory: bool = False,
    ) -> ScrapeResult:
        """Get weather forecast."""
        return await self.imd.scrape(
            state=state, district=district, include_advisory=include_advisory
        )

    async def get_news(
        self,
        source: str = "rural_voice",
        limit: int = 10,
    ) -> list[NewsArticle]:
        """Get agricultural news."""
        return await self.news.get_news(source, limit)

    async def get_state_data(self, state: str = "UP", **kwargs) -> ScrapeResult:
        """Get agricultural alerts and schemes for a state."""
        return await self.state_portals.scrape(state=state, **kwargs)

    def get_all_health(self) -> dict[str, dict]:
        """Get health status of all scrapers."""
        return {
            "agmarknet": self.agmarknet.get_health().model_dump(),
            "enam": self.enam.get_health().model_dump(),
            "state_portals": self.state_portals.get_health().model_dump(),
            "imd_weather": self.imd.get_health().model_dump(),
        }

    async def close_all(self):
        """Clean up all scrapers."""
        await self.agmarknet.close()
        await self.enam.close()
        await self.state_portals.close()
        await self.imd.close()
