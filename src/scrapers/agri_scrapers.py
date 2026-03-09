"""
Agricultural Data Scrapers (Production-Grade)
==============================================
Scrapling-powered scrapers for Indian agricultural data sources.

Supported Sources:
- eNAM (National Agriculture Market) - Live mandi prices
- Agmarknet - Historical commodity prices
- IMD Agrimet - Weather advisories
- Government schemes portals
- Agricultural news RSS feeds

Powered by: Scrapling (adaptive parsing + anti-bot bypass)
Author: CropFresh AI Team
Version: 2.0.0
"""

import asyncio
import re
import time
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field

from src.scrapers.base_scraper import (
    FetcherType,
    ScrapeResult,
    ScraplingBaseScraper,
)
from src.agents.web_scraping_agent import WebScrapingAgent
from src.scrapers.state_portals import StatePortalScraper
from src.scrapers.agmarknet.client import AgmarknetScraper


# ============================================================================
# Pydantic Schemas for Agricultural Data
# ============================================================================


class MandiPrice(BaseModel):
    """Mandi (market) price data structure."""

    commodity: str
    variety: Optional[str] = None
    mandi: str
    district: Optional[str] = None
    state: str
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    modal_price: float
    unit: str = "Rs/Quintal"
    date: date
    source: str = "unknown"


class MandiPriceList(BaseModel):
    """Wrapper for array of Mandi prices returned by LLM extraction."""
    prices: list[MandiPrice]


class WeatherData(BaseModel):
    """Weather data structure."""

    location: str
    district: str
    state: str
    temperature_celsius: Optional[float] = None
    humidity_percent: Optional[float] = None
    rainfall_mm: Optional[float] = None
    weather_condition: Optional[str] = None
    forecast_date: date
    advisory: Optional[str] = None
    source: str = "IMD"


class GovScheme(BaseModel):
    """Government scheme information."""

    name: str
    department: str
    description: str
    eligibility: Optional[str] = None
    benefits: Optional[str] = None
    application_url: Optional[str] = None
    deadline: Optional[date] = None
    state: str = "All India"


class NewsArticle(BaseModel):
    """Agricultural news article."""

    title: str
    summary: Optional[str] = None
    url: str
    source: str
    published_date: Optional[datetime] = None
    category: Optional[str] = None


# ============================================================================
# Data Source Configurations
# ============================================================================


class DataSource(str, Enum):
    """Available data sources."""

    ENAM = "enam"
    AGMARKNET = "agmarknet"
    IMD = "imd"
    DATA_GOV = "data_gov"
    PM_KISAN = "pm_kisan"
    RURAL_VOICE = "rural_voice"
    AI_KOSHA = "ai_kosha"


SOURCE_URLS = {
    DataSource.ENAM: "https://enam.gov.in/web/dashboard/trade-data",
    DataSource.AGMARKNET: "https://agmarknet.gov.in/SearchCmmMkt.aspx",
    DataSource.IMD: "https://mausam.imd.gov.in/",
    DataSource.DATA_GOV: "https://data.gov.in/catalog/current-daily-price-various-commodities-various-markets-mandi",
    DataSource.PM_KISAN: "https://pmkisan.gov.in/",
    DataSource.RURAL_VOICE: "https://ruralvoice.in/rss/latest-posts",
    DataSource.AI_KOSHA: "https://indiaai.gov.in/ai-kosha",
}


# ============================================================================
# Scrapling-Powered Scrapers
# ============================================================================


class ENAMScraper(ScraplingBaseScraper):
    """
    Scraper for eNAM portal — Live trading data.

    URL: https://enam.gov.in
    Data: Real-time bids, trade data from 1000+ mandis

    Uses Scrapling's StealthyFetcher for anti-bot bypass.
    """

    name = "enam"
    base_url = "https://enam.gov.in"
    fetcher_type = FetcherType.STEALTHY  # Anti-bot bypass
    cache_ttl_seconds = 180  # 3 min — live trading data
    rate_limit_delay = 3.0  # eNAM has stricter rate limits

    DASHBOARD_URL = "https://enam.gov.in/web/dashboard/trade-data"

    def __init__(self, llm_provider: Optional[Any] = None):
        super().__init__()
        self.web_agent = WebScrapingAgent(llm_provider=llm_provider) if llm_provider else None

    async def scrape(
        self,
        commodity: Optional[str] = None,
        state: Optional[str] = None,
        **kwargs,
    ) -> ScrapeResult:
        """
        Get live prices from eNAM dashboard.

        Args:
            commodity: Optional commodity filter
            state: Optional state filter

        Returns:
            ScrapeResult with MandiPrice records
        """
        start_time = time.time()

        try:
            # Fetch with StealthyFetcher (bypasses anti-bot)
            page = await self.fetch(self.DASHBOARD_URL)

            # Parse live trading data
            prices = self._parse_enam_data(page, commodity, state)

            if not prices and self.web_agent:
                logger.info("eNAM CSS extraction failed — falling back to LLM WebScrapingAgent")
                prices = await self._fallback_llm_extract(page, commodity, state)

            duration_ms = (time.time() - start_time) * 1000
            return self.build_result(
                url=self.DASHBOARD_URL,
                data=[p.model_dump() for p in prices],
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"eNAM scraping failed: {e}")
            return self.build_result(
                url=self.DASHBOARD_URL,
                data=[],
                error=str(e),
                duration_ms=duration_ms,
            )

    def _parse_enam_data(
        self,
        page: Any,
        commodity: Optional[str],
        state: Optional[str],
    ) -> list[MandiPrice]:
        """
        Parse eNAM dashboard data using Scrapling selectors.

        eNAM's dashboard renders data with JavaScript, so StealthyFetcher
        handles the rendering before we parse.
        """
        prices: list[MandiPrice] = []

        try:
            # eNAM trade data table
            rows = page.css("table.table tbody tr")

            for row in rows:
                cells = row.css("td::text").getall()
                if len(cells) >= 6:
                    try:
                        row_commodity = cells[0].strip()
                        row_state = cells[1].strip()

                        if commodity and commodity.lower() not in row_commodity.lower():
                            continue
                        if state and state.lower() not in row_state.lower():
                            continue

                        prices.append(
                            MandiPrice(
                                commodity=row_commodity,
                                mandi=cells[2].strip(),
                                state=row_state,
                                min_price=self._safe_float(cells[3]),
                                max_price=self._safe_float(cells[4]),
                                modal_price=self._safe_float(cells[5]) or 0.0,
                                date=date.today(),
                                source="enam",
                            )
                        )
                    except (ValueError, IndexError):
                        continue

        except Exception as e:
            logger.warning(f"eNAM table parsing failed: {e}")

        if not prices:
            logger.warning("No live eNAM data parsed")

        return prices

    async def _fallback_llm_extract(
        self, page: Any, commodity: Optional[str], state: Optional[str]
    ) -> list[MandiPrice]:
        if not self.web_agent:
            return []
            
        html_content = ""
        if hasattr(page, "body"):
            html_content = page.body.decode("utf-8", "ignore")
        elif hasattr(page, "html"):
            html_content = page.html
            
        if not html_content:
            return []
            
        instruction = "Extract all live trade prices from the eNAM dashboard table."
        result = await self.web_agent.extract_with_schema(
            html_content=html_content,
            url=self.DASHBOARD_URL,
            schema=MandiPriceList,
            instruction=instruction
        )
        
        fallback_prices = []
        if result.success and isinstance(result.extracted_data, dict):
            for item in result.extracted_data.get("prices", []):
                try:
                    c = item.get("commodity", "")
                    s = item.get("state", "")
                    
                    if commodity and commodity.lower() not in c.lower(): continue
                    if state and state.lower() not in s.lower(): continue
                    
                    item["source"] = "enam_llm"
                    fallback_prices.append(MandiPrice(**item))
                except Exception as e:
                    logger.debug(f"LLM item parse error: {e}")
                    
        return fallback_prices

    def _safe_float(self, value: str) -> Optional[float]:
        """Safely convert string to float."""
        try:
            cleaned = re.sub(r"[^\d.]", "", value.strip())
            return float(cleaned) if cleaned else None
        except (ValueError, AttributeError):
            return None




class IMDWeatherScraper(ScraplingBaseScraper):
    """
    Scraper for IMD weather and agricultural advisories.

    URLs:
    - mausam.imd.gov.in - Weather forecasts
    - imdagrimet.gov.in - Agricultural advisories

    Uses Scrapling's basic Fetcher (no anti-bot needed).
    """

    name = "imd_weather"
    base_url = "https://mausam.imd.gov.in"
    fetcher_type = FetcherType.BASIC
    cache_ttl_seconds = 1800  # 30 min — weather changes slowly
    rate_limit_delay = 2.0

    WEATHER_URL = "https://mausam.imd.gov.in/"
    AGRIMET_URL = "https://imdagrimet.gov.in/"

    async def scrape(
        self,
        state: str = "Karnataka",
        district: Optional[str] = None,
        include_advisory: bool = False,
        **kwargs,
    ) -> ScrapeResult:
        """
        Get weather forecast for a location.

        Args:
            state: State name
            district: Optional district name
            include_advisory: Whether to fetch agro advisory too

        Returns:
            ScrapeResult with WeatherData records
        """
        start_time = time.time()

        try:
            page = await self.fetch(self.WEATHER_URL)
            weather = self._parse_weather(page, state, district)

            # Optionally fetch agro advisory
            advisory = None
            if include_advisory:
                try:
                    advisory_page = await self.fetch(self.AGRIMET_URL)
                    advisory = self._parse_advisory(advisory_page, state, district)
                    if advisory and weather:
                        weather[0].advisory = advisory
                except Exception as e:
                    logger.debug(f"Advisory fetch failed (non-critical): {e}")

            duration_ms = (time.time() - start_time) * 1000
            return self.build_result(
                url=self.WEATHER_URL,
                data=[w.model_dump() for w in weather],
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"IMD weather scraping failed: {e}")
            return self.build_result(
                url=self.WEATHER_URL,
                data=[],
                error=str(e),
                duration_ms=duration_ms,
            )

    def _parse_weather(
        self,
        page: Any,
        state: str,
        district: Optional[str],
    ) -> list[WeatherData]:
        """Parse weather data from IMD page using Scrapling selectors."""
        weather: list[WeatherData] = []

        try:
            # Try to extract weather info using Scrapling
            temp_el = page.css(".temperature::text").get()
            humidity_el = page.css(".humidity::text").get()
            condition_el = page.css(".weather-condition::text").get()

            if temp_el or humidity_el:
                weather.append(
                    WeatherData(
                        location=district or state,
                        district=district or "State-wide",
                        state=state,
                        temperature_celsius=self._extract_number(temp_el),
                        humidity_percent=self._extract_number(humidity_el),
                        weather_condition=condition_el,
                        forecast_date=date.today(),
                    )
                )
        except Exception as e:
            logger.debug(f"Weather element extraction failed: {e}")

        if not weather:
            logger.warning("No weather data parsed")

        return weather

    def _parse_advisory(
        self,
        page: Any,
        state: str,
        district: Optional[str],
    ) -> Optional[str]:
        """Parse agricultural advisory from IMD agrimet page."""
        try:
            advisory_el = page.css(".advisory-text::text").get()
            if advisory_el:
                return advisory_el.strip()

            # Try broader selector
            advisory_el = page.find_by_text("advisory", first_match=True)
            if advisory_el:
                return advisory_el.get_all_text().strip()[:500]
        except Exception:
            pass
        return None

    def _extract_number(self, text: Optional[str]) -> Optional[float]:
        """Extract a number from text like '28°C' or '65%'."""
        if not text:
            return None
        match = re.search(r"[\d.]+", text)
        return float(match.group()) if match else None


class RSSNewsScraper:
    """
    Scraper for agricultural news via RSS feeds.

    Sources:
    - Rural Voice
    - Krishak Jagat
    - Agri Farming
    """

    RSS_FEEDS = {
        "rural_voice": "https://ruralvoice.in/rss/latest-posts",
        "agri_farming": "https://agrifarming.in/feed",
    }

    def __init__(self):
        try:
            import feedparser

            self._feedparser = feedparser
        except ImportError:
            logger.warning("feedparser not installed — install via: pip install feedparser")
            self._feedparser = None

    async def get_news(
        self,
        source: str = "rural_voice",
        limit: int = 10,
    ) -> list[NewsArticle]:
        """
        Get agricultural news articles from RSS feed.

        Args:
            source: News source key
            limit: Maximum articles to return

        Returns:
            List of NewsArticle records
        """
        if not self._feedparser:
            logger.error("feedparser not available")
            return []

        if source not in self.RSS_FEEDS:
            logger.error(f"Unknown news source: {source}")
            return []

        try:
            feed = self._feedparser.parse(self.RSS_FEEDS[source])
            articles = []

            for entry in feed.entries[:limit]:
                articles.append(
                    NewsArticle(
                        title=entry.get("title", "Untitled"),
                        summary=entry.get("summary", ""),
                        url=entry.get("link", ""),
                        source=source,
                        published_date=datetime.now(),
                    )
                )

            logger.info(f"Fetched {len(articles)} articles from {source}")
            return articles

        except Exception as e:
            logger.error(f"RSS feed parsing failed: {e}")
            return []


# ============================================================================
# Unified Agricultural Data API
# ============================================================================


class AgriculturalDataAPI:
    """
    Unified API for all agricultural data sources (Production-Grade).

    Uses Scrapling-powered scrapers with circuit breakers,
    retry logic, and automatic fallback chains.

    Usage:
        api = AgriculturalDataAPI()

        # Get mandi prices
        result = await api.get_mandi_prices("Tomato", state="Karnataka")

        # Get weather
        result = await api.get_weather("Karnataka", "Bangalore Rural")

        # Get all scraper health status
        health = api.get_all_health()
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
        """
        Get mandi prices with automatic fallback.

        Fallback chain: agmarknet -> enam
        """
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
