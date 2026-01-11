"""
Agricultural Data Scrapers
==========================
Pre-built scrapers for Indian agricultural data sources.

Supported Sources:
- eNAM (National Agriculture Market) - Live mandi prices
- Agmarknet - Historical commodity prices
- IMD Agrimet - Weather advisories
- Government schemes portals
- Agricultural news RSS feeds

Author: CropFresh AI Team
Version: 1.0.0
"""

import asyncio
from datetime import datetime, date
from typing import Optional, List
from enum import Enum

from loguru import logger
from pydantic import BaseModel, Field

from src.agents.web_scraping_agent import WebScrapingAgent, ScrapingConfig
from src.agents.browser_agent import BrowserAgent, BrowserAction, ActionType


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


SOURCE_URLS = {
    DataSource.ENAM: "https://enam.gov.in/web/dashboard/trade-data",
    DataSource.AGMARKNET: "https://agmarknet.gov.in/SearchCmmMkt.aspx",
    DataSource.IMD: "https://mausam.imd.gov.in/",
    DataSource.DATA_GOV: "https://data.gov.in/catalog/current-daily-price-various-commodities-various-markets-mandi",
    DataSource.PM_KISAN: "https://pmkisan.gov.in/",
    DataSource.RURAL_VOICE: "https://ruralvoice.in/rss/latest-posts",
}


# ============================================================================
# Scrapers
# ============================================================================

class AgmarknetScraper:
    """
    Scraper for Agmarknet portal - Daily commodity prices.
    
    URL: https://agmarknet.gov.in/SearchCmmMkt.aspx
    Data: Daily arrivals and prices from 7000+ mandis
    """
    
    BASE_URL = "https://agmarknet.gov.in/SearchCmmMkt.aspx"
    
    def __init__(self):
        self.scraper = WebScrapingAgent()
        self._initialized = False
    
    async def initialize(self):
        """Initialize scraper."""
        if not self._initialized:
            await self.scraper.initialize(ScrapingConfig(
                headless=True,
                stealth=True,
                wait_for_load_state="networkidle",
            ))
            self._initialized = True
    
    async def close(self):
        """Clean up resources."""
        await self.scraper.close()
        self._initialized = False
    
    async def get_prices(
        self,
        commodity: str,
        state: Optional[str] = None,
        market: Optional[str] = None,
        date_from: Optional[date] = None,
    ) -> List[MandiPrice]:
        """
        Get commodity prices from Agmarknet.
        
        Args:
            commodity: Commodity name (e.g., "Tomato", "Onion", "Rice")
            state: Optional state filter
            market: Optional market/mandi name
            date_from: Start date for price data
            
        Returns:
            List of MandiPrice records
        """
        await self.initialize()
        
        try:
            # For now, scrape the search page and extract available data
            result = await self.scraper.scrape_to_markdown(self.BASE_URL)
            
            if not result.success:
                logger.error("Failed to fetch Agmarknet: {}", result.error)
                return []
            
            # Parse the markdown content to extract prices
            # This is a simplified extraction - in production, use CSS selectors
            prices = self._parse_price_data(result.markdown, commodity, state)
            
            logger.info("Fetched {} prices for {} from Agmarknet", len(prices), commodity)
            return prices
            
        except Exception as e:
            logger.error("Agmarknet scraping failed: {}", str(e))
            return []
    
    def _parse_price_data(
        self,
        content: str,
        commodity: str,
        state: Optional[str],
    ) -> List[MandiPrice]:
        """Parse price data from content."""
        # This is a placeholder - actual implementation would parse
        # the HTML tables using CSS selectors
        prices = []
        
        # Example mock data for testing
        if "tomato" in commodity.lower() or "टमाटर" in commodity:
            prices.append(MandiPrice(
                commodity="Tomato",
                variety="Local",
                mandi="Azadpur",
                district="Delhi",
                state="Delhi",
                min_price=1500.0,
                max_price=2500.0,
                modal_price=2000.0,
                date=date.today(),
                source="agmarknet",
            ))
        
        return prices


class ENAMScraper:
    """
    Scraper for eNAM portal - Live trading data.
    
    URL: https://enam.gov.in
    Data: Real-time bids, trade data from 1000+ mandis
    Note: Some features require login
    """
    
    BASE_URL = "https://enam.gov.in"
    DASHBOARD_URL = "https://enam.gov.in/web/dashboard/trade-data"
    
    def __init__(self):
        self.browser = BrowserAgent(headless=True, stealth=True)
        self._initialized = False
    
    async def initialize(self):
        """Initialize browser session."""
        if not self._initialized:
            await self.browser.start_session()
            self._initialized = True
    
    async def close(self):
        """Clean up resources."""
        await self.browser.close_session()
        self._initialized = False
    
    async def get_live_prices(
        self,
        commodity: Optional[str] = None,
        state: Optional[str] = None,
    ) -> List[MandiPrice]:
        """
        Get live prices from eNAM dashboard.
        
        Args:
            commodity: Optional commodity filter
            state: Optional state filter
            
        Returns:
            List of MandiPrice records
        """
        await self.initialize()
        
        try:
            # Navigate to dashboard
            await self.browser.execute_action(BrowserAction(
                action=ActionType.GOTO,
                value=self.DASHBOARD_URL,
            ))
            
            # Wait for data to load
            await asyncio.sleep(3)
            
            # Get page content
            content = await self.browser.get_page_markdown()
            
            if not content:
                logger.error("Failed to get eNAM page content")
                return []
            
            # Parse prices from content
            prices = self._parse_enam_data(content, commodity, state)
            
            logger.info("Fetched {} live prices from eNAM", len(prices))
            return prices
            
        except Exception as e:
            logger.error("eNAM scraping failed: {}", str(e))
            return []
    
    def _parse_enam_data(
        self,
        content: str,
        commodity: Optional[str],
        state: Optional[str],
    ) -> List[MandiPrice]:
        """Parse eNAM dashboard data."""
        prices = []
        
        # Placeholder - actual implementation would parse the dashboard tables
        # eNAM uses JavaScript-heavy rendering, so we need the full browser
        
        return prices


class IMDWeatherScraper:
    """
    Scraper for IMD weather and agricultural advisories.
    
    URLs:
    - mausam.imd.gov.in - Weather forecasts
    - imdagrimet.gov.in - Agricultural advisories
    """
    
    WEATHER_URL = "https://mausam.imd.gov.in/"
    AGRIMET_URL = "https://imdagrimet.gov.in/"
    
    def __init__(self):
        self.scraper = WebScrapingAgent()
        self._initialized = False
    
    async def initialize(self):
        """Initialize scraper."""
        if not self._initialized:
            await self.scraper.initialize()
            self._initialized = True
    
    async def close(self):
        """Clean up resources."""
        await self.scraper.close()
        self._initialized = False
    
    async def get_weather(
        self,
        state: str,
        district: Optional[str] = None,
    ) -> List[WeatherData]:
        """
        Get weather forecast for a location.
        
        Args:
            state: State name
            district: Optional district name
            
        Returns:
            List of WeatherData records
        """
        await self.initialize()
        
        try:
            result = await self.scraper.scrape_to_markdown(self.WEATHER_URL)
            
            if not result.success:
                logger.error("Failed to fetch IMD weather: {}", result.error)
                return []
            
            # Parse weather data
            weather = self._parse_weather(result.markdown, state, district)
            return weather
            
        except Exception as e:
            logger.error("IMD weather scraping failed: {}", str(e))
            return []
    
    async def get_agro_advisory(
        self,
        state: str,
        district: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get agricultural advisory for a location.
        
        Args:
            state: State name
            district: Optional district name
            
        Returns:
            Advisory text if available
        """
        await self.initialize()
        
        try:
            result = await self.scraper.scrape_to_markdown(self.AGRIMET_URL)
            
            if not result.success:
                return None
            
            # Extract advisory section
            # Placeholder - would parse actual advisory content
            return None
            
        except Exception as e:
            logger.error("IMD advisory scraping failed: {}", str(e))
            return None
    
    def _parse_weather(
        self,
        content: str,
        state: str,
        district: Optional[str],
    ) -> List[WeatherData]:
        """Parse weather data from content."""
        weather = []
        
        # Placeholder - would extract actual weather data
        weather.append(WeatherData(
            location=district or state,
            district=district or "Unknown",
            state=state,
            temperature_celsius=28.0,
            humidity_percent=65.0,
            weather_condition="Partly Cloudy",
            forecast_date=date.today(),
        ))
        
        return weather


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
            logger.warning("feedparser not installed")
            self._feedparser = None
    
    async def get_news(
        self,
        source: str = "rural_voice",
        limit: int = 10,
    ) -> List[NewsArticle]:
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
            logger.error("Unknown news source: {}", source)
            return []
        
        try:
            feed = self._feedparser.parse(self.RSS_FEEDS[source])
            articles = []
            
            for entry in feed.entries[:limit]:
                articles.append(NewsArticle(
                    title=entry.get("title", "Untitled"),
                    summary=entry.get("summary", ""),
                    url=entry.get("link", ""),
                    source=source,
                    published_date=datetime.now(),  # Would parse entry.published
                ))
            
            logger.info("Fetched {} articles from {}", len(articles), source)
            return articles
            
        except Exception as e:
            logger.error("RSS feed parsing failed: {}", str(e))
            return []


# ============================================================================
# Unified Agricultural Data API
# ============================================================================

class AgriculturalDataAPI:
    """
    Unified API for all agricultural data sources.
    
    Usage:
        api = AgriculturalDataAPI()
        
        # Get mandi prices
        prices = await api.get_mandi_prices("Tomato", state="Karnataka")
        
        # Get weather
        weather = await api.get_weather("Karnataka", "Bangalore Rural")
        
        # Get news
        news = await api.get_news(limit=5)
    """
    
    def __init__(self):
        self.agmarknet = AgmarknetScraper()
        self.enam = ENAMScraper()
        self.imd = IMDWeatherScraper()
        self.news = RSSNewsScraper()
    
    async def get_mandi_prices(
        self,
        commodity: str,
        state: Optional[str] = None,
        source: str = "agmarknet",
    ) -> List[MandiPrice]:
        """Get mandi prices from specified source."""
        if source == "agmarknet":
            return await self.agmarknet.get_prices(commodity, state)
        elif source == "enam":
            return await self.enam.get_live_prices(commodity, state)
        else:
            logger.warning("Unknown price source: {}", source)
            return []
    
    async def get_weather(
        self,
        state: str,
        district: Optional[str] = None,
    ) -> List[WeatherData]:
        """Get weather forecast."""
        return await self.imd.get_weather(state, district)
    
    async def get_news(
        self,
        source: str = "rural_voice",
        limit: int = 10,
    ) -> List[NewsArticle]:
        """Get agricultural news."""
        return await self.news.get_news(source, limit)
    
    async def close_all(self):
        """Clean up all scrapers."""
        await self.agmarknet.close()
        await self.enam.close()
        await self.imd.close()
