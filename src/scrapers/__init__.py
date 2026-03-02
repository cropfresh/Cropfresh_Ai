"""CropFresh AI Scrapers — Production-grade data collection."""

from src.scrapers.base_scraper import (
    FetcherType,
    ScrapeResult,
    ScraperHealth,
    ScraplingBaseScraper,
)
from src.scrapers.agri_scrapers import (
    AgriculturalDataAPI,
    AgmarknetScraper as AgmarknetWebScraper,  # Scrapling-based (agri_scrapers.py)
    DataSource,
    ENAMScraper,
    GovScheme,
    IMDWeatherScraper,
    MandiPrice as AgriMandiPrice,
    NewsArticle,
    RSSNewsScraper,
    WeatherData,
)
from src.scrapers.agmarknet import (
    AgmarknetScraper,         # Task 13 enhanced scraper (ScraplingBaseScraper)
    AgmarknetTool,
    MandiPrice,
    get_agmarknet_scraper,
)
from src.scrapers.scraper_scheduler import (
    ScraperScheduler,
    get_scraper_scheduler,
)
from src.scrapers.aikosha_client import (
    AIKoshaCategory,
    AIKoshaClient,
    AIKoshaDataset,
)

__all__ = [
    # Base
    "ScraplingBaseScraper",
    "FetcherType",
    "ScrapeResult",
    "ScraperHealth",
    # Scrapers (agri_scrapers.py — Scrapling-based)
    "AgmarknetWebScraper",
    "ENAMScraper",
    "IMDWeatherScraper",
    "RSSNewsScraper",
    "AgriculturalDataAPI",
    # Agmarknet (Task 13 — enhanced scraper + API tool)
    "AgmarknetScraper",
    "AgmarknetTool",
    "MandiPrice",
    "get_agmarknet_scraper",
    # Scheduler (Task 13)
    "ScraperScheduler",
    "get_scraper_scheduler",
    # AI Kosha
    "AIKoshaClient",
    "AIKoshaDataset",
    "AIKoshaCategory",
    # Models
    "AgriMandiPrice",
    "WeatherData",
    "GovScheme",
    "NewsArticle",
    "DataSource",
]
