"""CropFresh AI Scrapers — Production-grade data collection."""

from src.scrapers.agmarknet import (
    AgmarknetParser,
    AgmarknetScraper,  # Task 13 enhanced scraper (ScraplingBaseScraper)
)
from src.scrapers.agmarknet_api import AgmarknetTool
from src.scrapers.agri_scrapers import (
    AgriculturalDataAPI,
    DataSource,
    ENAMScraper,
    GovScheme,
    IMDWeatherScraper,
    NewsArticle,
    RSSNewsScraper,
    WeatherData,
)
from src.scrapers.agri_scrapers import (
    MandiPrice as AgriMandiPrice,
)
from src.scrapers.aikosha_client import (
    AIKoshaCategory,
    AIKoshaClient,
    AIKoshaDataset,
)
from src.scrapers.base_scraper import (
    FetcherType,
    ScrapeResult,
    ScraperHealth,
    ScraplingBaseScraper,
)
from src.scrapers.scraper_scheduler import (
    ScraperScheduler,
    get_scraper_scheduler,
)

__all__ = [
    # Base
    "ScraplingBaseScraper",
    "FetcherType",
    "ScrapeResult",
    "ScraperHealth",
    # Scrapers (agri_scrapers — modular package)
    "ENAMScraper",
    "IMDWeatherScraper",
    "RSSNewsScraper",
    "AgriculturalDataAPI",
    # Agmarknet (Task 13 — enhanced scraper + API tool)
    "AgmarknetScraper",
    "AgmarknetParser",
    "AgmarknetTool",
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
