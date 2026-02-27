"""CropFresh AI Scrapers — Production-grade data collection."""

from src.scrapers.base_scraper import (
    FetcherType,
    ScrapeResult,
    ScraperHealth,
    ScraplingBaseScraper,
)
from src.scrapers.agri_scrapers import (
    AgriculturalDataAPI,
    AgmarknetScraper,
    DataSource,
    ENAMScraper,
    GovScheme,
    IMDWeatherScraper,
    MandiPrice,
    NewsArticle,
    RSSNewsScraper,
    WeatherData,
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
    # Scrapers
    "AgmarknetScraper",
    "ENAMScraper",
    "IMDWeatherScraper",
    "RSSNewsScraper",
    "AgriculturalDataAPI",
    # AI Kosha
    "AIKoshaClient",
    "AIKoshaDataset",
    "AIKoshaCategory",
    # Models
    "MandiPrice",
    "WeatherData",
    "GovScheme",
    "NewsArticle",
    "DataSource",
]
