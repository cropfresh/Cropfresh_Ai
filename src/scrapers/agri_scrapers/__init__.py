"""
Agri Scrapers Package
=====================
Unified interface for Indian agricultural data scraping.
"""

from .constants import SOURCE_URLS, DataSource
from .enam import ENAMScraper
from .imd import IMDWeatherScraper
from .models import GovScheme, MandiPrice, MandiPriceList, NewsArticle, WeatherData
from .rss import RSSNewsScraper

__all__ = [
    "MandiPrice",
    "MandiPriceList",
    "WeatherData",
    "GovScheme",
    "NewsArticle",
    "DataSource",
    "SOURCE_URLS",
    "ENAMScraper",
    "IMDWeatherScraper",
    "RSSNewsScraper",
    "AgriculturalDataAPI",
]


def __getattr__(name: str):
    """Lazily import heavier compatibility surfaces to avoid circular imports."""
    if name == "AgriculturalDataAPI":
        from .api import AgriculturalDataAPI

        return AgriculturalDataAPI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
