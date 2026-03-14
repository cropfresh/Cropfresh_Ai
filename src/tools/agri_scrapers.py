"""
Agricultural Data Scrapers (Proxy)
==================================
This file exists for backward compatibility.
The actual implementation has been modularized into `src.scrapers.agri_scrapers`.
"""

from src.scrapers.agri_scrapers import (
    SOURCE_URLS,
    AgriculturalDataAPI,
    DataSource,
    ENAMScraper,
    GovScheme,
    IMDWeatherScraper,
    MandiPrice,
    NewsArticle,
    RSSNewsScraper,
    WeatherData,
)

__all__ = [
    "MandiPrice",
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

