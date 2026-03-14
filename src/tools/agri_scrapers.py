"""
Agricultural Data Scrapers (Proxy)
==================================
This file exists for backward compatibility.
The actual implementation has been modularized into `src.scrapers.agri_scrapers`.
"""

from src.scrapers.agri_scrapers import (
    MandiPrice,
    WeatherData,
    GovScheme,
    NewsArticle,
    DataSource,
    SOURCE_URLS,
    ENAMScraper,
    IMDWeatherScraper,
    RSSNewsScraper,
    AgriculturalDataAPI,
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

