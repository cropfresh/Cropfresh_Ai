"""
IMD Weather Proxy
=================
Proxy to avoid duplicating code between src/tools and src/scrapers.
Features are imported directly from src/scrapers/imd_weather.
"""

from src.scrapers.imd_weather import (
    AgroAdvisory,
    AlertSeverity,
    CurrentWeather,
    DailyForecast,
    IMDWeatherClient,
    WeatherAlert,
    WeatherCondition,
    WeatherForecast,
    get_imd_client,
)

__all__ = [
    "WeatherCondition",
    "AlertSeverity",
    "CurrentWeather",
    "DailyForecast",
    "WeatherForecast",
    "WeatherAlert",
    "AgroAdvisory",
    "IMDWeatherClient",
    "get_imd_client",
]
