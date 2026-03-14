"""
IMD Weather Proxy
=================
Proxy to avoid duplicating code between src/tools and src/scrapers.
Features are imported directly from src/scrapers/imd_weather.
"""

from src.scrapers.imd_weather import (
    WeatherCondition,
    AlertSeverity,
    CurrentWeather,
    DailyForecast,
    WeatherForecast,
    WeatherAlert,
    AgroAdvisory,
    IMDWeatherClient,
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
