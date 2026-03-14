"""
IMD Weather Client Main Module
"""

import httpx
from typing import Optional
from loguru import logger

from .models import (
    CurrentWeather,
    WeatherForecast,
    WeatherAlert,
    AgroAdvisory,
    WeatherCondition,
)
from .constants import DISTRICT_COORDS, OWM_API_BASE
from .cache import IMDCacheManager
from .mock_data import (
    get_mock_current_weather,
    get_mock_forecast_data,
    get_mock_alerts_data,
)
from .advisory import generate_agro_advisory_data


class IMDWeatherClient:
    """
    IMD Weather Client for agricultural weather data.
    """
    
    def __init__(
        self,
        imd_api_key: str = "",
        owm_api_key: str = "",
        cache_ttl: int = 1800,
        use_mock: bool = True,
    ):
        self.imd_api_key = imd_api_key
        self.owm_api_key = owm_api_key
        self.use_mock = use_mock or (not imd_api_key and not owm_api_key)
        self.cache_manager = IMDCacheManager(ttl=cache_ttl)
        
        if self.use_mock:
            logger.info("IMDWeatherClient initialized in MOCK mode")
        else:
            logger.info("IMDWeatherClient initialized with live API")
            
    async def get_current_weather(
        self,
        state: str,
        district: str,
    ) -> CurrentWeather:
        """Get current weather for a district."""
        cache_key = self.cache_manager.get_cache_key("current", state, district)
        cached = self.cache_manager.get(cache_key)
        if cached:
            return cached
            
        if self.use_mock:
            weather = get_mock_current_weather(state, district)
        elif self.owm_api_key:
            weather = await self._fetch_owm_current(state, district)
        else:
            weather = get_mock_current_weather(state, district)
            
        self.cache_manager.set(cache_key, weather)
        return weather
        
    async def _fetch_owm_current(
        self,
        state: str,
        district: str,
    ) -> CurrentWeather:
        """Fetch current weather from OpenWeatherMap."""
        coords = DISTRICT_COORDS.get(
            (state.lower(), district.lower()),
            (20.5937, 78.9629)  # Default: center of India
        )
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"{OWM_API_BASE}/weather",
                    params={
                        "lat": coords[0],
                        "lon": coords[1],
                        "appid": self.owm_api_key,
                        "units": "metric",
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                # Map OWM weather to our conditions
                owm_main = data.get("weather", [{}])[0].get("main", "Clear").lower()
                condition_map = {
                    "clear": WeatherCondition.CLEAR,
                    "clouds": WeatherCondition.PARTLY_CLOUDY,
                    "rain": WeatherCondition.MODERATE_RAIN,
                    "drizzle": WeatherCondition.LIGHT_RAIN,
                    "thunderstorm": WeatherCondition.THUNDERSTORM,
                    "fog": WeatherCondition.FOG,
                    "haze": WeatherCondition.HAZE,
                }
                
                return CurrentWeather(
                    state=state.title(),
                    district=district.title(),
                    station=data.get("name", district),
                    temperature_c=data["main"]["temp"],
                    feels_like_c=data["main"]["feels_like"],
                    temp_min_c=data["main"]["temp_min"],
                    temp_max_c=data["main"]["temp_max"],
                    humidity_pct=data["main"]["humidity"],
                    pressure_hpa=data["main"]["pressure"],
                    wind_speed_kmh=data["wind"]["speed"] * 3.6,
                    condition=condition_map.get(owm_main, WeatherCondition.CLEAR),
                    description=data.get("weather", [{}])[0].get("description", ""),
                    source="openweathermap",
                )
                
        except Exception as e:
            logger.warning(f"OWM API error: {e}, using mock data")
            return get_mock_current_weather(state, district)
            
    async def get_forecast(
        self,
        state: str,
        district: str,
        days: int = 7,
    ) -> WeatherForecast:
        """Get weather forecast for a district."""
        cache_key = self.cache_manager.get_cache_key("forecast", state, district, days)
        cached = self.cache_manager.get(cache_key)
        if cached:
            return cached
            
        current = await self.get_current_weather(state, district)
        
        if self.use_mock:
            daily_forecasts = get_mock_forecast_data(state, district, days)
        else:
            daily_forecasts = await self._fetch_forecast(state, district, days)
            
        forecast = WeatherForecast(
            state=state.title(),
            district=district.title(),
            current=current,
            daily_forecasts=daily_forecasts,
        )
        
        self.cache_manager.set(cache_key, forecast)
        return forecast
        
    async def _fetch_forecast(
        self,
        state: str,
        district: str,
        days: int,
    ) -> list:
        """Fetch forecast from API."""
        return get_mock_forecast_data(state, district, days)
        
    async def get_alerts(
        self,
        state: str,
        district: Optional[str] = None,
    ) -> list[WeatherAlert]:
        """Get active weather alerts."""
        return get_mock_alerts_data(state, district)
        
    async def get_agro_advisory(
        self,
        state: str,
        district: str,
        crop: str = "general",
    ) -> AgroAdvisory:
        """Get agricultural advisory based on weather."""
        weather = await self.get_current_weather(state, district)
        forecast = await self.get_forecast(state, district, days=5)
        
        return generate_agro_advisory_data(weather, forecast, crop)


# Singleton instance
_imd_client: Optional[IMDWeatherClient] = None

def get_imd_client(
    imd_api_key: str = "",
    owm_api_key: str = "",
    use_mock: bool = True,
) -> IMDWeatherClient:
    """Get or create singleton IMD client instance."""
    global _imd_client
    if _imd_client is None:
        _imd_client = IMDWeatherClient(
            imd_api_key=imd_api_key,
            owm_api_key=owm_api_key,
            use_mock=use_mock,
        )
    return _imd_client
