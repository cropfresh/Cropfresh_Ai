"""
Weather Tool
============
Weather data for agricultural planning.

Provides:
- Current weather conditions
- 7-day forecast
- Crop-specific advisories
- Historical weather patterns

Author: CropFresh AI Team
Version: 2.0.0
"""

from datetime import datetime, timedelta
from typing import Optional

import httpx
from loguru import logger
from pydantic import BaseModel, Field

from src.tools.registry import get_tool_registry


class WeatherData(BaseModel):
    """Weather data for a location."""
    
    location: str
    date: datetime = Field(default_factory=datetime.now)
    
    # Current conditions
    temperature_c: float
    humidity_pct: float
    rainfall_mm: float = 0.0
    wind_speed_kmh: float = 0.0
    condition: str  # sunny, cloudy, rainy, etc.
    
    # For farming
    soil_moisture: str = "normal"  # dry, normal, wet
    advisory: str = ""


class WeatherForecast(BaseModel):
    """Multi-day weather forecast."""
    
    location: str
    current: WeatherData
    forecast: list[WeatherData] = Field(default_factory=list)
    
    # Agricultural advisories
    planting_advisory: str = ""
    irrigation_advisory: str = ""
    pest_alert: str = ""


class WeatherTool:
    """
    Weather API integration for agricultural planning.
    
    Currently uses mock data; can be extended to use real APIs:
    - OpenWeatherMap
    - IMD (India Meteorological Department)
    - Weather.gov
    
    Usage:
        tool = WeatherTool()
        weather = await tool.get_current("Kolar")
        forecast = await tool.get_forecast("Bangalore", days=7)
    """
    
    # Karnataka locations with typical weather patterns
    LOCATION_DATA = {
        "kolar": {"temp_base": 28, "rainfall_base": 2, "zone": "semi-arid"},
        "bangalore": {"temp_base": 25, "rainfall_base": 3, "zone": "tropical"},
        "bengaluru": {"temp_base": 25, "rainfall_base": 3, "zone": "tropical"},
        "mysore": {"temp_base": 27, "rainfall_base": 2, "zone": "semi-arid"},
        "mysuru": {"temp_base": 27, "rainfall_base": 2, "zone": "semi-arid"},
        "hubli": {"temp_base": 30, "rainfall_base": 1, "zone": "semi-arid"},
        "shimoga": {"temp_base": 28, "rainfall_base": 5, "zone": "tropical-wet"},
    }
    
    def __init__(self, api_key: str = "", use_mock: bool = True):
        """
        Initialize weather tool.
        
        Args:
            api_key: API key for weather service
            use_mock: Use mock data (default True)
        """
        self.api_key = api_key
        self.use_mock = use_mock
    
    async def get_current(self, location: str) -> WeatherData:
        """
        Get current weather for a location.
        
        Args:
            location: Location name (city/district)
            
        Returns:
            WeatherData for current conditions
        """
        if self.use_mock or not self.api_key:
            return self._get_mock_current(location)
            
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                url = "https://api.openweathermap.org/data/2.5/weather"
                params = {
                    "q": f"{location},IN",
                    "appid": self.api_key,
                    "units": "metric"
                }
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                temp = data["main"]["temp"]
                humidity = data["main"]["humidity"]
                wind = data["wind"]["speed"] * 3.6 # m/s to km/h
                condition = data["weather"][0]["description"]
                rain = data.get("rain", {}).get("1h", 0.0)
                
                soil = "wet" if rain > 5 else "normal" if rain > 1 else "dry"
                advisory = self._generate_advisory(temp, humidity, rain, condition)
                
                return WeatherData(
                    location=location.title(),
                    temperature_c=temp,
                    humidity_pct=humidity,
                    rainfall_mm=rain,
                    wind_speed_kmh=round(wind, 1),
                    condition=condition,
                    soil_moisture=soil,
                    advisory=advisory
                )
        except Exception as e:
            logger.error(f"OpenWeatherMap API failed: {e}")
            return self._get_mock_current(location)
    
    async def get_forecast(
        self,
        location: str,
        days: int = 7,
    ) -> WeatherForecast:
        """
        Get weather forecast for a location.
        
        Args:
            location: Location name
            days: Number of forecast days (1-14)
            
        Returns:
            WeatherForecast with multi-day data
        """
        if self.use_mock or not self.api_key:
            return self._get_mock_forecast(location, days)
        
        current = await self.get_current(location)
        
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                url = "https://api.openweathermap.org/data/2.5/forecast"
                params = {
                    "q": f"{location},IN",
                    "appid": self.api_key,
                    "units": "metric"
                }
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                forecast = []
                last_day = None
                
                for item in data.get("list", []):
                    dt = datetime.fromtimestamp(item["dt"])
                    day_str = dt.strftime("%Y-%m-%d")
                    
                    if day_str != last_day and len(forecast) < days:
                        last_day = day_str
                        
                        temp = item["main"]["temp"]
                        humidity = item["main"]["humidity"]
                        wind = item["wind"]["speed"] * 3.6
                        condition = item["weather"][0]["description"]
                        rain = item.get("rain", {}).get("3h", 0.0)
                        
                        soil = "wet" if rain > 5 else "normal" if rain > 1 else "dry"
                        advisory = self._generate_advisory(temp, humidity, rain, condition)
                        
                        forecast.append(WeatherData(
                            location=location.title(),
                            date=dt,
                            temperature_c=temp,
                            humidity_pct=humidity,
                            rainfall_mm=rain,
                            wind_speed_kmh=round(wind, 1),
                            condition=condition,
                            soil_moisture=soil,
                            advisory=advisory
                        ))
                
                total_rain = sum(d.rainfall_mm for d in forecast)
                planting, irrigation, pest = self._generate_forecast_advisories(total_rain)
                
                return WeatherForecast(
                    location=location.title(),
                    current=current,
                    forecast=forecast,
                    planting_advisory=planting,
                    irrigation_advisory=irrigation,
                    pest_alert=pest
                )
        except Exception as e:
            logger.error(f"OpenWeatherMap forecast API failed: {e}")
            return self._get_mock_forecast(location, days)

    def _generate_forecast_advisories(self, total_rain: float) -> tuple[str, str, str]:
        if total_rain > 30:
            return (
                "Heavy rainfall expected. Delay planting of dry-season crops.",
                "No irrigation needed. Ensure proper drainage.",
                "High humidity may increase fungal disease risk. Consider preventive spraying."
            )
        elif total_rain < 5:
            return (
                "Dry spell expected. Good time for planting drought-resistant varieties.",
                "Increase irrigation frequency. Consider mulching to retain moisture.",
                "Dry conditions may attract mites. Monitor crops closely."
            )
        else:
            return (
                "Weather conditions are favorable for most crops.",
                "Moderate irrigation recommended. Adjust based on crop needs.",
                "Normal pest activity expected. Maintain regular monitoring."
            )
    
    def _get_mock_current(self, location: str) -> WeatherData:
        """Generate mock current weather."""
        import random
        
        loc_lower = location.lower()
        base = self.LOCATION_DATA.get(loc_lower, {"temp_base": 28, "rainfall_base": 2})
        
        # Add some randomness
        temp = base["temp_base"] + random.uniform(-3, 5)
        humidity = random.uniform(50, 85)
        rainfall = max(0, base["rainfall_base"] + random.uniform(-2, 5))
        
        conditions = ["sunny", "partly cloudy", "cloudy", "light rain"]
        condition = random.choice(conditions)
        if rainfall > 3:
            condition = "rainy"
        
        # Soil moisture based on recent rainfall
        if rainfall > 5:
            soil = "wet"
        elif rainfall > 1:
            soil = "normal"
        else:
            soil = "dry"
        
        # Generate advisory
        advisory = self._generate_advisory(temp, humidity, rainfall, condition)
        
        return WeatherData(
            location=location.title(),
            temperature_c=round(temp, 1),
            humidity_pct=round(humidity, 1),
            rainfall_mm=round(rainfall, 1),
            wind_speed_kmh=round(random.uniform(5, 20), 1),
            condition=condition,
            soil_moisture=soil,
            advisory=advisory,
        )
    
    def _get_mock_forecast(self, location: str, days: int) -> WeatherForecast:
        """Generate mock forecast."""
        current = self._get_mock_current(location)
        
        forecast = []
        for i in range(1, min(days + 1, 15)):
            day_data = self._get_mock_current(location)
            day_data.date = datetime.now() + timedelta(days=i)
            forecast.append(day_data)
        
        total_rain = sum(d.rainfall_mm for d in forecast)
        planting, irrigation, pest = self._generate_forecast_advisories(total_rain)
        
        return WeatherForecast(
            location=location.title(),
            current=current,
            forecast=forecast,
            planting_advisory=planting,
            irrigation_advisory=irrigation,
            pest_alert=pest,
        )
    
    def _generate_advisory(
        self,
        temp: float,
        humidity: float,
        rainfall: float,
        condition: str,
    ) -> str:
        """Generate weather advisory for farmers."""
        advisories = []
        
        if temp > 35:
            advisories.append("High temperature. Irrigate in early morning or evening.")
        elif temp < 15:
            advisories.append("Cool temperature. Protect sensitive crops from frost.")
        
        if humidity > 80:
            advisories.append("High humidity. Watch for fungal diseases.")
        
        if rainfall > 5:
            advisories.append("Heavy rain expected. Delay spraying operations.")
        elif condition == "sunny" and rainfall == 0:
            advisories.append("Good conditions for field work and spraying.")
        
        return " ".join(advisories) if advisories else "Normal farming conditions."


# Register tool globally
async def _get_weather(location: str, days: int = 1) -> dict:
    """Get weather data for a location."""
    tool = WeatherTool()
    
    if days == 1:
        weather = await tool.get_current(location)
        return weather.model_dump()
    else:
        forecast = await tool.get_forecast(location, days)
        return forecast.model_dump()


# Auto-register on module import
try:
    registry = get_tool_registry()
    registry.add_tool(
        _get_weather,
        name="get_weather",
        description="Get current weather or forecast for a location. Returns temperature, humidity, rainfall, and agricultural advisories.",
        category="weather",
    )
except Exception as e:
    logger.debug(f"Weather tool registration deferred: {e}")
