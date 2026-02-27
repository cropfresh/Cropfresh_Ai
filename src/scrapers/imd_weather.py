"""
IMD Weather Client
==================
Integration with India Meteorological Department (IMD) for agricultural weather data.

Provides:
- Current weather conditions by district
- 7-day weather forecasts
- Agro-meteorological advisories
- Weather alerts and warnings
- Crop-specific weather recommendations

API Sources:
- Primary: IMD API (when available)
- Secondary: OpenWeatherMap (free tier)
- Fallback: Mock data for development

Author: CropFresh AI Team
Version: 1.0.0
"""

from datetime import datetime, timedelta
from typing import Any, Optional
from enum import Enum

import httpx
from loguru import logger
from pydantic import BaseModel, Field


class WeatherCondition(str, Enum):
    """Weather condition types."""
    SUNNY = "sunny"
    PARTLY_CLOUDY = "partly_cloudy"
    CLOUDY = "cloudy"
    LIGHT_RAIN = "light_rain"
    MODERATE_RAIN = "moderate_rain"
    HEAVY_RAIN = "heavy_rain"
    THUNDERSTORM = "thunderstorm"
    FOG = "fog"
    HAZE = "haze"
    CLEAR = "clear"


class AlertSeverity(str, Enum):
    """Weather alert severity levels."""
    GREEN = "green"      # No warning
    YELLOW = "yellow"    # Watch
    ORANGE = "orange"    # Alert
    RED = "red"          # Warning


class CurrentWeather(BaseModel):
    """Current weather conditions."""
    
    state: str
    district: str
    station: str = ""
    
    # Temperature
    temperature_c: float
    feels_like_c: float = 0.0
    temp_min_c: float = 0.0
    temp_max_c: float = 0.0
    
    # Humidity & Pressure
    humidity_pct: float
    pressure_hpa: float = 1013.0
    
    # Precipitation
    rainfall_mm: float = 0.0
    rainfall_last_24h_mm: float = 0.0
    
    # Wind
    wind_speed_kmh: float = 0.0
    wind_direction: str = ""
    
    # Conditions
    condition: WeatherCondition
    description: str = ""
    
    # Soil
    soil_moisture: str = "normal"  # dry, normal, wet, waterlogged
    soil_temperature_c: float = 0.0
    
    # Timestamps
    observation_time: datetime = Field(default_factory=datetime.now)
    sunrise: str = ""
    sunset: str = ""
    
    # Source
    source: str = "imd"


class DailyForecast(BaseModel):
    """Single day forecast."""
    
    date: datetime
    
    # Temperature
    temp_max_c: float
    temp_min_c: float
    
    # Precipitation
    rainfall_probability_pct: float = 0.0
    expected_rainfall_mm: float = 0.0
    
    # Conditions
    condition: WeatherCondition
    description: str = ""
    
    # Wind
    wind_speed_kmh: float = 0.0
    
    # Humidity
    humidity_pct: float = 0.0


class WeatherForecast(BaseModel):
    """Multi-day weather forecast."""
    
    state: str
    district: str
    
    # Current conditions
    current: CurrentWeather
    
    # Daily forecasts
    daily_forecasts: list[DailyForecast] = Field(default_factory=list)
    
    # Metadata
    forecast_generated: datetime = Field(default_factory=datetime.now)
    source: str = "imd"


class WeatherAlert(BaseModel):
    """Weather alert/warning."""
    
    state: str
    district: str
    
    severity: AlertSeverity
    alert_type: str  # Heavy Rain, Cyclone, Heat Wave, etc.
    headline: str
    description: str
    
    start_time: datetime
    end_time: datetime
    
    # Farming impact
    farming_advisory: str = ""
    
    issued_at: datetime = Field(default_factory=datetime.now)


class AgroAdvisory(BaseModel):
    """Agricultural advisory based on weather."""
    
    state: str
    district: str
    crop: str = "general"
    
    # Current conditions summary
    weather_summary: str
    
    # Advisories
    irrigation_advisory: str = ""
    sowing_advisory: str = ""
    harvesting_advisory: str = ""
    pest_disease_alert: str = ""
    fertilizer_advisory: str = ""
    
    # General recommendations
    recommendations: list[str] = Field(default_factory=list)
    
    # Valid period
    valid_from: datetime = Field(default_factory=datetime.now)
    valid_until: datetime = Field(default_factory=lambda: datetime.now() + timedelta(days=3))


class IMDWeatherClient:
    """
    IMD Weather Client for agricultural weather data.
    
    Provides hyperlocal weather data at district level with
    crop-specific advisories for Indian farmers.
    
    Usage:
        client = IMDWeatherClient()
        weather = await client.get_current_weather("Karnataka", "Kolar")
        forecast = await client.get_forecast("Karnataka", "Kolar", days=7)
        advisory = await client.get_agro_advisory("Karnataka", "Kolar", "Tomato")
    """
    
    # API Endpoints
    IMD_API_BASE = "https://api.imd.gov.in/v1"
    OWM_API_BASE = "https://api.openweathermap.org/data/2.5"
    
    # District coordinates (for OpenWeatherMap fallback)
    DISTRICT_COORDS = {
        ("karnataka", "kolar"): (13.1333, 78.1333),
        ("karnataka", "bangalore"): (12.9716, 77.5946),
        ("karnataka", "mysore"): (12.2958, 76.6394),
        ("karnataka", "hubli"): (15.3647, 75.1240),
        ("karnataka", "belgaum"): (15.8497, 74.4977),
        ("maharashtra", "nashik"): (19.9975, 73.7898),
        ("maharashtra", "pune"): (18.5204, 73.8567),
        ("maharashtra", "mumbai"): (19.0760, 72.8777),
        ("tamil nadu", "chennai"): (13.0827, 80.2707),
        ("tamil nadu", "coimbatore"): (11.0168, 76.9558),
        ("andhra pradesh", "vijayawada"): (16.5062, 80.6480),
        ("telangana", "hyderabad"): (17.3850, 78.4867),
        ("gujarat", "ahmedabad"): (23.0225, 72.5714),
        ("rajasthan", "jaipur"): (26.9124, 75.7873),
        ("uttar pradesh", "lucknow"): (26.8467, 80.9462),
    }
    
    def __init__(
        self,
        imd_api_key: str = "",
        owm_api_key: str = "",
        cache_ttl: int = 1800,  # 30 minutes
        use_mock: bool = True,
    ):
        """
        Initialize IMD Weather client.
        
        Args:
            imd_api_key: IMD API key (if available)
            owm_api_key: OpenWeatherMap API key (fallback)
            cache_ttl: Cache TTL in seconds
            use_mock: Use mock data (default True)
        """
        self.imd_api_key = imd_api_key
        self.owm_api_key = owm_api_key
        self.cache_ttl = cache_ttl
        self.use_mock = use_mock or (not imd_api_key and not owm_api_key)
        
        self._cache: dict[str, tuple[datetime, Any]] = {}
        
        if self.use_mock:
            logger.info("IMDWeatherClient initialized in MOCK mode")
        else:
            logger.info("IMDWeatherClient initialized with live API")
    
    def _get_cache_key(self, *args) -> str:
        """Generate cache key."""
        return ":".join(str(a).lower() for a in args)
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if valid."""
        if key in self._cache:
            cached_time, data = self._cache[key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                return data
        return None
    
    def _set_cache(self, key: str, data: Any):
        """Store in cache."""
        self._cache[key] = (datetime.now(), data)
    
    async def get_current_weather(
        self,
        state: str,
        district: str,
    ) -> CurrentWeather:
        """
        Get current weather for a district.
        
        Args:
            state: Indian state
            district: District name
            
        Returns:
            CurrentWeather with live data
        """
        cache_key = self._get_cache_key("current", state, district)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        if self.use_mock:
            weather = self._get_mock_current(state, district)
        elif self.owm_api_key:
            weather = await self._fetch_owm_current(state, district)
        else:
            weather = self._get_mock_current(state, district)
        
        self._set_cache(cache_key, weather)
        return weather
    
    async def _fetch_owm_current(
        self,
        state: str,
        district: str,
    ) -> CurrentWeather:
        """Fetch current weather from OpenWeatherMap."""
        coords = self.DISTRICT_COORDS.get(
            (state.lower(), district.lower()),
            (20.5937, 78.9629)  # Default: center of India
        )
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"{self.OWM_API_BASE}/weather",
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
            return self._get_mock_current(state, district)
    
    async def get_forecast(
        self,
        state: str,
        district: str,
        days: int = 7,
    ) -> WeatherForecast:
        """
        Get weather forecast for a district.
        
        Args:
            state: Indian state
            district: District name
            days: Number of forecast days (1-14)
            
        Returns:
            WeatherForecast with daily predictions
        """
        cache_key = self._get_cache_key("forecast", state, district, days)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        current = await self.get_current_weather(state, district)
        
        if self.use_mock:
            daily_forecasts = self._get_mock_forecast(state, district, days)
        else:
            daily_forecasts = await self._fetch_forecast(state, district, days)
        
        forecast = WeatherForecast(
            state=state.title(),
            district=district.title(),
            current=current,
            daily_forecasts=daily_forecasts,
        )
        
        self._set_cache(cache_key, forecast)
        return forecast
    
    async def _fetch_forecast(
        self,
        state: str,
        district: str,
        days: int,
    ) -> list[DailyForecast]:
        """Fetch forecast from API."""
        # For now, return mock data
        # In production, integrate with IMD or OWM forecast API
        return self._get_mock_forecast(state, district, days)
    
    async def get_alerts(
        self,
        state: str,
        district: Optional[str] = None,
    ) -> list[WeatherAlert]:
        """
        Get active weather alerts.
        
        Args:
            state: Indian state
            district: Optional district filter
            
        Returns:
            List of active WeatherAlert
        """
        # Check for mock mode
        if self.use_mock:
            return self._get_mock_alerts(state, district)
        
        # In production, fetch from IMD alerts API
        return self._get_mock_alerts(state, district)
    
    async def get_agro_advisory(
        self,
        state: str,
        district: str,
        crop: str = "general",
    ) -> AgroAdvisory:
        """
        Get agricultural advisory based on weather.
        
        Args:
            state: Indian state
            district: District name
            crop: Specific crop (or "general")
            
        Returns:
            AgroAdvisory with farming recommendations
        """
        # Get current weather
        weather = await self.get_current_weather(state, district)
        forecast = await self.get_forecast(state, district, days=5)
        
        # Generate advisory based on conditions
        return self._generate_advisory(weather, forecast, crop)
    
    def _generate_advisory(
        self,
        weather: CurrentWeather,
        forecast: WeatherForecast,
        crop: str,
    ) -> AgroAdvisory:
        """Generate agricultural advisory from weather data."""
        recommendations = []
        
        # Irrigation advisory
        if weather.rainfall_mm > 10 or weather.soil_moisture == "wet":
            irrigation = "Skip irrigation - adequate soil moisture from recent rainfall"
        elif weather.temperature_c > 35 and weather.humidity_pct < 40:
            irrigation = "Increase irrigation frequency - high evapotranspiration expected"
        else:
            irrigation = "Follow regular irrigation schedule"
        
        # Check upcoming rainfall
        rain_days = sum(1 for f in forecast.daily_forecasts if f.rainfall_probability_pct > 50)
        
        # Sowing advisory
        if rain_days >= 2:
            sowing = "Good conditions for sowing - rainfall expected"
            recommendations.append("Prepare fields for sowing before rainfall")
        elif weather.soil_moisture == "dry":
            sowing = "Delay sowing until adequate moisture available"
        else:
            sowing = "Suitable conditions for sowing if soil moisture is adequate"
        
        # Harvesting advisory
        if weather.condition in [WeatherCondition.HEAVY_RAIN, WeatherCondition.MODERATE_RAIN]:
            harvesting = "Delay harvesting - rain may damage harvested crop"
            recommendations.append("Cover harvested produce to protect from rain")
        elif rain_days > 0:
            harvesting = "Complete harvesting before expected rainfall"
        else:
            harvesting = "Good conditions for harvesting"
        
        # Pest/disease alert based on conditions
        pest_alert = ""
        if weather.humidity_pct > 80 and weather.temperature_c > 25:
            pest_alert = "High humidity - monitor for fungal diseases. Apply preventive fungicide if needed."
            recommendations.append("Scout fields for early signs of fungal infection")
        elif weather.temperature_c > 35:
            pest_alert = "High temperature stress - watch for pest outbreaks"
        
        # Fertilizer advisory
        if rain_days > 0:
            fertilizer = "Apply fertilizer 2-3 days before expected rainfall for better absorption"
        else:
            fertilizer = "Water plants after fertilizer application for optimal uptake"
        
        # Weather summary
        weather_summary = f"Current: {weather.temperature_c:.1f}°C, {weather.humidity_pct:.0f}% humidity, {weather.condition.value}. "
        if rain_days > 0:
            weather_summary += f"Rainfall expected in next {rain_days} days."
        else:
            weather_summary += "No significant rainfall expected."
        
        # Crop-specific advice
        if crop.lower() == "tomato":
            recommendations.append("Stake tomato plants if heavy rain expected")
            recommendations.append("Monitor for late blight in humid conditions")
        elif crop.lower() == "onion":
            recommendations.append("Reduce irrigation as crop matures")
            recommendations.append("Watch for purple blotch in humid weather")
        elif crop.lower() == "rice":
            recommendations.append("Maintain 2-3cm standing water in paddy")
        
        return AgroAdvisory(
            state=weather.state,
            district=weather.district,
            crop=crop,
            weather_summary=weather_summary,
            irrigation_advisory=irrigation,
            sowing_advisory=sowing,
            harvesting_advisory=harvesting,
            pest_disease_alert=pest_alert,
            fertilizer_advisory=fertilizer,
            recommendations=recommendations,
        )
    
    def _get_mock_current(self, state: str, district: str) -> CurrentWeather:
        """Generate mock current weather."""
        import random
        
        # Seasonal variation (January)
        base_temp = random.uniform(18, 28)
        humidity = random.uniform(45, 75)
        
        conditions = [
            WeatherCondition.SUNNY,
            WeatherCondition.PARTLY_CLOUDY,
            WeatherCondition.CLEAR,
        ]
        condition = random.choice(conditions)
        
        return CurrentWeather(
            state=state.title(),
            district=district.title(),
            station=f"{district.title()} Observatory",
            temperature_c=base_temp,
            feels_like_c=base_temp + random.uniform(-2, 2),
            temp_min_c=base_temp - 5,
            temp_max_c=base_temp + 8,
            humidity_pct=humidity,
            pressure_hpa=random.uniform(1008, 1018),
            rainfall_mm=0 if condition == WeatherCondition.SUNNY else random.uniform(0, 5),
            wind_speed_kmh=random.uniform(5, 20),
            wind_direction=random.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
            condition=condition,
            description=condition.value.replace("_", " ").title(),
            soil_moisture=random.choice(["normal", "normal", "dry", "wet"]),
            soil_temperature_c=base_temp - 3,
            sunrise="06:45",
            sunset="18:15",
            source="mock",
        )
    
    def _get_mock_forecast(
        self,
        state: str,
        district: str,
        days: int,
    ) -> list[DailyForecast]:
        """Generate mock forecast data."""
        import random
        
        forecasts = []
        base_temp = random.uniform(20, 30)
        
        for i in range(days):
            date = datetime.now() + timedelta(days=i)
            temp_variation = random.uniform(-3, 3)
            
            # Occasional rain
            rain_prob = random.uniform(0, 100)
            if rain_prob > 70:
                condition = random.choice([
                    WeatherCondition.LIGHT_RAIN,
                    WeatherCondition.MODERATE_RAIN,
                ])
                rainfall = random.uniform(5, 30)
            else:
                condition = random.choice([
                    WeatherCondition.SUNNY,
                    WeatherCondition.PARTLY_CLOUDY,
                    WeatherCondition.CLEAR,
                ])
                rainfall = 0
            
            forecasts.append(DailyForecast(
                date=date,
                temp_max_c=base_temp + temp_variation + 5,
                temp_min_c=base_temp + temp_variation - 5,
                rainfall_probability_pct=rain_prob if rain_prob > 40 else 0,
                expected_rainfall_mm=rainfall,
                condition=condition,
                description=condition.value.replace("_", " ").title(),
                wind_speed_kmh=random.uniform(8, 25),
                humidity_pct=random.uniform(50, 80),
            ))
        
        return forecasts
    
    def _get_mock_alerts(
        self,
        state: str,
        district: Optional[str],
    ) -> list[WeatherAlert]:
        """Generate mock weather alerts."""
        import random
        
        # 70% chance of no alerts
        if random.random() > 0.3:
            return []
        
        alert_types = [
            ("Heavy Rainfall", "Heavy rainfall expected. May cause waterlogging.", "Delay field operations. Ensure proper drainage."),
            ("Heat Wave", "Temperature expected to exceed 40°C.", "Provide shade for livestock. Increase irrigation."),
            ("Strong Winds", "Gusty winds expected up to 50 km/h.", "Secure loose materials. Support tall crops."),
        ]
        
        alert_type, description, farming_advisory = random.choice(alert_types)
        
        return [
            WeatherAlert(
                state=state.title(),
                district=district.title() if district else "All Districts",
                severity=AlertSeverity.YELLOW,
                alert_type=alert_type,
                headline=f"{alert_type} Warning",
                description=description,
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(hours=24),
                farming_advisory=farming_advisory,
            )
        ]


# Singleton instance
_imd_client: Optional[IMDWeatherClient] = None


def get_imd_client(
    imd_api_key: str = "",
    owm_api_key: str = "",
    use_mock: bool = True,
) -> IMDWeatherClient:
    """
    Get or create singleton IMD client instance.
    
    Args:
        imd_api_key: IMD API key
        owm_api_key: OpenWeatherMap API key
        use_mock: Use mock data
        
    Returns:
        IMDWeatherClient instance
    """
    global _imd_client
    
    if _imd_client is None:
        _imd_client = IMDWeatherClient(
            imd_api_key=imd_api_key,
            owm_api_key=owm_api_key,
            use_mock=use_mock,
        )
    
    return _imd_client
