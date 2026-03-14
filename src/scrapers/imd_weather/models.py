"""
IMD Weather Data Models
"""

from datetime import datetime, timedelta
from enum import Enum
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
