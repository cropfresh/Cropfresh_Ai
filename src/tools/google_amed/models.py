"""
Google AMED Models
==================
Data structures for Google's Agricultural Monitoring and Event Detection API.
"""

from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class CropType(str, Enum):
    """Identified crop types."""
    RICE = "rice"
    WHEAT = "wheat"
    MAIZE = "maize"
    COTTON = "cotton"
    SUGARCANE = "sugarcane"
    GROUNDNUT = "groundnut"
    SOYBEAN = "soybean"
    VEGETABLES = "vegetables"
    FRUITS = "fruits"
    PULSES = "pulses"
    UNKNOWN = "unknown"


class CropStage(str, Enum):
    """Crop growth stages."""
    BARE_SOIL = "bare_soil"
    SOWING = "sowing"
    VEGETATIVE = "vegetative"
    FLOWERING = "flowering"
    GRAIN_FILLING = "grain_filling"
    MATURITY = "maturity"
    HARVESTED = "harvested"


class HealthStatus(str, Enum):
    """Vegetation health status."""
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    STRESSED = "stressed"
    SEVERE_STRESS = "severe_stress"


class FieldBoundary(BaseModel):
    """Detected field boundary."""
    field_id: str
    center_lat: float
    center_lon: float
    area_hectares: float
    polygon_coordinates: list[tuple[float, float]] = Field(default_factory=list)
    detected_at: datetime = Field(default_factory=datetime.now)


class CropMonitoringData(BaseModel):
    """Satellite-based crop monitoring data."""
    location_lat: float
    location_lon: float
    radius_km: float = 10.0
    
    # Detected crops
    primary_crop: CropType
    secondary_crops: list[CropType] = Field(default_factory=list)
    crop_confidence: float = 0.0
    
    # Growth stage
    current_stage: CropStage
    days_since_sowing: int = 0
    
    # Health indices
    ndvi: float = 0.0
    ndwi: float = 0.0
    evi: float = 0.0
    
    # Health status
    health_status: HealthStatus
    stress_indicators: list[str] = Field(default_factory=list)
    
    # Coverage
    total_fields: int = 0
    cultivated_area_hectares: float = 0.0
    
    # Timestamps
    satellite_date: datetime = Field(default_factory=datetime.now)
    data_quality: str = "good"


class SeasonInfo(BaseModel):
    """Crop season information."""
    crop: str
    location_lat: float
    location_lon: float
    season_name: str
    expected_sowing_start: datetime
    expected_sowing_end: datetime
    expected_harvest_start: datetime
    expected_harvest_end: datetime
    current_stage: CropStage
    progress_pct: float = 0.0
    avg_yield_kg_per_hectare: float = 0.0
    last_year_yield: float = 0.0


class RegionalCropStats(BaseModel):
    """Regional crop statistics."""
    state: str
    district: str
    top_crops: list[dict] = Field(default_factory=list)
    total_agricultural_area_sq_km: float = 0.0
    cultivated_area_sq_km: float = 0.0
    irrigated_area_pct: float = 0.0
    current_season: str = ""
    data_date: datetime = Field(default_factory=datetime.now)
