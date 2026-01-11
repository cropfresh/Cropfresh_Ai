"""
Google AMED API Client
======================
Integration with Google's Agricultural Monitoring and Event Detection (AMED) API.

Provides:
- Crop type identification
- Field boundary detection
- Sowing and harvesting date estimation
- Vegetation health indices
- Crop stress detection

This is part of Google's Agricultural Landscape Understanding (ALU) initiative.

Author: CropFresh AI Team
Version: 1.0.0
"""

from datetime import datetime, timedelta
from typing import Any, Optional
from enum import Enum

from loguru import logger
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
    ndvi: float = 0.0  # Normalized Difference Vegetation Index
    ndwi: float = 0.0  # Normalized Difference Water Index
    evi: float = 0.0   # Enhanced Vegetation Index
    
    # Health status
    health_status: HealthStatus
    stress_indicators: list[str] = Field(default_factory=list)
    
    # Coverage
    total_fields: int = 0
    cultivated_area_hectares: float = 0.0
    
    # Timestamps
    satellite_date: datetime = Field(default_factory=datetime.now)
    data_quality: str = "good"  # good, moderate, poor (cloud cover)


class SeasonInfo(BaseModel):
    """Crop season information."""
    
    crop: str
    location_lat: float
    location_lon: float
    
    # Season details
    season_name: str  # Kharif, Rabi, Zaid
    
    # Dates
    expected_sowing_start: datetime
    expected_sowing_end: datetime
    expected_harvest_start: datetime
    expected_harvest_end: datetime
    
    # Current status
    current_stage: CropStage
    progress_pct: float = 0.0
    
    # Historical yields
    avg_yield_kg_per_hectare: float = 0.0
    last_year_yield: float = 0.0


class RegionalCropStats(BaseModel):
    """Regional crop statistics."""
    
    state: str
    district: str
    
    # Top crops
    top_crops: list[dict] = Field(default_factory=list)
    
    # Area statistics
    total_agricultural_area_sq_km: float = 0.0
    cultivated_area_sq_km: float = 0.0
    irrigated_area_pct: float = 0.0
    
    # Season
    current_season: str = ""
    
    # Updated
    data_date: datetime = Field(default_factory=datetime.now)


class GoogleAMEDClient:
    """
    Google AMED (Agricultural Monitoring and Event Detection) API Client.
    
    Provides satellite-based crop monitoring and analytics for Indian agriculture.
    
    Usage:
        client = GoogleAMEDClient(api_key="your_gcp_key")
        monitoring = await client.get_crop_monitoring(13.1333, 78.1333)
        season = await client.get_season_info(13.1333, 78.1333, "Tomato")
    """
    
    # API Base URL (placeholder - actual URL requires GCP setup)
    AMED_API_BASE = "https://amed.googleapis.com/v1"
    
    # Crop calendar for Karnataka (example)
    CROP_CALENDAR = {
        "rice": {
            "kharif": {
                "sowing": (6, 15, 7, 31),  # Jun 15 - Jul 31
                "harvest": (10, 15, 11, 30),  # Oct 15 - Nov 30
            },
            "rabi": {
                "sowing": (11, 1, 12, 15),
                "harvest": (3, 1, 4, 15),
            },
        },
        "tomato": {
            "kharif": {
                "sowing": (6, 1, 7, 15),
                "harvest": (10, 1, 11, 15),
            },
            "rabi": {
                "sowing": (10, 15, 11, 30),
                "harvest": (2, 1, 3, 15),
            },
        },
        "onion": {
            "kharif": {
                "sowing": (5, 15, 6, 30),
                "harvest": (9, 1, 10, 15),
            },
            "rabi": {
                "sowing": (10, 1, 11, 15),
                "harvest": (2, 1, 3, 15),
            },
        },
    }
    
    def __init__(
        self,
        api_key: str = "",
        project_id: str = "",
        cache_ttl: int = 86400,  # 24 hours (satellite data doesn't change frequently)
        use_mock: bool = True,
    ):
        """
        Initialize Google AMED client.
        
        Args:
            api_key: GCP API key
            project_id: GCP project ID
            cache_ttl: Cache TTL in seconds
            use_mock: Use mock data (default True)
        """
        self.api_key = api_key
        self.project_id = project_id
        self.cache_ttl = cache_ttl
        self.use_mock = use_mock or not api_key
        
        self._cache: dict[str, tuple[datetime, Any]] = {}
        
        if self.use_mock:
            logger.info("GoogleAMEDClient initialized in MOCK mode")
        else:
            logger.info("GoogleAMEDClient initialized with GCP API")
    
    def _get_cache_key(self, *args) -> str:
        """Generate cache key."""
        return ":".join(str(a) for a in args)
    
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
    
    async def get_crop_monitoring(
        self,
        lat: float,
        lon: float,
        radius_km: float = 10.0,
    ) -> CropMonitoringData:
        """
        Get crop monitoring data for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Search radius in km
            
        Returns:
            CropMonitoringData with satellite-derived insights
        """
        cache_key = self._get_cache_key("monitoring", lat, lon, radius_km)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        if self.use_mock:
            data = self._get_mock_monitoring(lat, lon, radius_km)
        else:
            data = await self._fetch_monitoring(lat, lon, radius_km)
        
        self._set_cache(cache_key, data)
        return data
    
    async def _fetch_monitoring(
        self,
        lat: float,
        lon: float,
        radius_km: float,
    ) -> CropMonitoringData:
        """Fetch monitoring data from AMED API."""
        # In production, make actual API call
        # For now, return mock data
        return self._get_mock_monitoring(lat, lon, radius_km)
    
    async def get_season_info(
        self,
        lat: float,
        lon: float,
        crop: str,
    ) -> SeasonInfo:
        """
        Get season information for a crop at a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            crop: Crop name
            
        Returns:
            SeasonInfo with sowing/harvest dates
        """
        cache_key = self._get_cache_key("season", lat, lon, crop)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        data = self._get_mock_season_info(lat, lon, crop)
        self._set_cache(cache_key, data)
        return data
    
    async def get_field_boundaries(
        self,
        lat: float,
        lon: float,
        radius_km: float = 5.0,
    ) -> list[FieldBoundary]:
        """
        Detect field boundaries in an area.
        
        Args:
            lat: Center latitude
            lon: Center longitude
            radius_km: Search radius
            
        Returns:
            List of detected FieldBoundary
        """
        if self.use_mock:
            return self._get_mock_boundaries(lat, lon, radius_km)
        return []
    
    async def get_regional_stats(
        self,
        state: str,
        district: str,
    ) -> RegionalCropStats:
        """
        Get regional crop statistics.
        
        Args:
            state: Indian state
            district: District name
            
        Returns:
            RegionalCropStats with area and crop distribution
        """
        cache_key = self._get_cache_key("stats", state, district)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        data = self._get_mock_regional_stats(state, district)
        self._set_cache(cache_key, data)
        return data
    
    def _get_mock_monitoring(
        self,
        lat: float,
        lon: float,
        radius_km: float,
    ) -> CropMonitoringData:
        """Generate mock crop monitoring data."""
        import random
        
        now = datetime.now()
        month = now.month
        
        # Determine likely crop based on season
        if 6 <= month <= 10:  # Kharif
            primary_crop = random.choice([CropType.RICE, CropType.MAIZE, CropType.COTTON])
            stage = random.choice([CropStage.VEGETATIVE, CropStage.FLOWERING])
        elif 11 <= month or month <= 3:  # Rabi  
            primary_crop = random.choice([CropType.WHEAT, CropType.VEGETABLES])
            stage = random.choice([CropStage.SOWING, CropStage.VEGETATIVE])
        else:  # Zaid
            primary_crop = random.choice([CropType.VEGETABLES, CropType.FRUITS])
            stage = CropStage.VEGETATIVE
        
        # Generate NDVI (0.2-0.8 for crops)
        ndvi = random.uniform(0.35, 0.75)
        
        # Determine health based on NDVI
        if ndvi > 0.6:
            health = HealthStatus.EXCELLENT
            stress = []
        elif ndvi > 0.45:
            health = HealthStatus.GOOD
            stress = []
        elif ndvi > 0.35:
            health = HealthStatus.MODERATE
            stress = ["Slight water stress"]
        else:
            health = HealthStatus.STRESSED
            stress = ["Water stress", "Possible nutrient deficiency"]
        
        return CropMonitoringData(
            location_lat=lat,
            location_lon=lon,
            radius_km=radius_km,
            primary_crop=primary_crop,
            secondary_crops=[random.choice(list(CropType)) for _ in range(random.randint(0, 2))],
            crop_confidence=random.uniform(0.75, 0.95),
            current_stage=stage,
            days_since_sowing=random.randint(30, 90),
            ndvi=ndvi,
            ndwi=random.uniform(0.1, 0.4),
            evi=ndvi * 0.9,
            health_status=health,
            stress_indicators=stress,
            total_fields=random.randint(50, 200),
            cultivated_area_hectares=random.uniform(100, 500),
            satellite_date=datetime.now() - timedelta(days=random.randint(1, 5)),
            data_quality="good" if random.random() > 0.2 else "moderate",
        )
    
    def _get_mock_season_info(
        self,
        lat: float,
        lon: float,
        crop: str,
    ) -> SeasonInfo:
        """Generate mock season information."""
        now = datetime.now()
        year = now.year
        month = now.month
        
        # Determine current season
        if 6 <= month <= 10:
            season = "Kharif"
            sowing_start = datetime(year, 6, 15)
            sowing_end = datetime(year, 7, 31)
            harvest_start = datetime(year, 10, 15)
            harvest_end = datetime(year, 11, 30)
        elif 11 <= month or month <= 3:
            season = "Rabi"
            sowing_start = datetime(year, 11, 1) if month >= 11 else datetime(year-1, 11, 1)
            sowing_end = datetime(year, 12, 15) if month >= 11 else datetime(year-1, 12, 15)
            harvest_start = datetime(year, 3, 1) if month <= 3 else datetime(year+1, 3, 1)
            harvest_end = datetime(year, 4, 15) if month <= 3 else datetime(year+1, 4, 15)
        else:
            season = "Zaid"
            sowing_start = datetime(year, 3, 1)
            sowing_end = datetime(year, 4, 15)
            harvest_start = datetime(year, 5, 15)
            harvest_end = datetime(year, 6, 30)
        
        # Calculate progress
        season_start = sowing_start
        season_end = harvest_end
        total_days = (season_end - season_start).days
        elapsed_days = (now - season_start).days
        progress = min(100, max(0, (elapsed_days / total_days) * 100))
        
        # Determine stage based on progress
        if progress < 10:
            stage = CropStage.SOWING
        elif progress < 40:
            stage = CropStage.VEGETATIVE
        elif progress < 60:
            stage = CropStage.FLOWERING
        elif progress < 80:
            stage = CropStage.GRAIN_FILLING
        elif progress < 95:
            stage = CropStage.MATURITY
        else:
            stage = CropStage.HARVESTED
        
        return SeasonInfo(
            crop=crop.title(),
            location_lat=lat,
            location_lon=lon,
            season_name=season,
            expected_sowing_start=sowing_start,
            expected_sowing_end=sowing_end,
            expected_harvest_start=harvest_start,
            expected_harvest_end=harvest_end,
            current_stage=stage,
            progress_pct=progress,
            avg_yield_kg_per_hectare=2500 if crop.lower() in ["rice", "wheat"] else 15000,
            last_year_yield=2400 if crop.lower() in ["rice", "wheat"] else 14500,
        )
    
    def _get_mock_boundaries(
        self,
        lat: float,
        lon: float,
        radius_km: float,
    ) -> list[FieldBoundary]:
        """Generate mock field boundaries."""
        import random
        
        boundaries = []
        num_fields = random.randint(5, 15)
        
        for i in range(num_fields):
            # Random offset from center
            offset_lat = random.uniform(-0.05, 0.05)
            offset_lon = random.uniform(-0.05, 0.05)
            
            boundaries.append(FieldBoundary(
                field_id=f"field_{i+1:03d}",
                center_lat=lat + offset_lat,
                center_lon=lon + offset_lon,
                area_hectares=random.uniform(0.5, 5.0),
            ))
        
        return boundaries
    
    def _get_mock_regional_stats(
        self,
        state: str,
        district: str,
    ) -> RegionalCropStats:
        """Generate mock regional statistics."""
        import random
        
        # Top crops by state
        state_crops = {
            "karnataka": ["Rice", "Sugarcane", "Cotton", "Maize", "Ragi"],
            "maharashtra": ["Cotton", "Sugarcane", "Soybean", "Rice", "Wheat"],
            "andhra pradesh": ["Rice", "Chillies", "Cotton", "Groundnut", "Sugarcane"],
            "tamil nadu": ["Rice", "Sugarcane", "Cotton", "Groundnut", "Banana"],
        }
        
        crops = state_crops.get(state.lower(), ["Rice", "Wheat", "Cotton", "Maize", "Groundnut"])
        
        top_crops = [
            {
                "crop": crop,
                "area_pct": random.uniform(10, 35),
                "yield_trend": random.choice(["up", "stable", "down"]),
            }
            for crop in crops[:5]
        ]
        
        return RegionalCropStats(
            state=state.title(),
            district=district.title(),
            top_crops=top_crops,
            total_agricultural_area_sq_km=random.uniform(1000, 3000),
            cultivated_area_sq_km=random.uniform(600, 2000),
            irrigated_area_pct=random.uniform(40, 80),
            current_season="Rabi" if datetime.now().month in [11, 12, 1, 2, 3] else "Kharif",
        )


# Singleton instance
_amed_client: Optional[GoogleAMEDClient] = None


def get_amed_client(
    api_key: str = "",
    project_id: str = "",
    use_mock: bool = True,
) -> GoogleAMEDClient:
    """
    Get or create singleton AMED client instance.
    
    Args:
        api_key: GCP API key
        project_id: GCP project ID
        use_mock: Use mock data
        
    Returns:
        GoogleAMEDClient instance
    """
    global _amed_client
    
    if _amed_client is None:
        _amed_client = GoogleAMEDClient(
            api_key=api_key,
            project_id=project_id,
            use_mock=use_mock,
        )
    
    return _amed_client
