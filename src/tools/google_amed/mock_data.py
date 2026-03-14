"""
Google AMED Mock Data Generators
================================
Mock data generation logic for AMED API fallback.
"""

import random
from datetime import datetime, timedelta

from .models import (
    CropMonitoringData,
    CropStage,
    CropType,
    FieldBoundary,
    HealthStatus,
    RegionalCropStats,
    SeasonInfo,
)


class AMEDMockDataMixin:
    """Mixin to provide mock data generation for Google AMED."""

    def _get_mock_monitoring(
        self,
        lat: float,
        lon: float,
        radius_km: float,
    ) -> CropMonitoringData:
        """Generate mock crop monitoring data."""
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
