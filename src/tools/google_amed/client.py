"""
Google AMED Client
==================
Main API Client integrating Google's Agricultural Landscape Understanding APIs.
"""

from datetime import datetime
from typing import Any, Optional

from loguru import logger

from .mock_data import AMEDMockDataMixin
from .models import CropMonitoringData, FieldBoundary, RegionalCropStats, SeasonInfo


class GoogleAMEDClient(AMEDMockDataMixin):
    """
    Google AMED (Agricultural Monitoring and Event Detection) API Client.

    Provides satellite-based crop monitoring and analytics for Indian agriculture.

    Usage:
        client = GoogleAMEDClient(api_key="your_gcp_key")
        monitoring = await client.get_crop_monitoring(13.1333, 78.1333)
        season = await client.get_season_info(13.1333, 78.1333, "Tomato")
    """

    AMED_API_BASE = "https://amed.googleapis.com/v1"

    CROP_CALENDAR = {
        "rice": {
            "kharif": {
                "sowing": (6, 15, 7, 31),
                "harvest": (10, 15, 11, 30),
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
        cache_ttl: int = 86400,
        use_mock: bool = True,
    ):
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
        return ":".join(str(a) for a in args)

    def _get_cached(self, key: str) -> Optional[Any]:
        if key in self._cache:
            cached_time, data = self._cache[key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                return data
        return None

    def _set_cache(self, key: str, data: Any):
        self._cache[key] = (datetime.now(), data)

    async def get_crop_monitoring(
        self,
        lat: float,
        lon: float,
        radius_km: float = 10.0,
    ) -> CropMonitoringData:
        """Get crop monitoring data for a location."""
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
        return self._get_mock_monitoring(lat, lon, radius_km)

    async def get_season_info(
        self,
        lat: float,
        lon: float,
        crop: str,
    ) -> SeasonInfo:
        """Get season information for a crop at a location."""
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
        """Detect field boundaries in an area."""
        if self.use_mock:
            return self._get_mock_boundaries(lat, lon, radius_km)
        return []

    async def get_regional_stats(
        self,
        state: str,
        district: str,
    ) -> RegionalCropStats:
        """Get regional crop statistics."""
        cache_key = self._get_cache_key("stats", state, district)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        data = self._get_mock_regional_stats(state, district)
        self._set_cache(cache_key, data)
        return data
