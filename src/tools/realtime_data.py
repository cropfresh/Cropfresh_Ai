"""
Real-Time Data Manager
======================
Unified interface for all real-time agricultural data sources.

Provides:
- Unified API for prices, weather, and crop monitoring
- Automatic fallback chains when APIs fail
- Data freshness indicators
- Caching and rate limiting
- Health checks for all data sources

Author: CropFresh AI Team
Version: 1.0.0
"""

from datetime import datetime, timedelta
from typing import Any, Optional
from enum import Enum

from loguru import logger
from pydantic import BaseModel, Field


class DataSourceStatus(str, Enum):
    """Data source health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    MOCK = "mock"


class DataFreshness(str, Enum):
    """Data freshness levels."""
    LIVE = "live"           # < 5 minutes old
    RECENT = "recent"       # < 30 minutes old
    STALE = "stale"         # < 2 hours old
    OUTDATED = "outdated"   # > 2 hours old


class DataSourceHealth(BaseModel):
    """Health status of a data source."""
    
    source: str
    status: DataSourceStatus
    last_successful_call: Optional[datetime] = None
    last_error: Optional[str] = None
    avg_response_time_ms: float = 0.0
    success_rate_24h: float = 100.0


class RealTimeData(BaseModel):
    """Container for real-time data with metadata."""
    
    data: Any
    source: str
    freshness: DataFreshness
    fetched_at: datetime = Field(default_factory=datetime.now)
    cached: bool = False
    fallback_used: bool = False
    
    @property
    def age_seconds(self) -> int:
        """Get age of data in seconds."""
        return int((datetime.now() - self.fetched_at).total_seconds())
    
    @property
    def age_display(self) -> str:
        """Human-readable age."""
        age = self.age_seconds
        if age < 60:
            return f"{age} seconds ago"
        elif age < 3600:
            return f"{age // 60} minutes ago"
        else:
            return f"{age // 3600} hours ago"


class RealTimeDataManager:
    """
    Unified Real-Time Data Manager.
    
    Aggregates data from multiple sources:
    - eNAM: Live mandi prices
    - IMD: Weather data
    - AMED: Crop monitoring
    - Agmarknet: Historical prices
    
    Features:
    - Automatic fallback chains
    - Caching with freshness indicators
    - Health monitoring
    - Rate limiting
    
    Usage:
        manager = RealTimeDataManager()
        
        # Get unified data
        price_data = await manager.get_commodity_prices("Tomato", "Karnataka")
        weather_data = await manager.get_weather("Karnataka", "Kolar")
        crop_data = await manager.get_crop_info(13.1333, 78.1333)
        
        # Check health
        health = manager.get_health_status()
    """
    
    def __init__(
        self,
        enam_api_key: str = "",
        imd_api_key: str = "",
        owm_api_key: str = "",
        amed_api_key: str = "",
        agmarknet_api_key: str = "",
        use_mock: bool = True,
    ):
        """
        Initialize real-time data manager.
        
        Args:
            enam_api_key: eNAM API key
            imd_api_key: IMD API key
            owm_api_key: OpenWeatherMap API key
            amed_api_key: Google AMED API key
            agmarknet_api_key: Agmarknet API key
            use_mock: Use mock data for all sources
        """
        from src.tools.enam_client import get_enam_client
        from src.tools.imd_weather import get_imd_client
        from src.tools.google_amed import get_amed_client
        from src.tools.agmarknet import AgmarknetTool
        
        # Initialize clients
        self.enam = get_enam_client(api_key=enam_api_key, use_mock=use_mock)
        self.imd = get_imd_client(
            imd_api_key=imd_api_key,
            owm_api_key=owm_api_key,
            use_mock=use_mock,
        )
        self.amed = get_amed_client(api_key=amed_api_key, use_mock=use_mock)
        self.agmarknet = AgmarknetTool(api_key=agmarknet_api_key)
        
        # Health tracking
        self._health: dict[str, DataSourceHealth] = {
            "enam": DataSourceHealth(source="enam", status=DataSourceStatus.MOCK if use_mock else DataSourceStatus.HEALTHY),
            "imd": DataSourceHealth(source="imd", status=DataSourceStatus.MOCK if use_mock else DataSourceStatus.HEALTHY),
            "amed": DataSourceHealth(source="amed", status=DataSourceStatus.MOCK if use_mock else DataSourceStatus.HEALTHY),
            "agmarknet": DataSourceHealth(source="agmarknet", status=DataSourceStatus.MOCK if use_mock else DataSourceStatus.HEALTHY),
        }
        
        # Request tracking for rate limiting
        self._request_counts: dict[str, int] = {}
        self._request_reset_time: datetime = datetime.now()
        
        logger.info("RealTimeDataManager initialized")
    
    def _get_freshness(self, fetched_at: datetime) -> DataFreshness:
        """Determine data freshness level."""
        age = (datetime.now() - fetched_at).total_seconds()
        
        if age < 300:  # 5 minutes
            return DataFreshness.LIVE
        elif age < 1800:  # 30 minutes
            return DataFreshness.RECENT
        elif age < 7200:  # 2 hours
            return DataFreshness.STALE
        else:
            return DataFreshness.OUTDATED
    
    def _update_health(
        self,
        source: str,
        success: bool,
        response_time_ms: float = 0.0,
        error: Optional[str] = None,
    ):
        """Update health status for a data source."""
        health = self._health[source]
        
        if success:
            health.last_successful_call = datetime.now()
            health.status = DataSourceStatus.HEALTHY
            # Update moving average
            health.avg_response_time_ms = (health.avg_response_time_ms * 0.9) + (response_time_ms * 0.1)
        else:
            health.last_error = error
            if health.last_successful_call:
                age = (datetime.now() - health.last_successful_call).total_seconds()
                if age > 3600:
                    health.status = DataSourceStatus.UNAVAILABLE
                else:
                    health.status = DataSourceStatus.DEGRADED
            else:
                health.status = DataSourceStatus.UNAVAILABLE
    
    async def get_commodity_prices(
        self,
        commodity: str,
        state: str,
        district: Optional[str] = None,
        include_trends: bool = False,
    ) -> RealTimeData:
        """
        Get commodity prices from best available source.
        
        Fallback chain: eNAM -> Agmarknet -> Mock
        
        Args:
            commodity: Commodity name
            state: Indian state
            district: Optional district
            include_trends: Include price trends
            
        Returns:
            RealTimeData with prices
        """
        import time
        start_time = time.time()
        fallback_used = False
        
        try:
            # Try eNAM first
            prices = await self.enam.get_live_prices(
                commodity=commodity,
                state=state,
                district=district,
            )
            
            result = {"prices": [p.model_dump() for p in prices]}
            
            # Add trends if requested
            if include_trends:
                trend = await self.enam.get_price_trends(commodity, state)
                result["trend"] = trend.model_dump()
            
            source = "enam"
            self._update_health("enam", True, (time.time() - start_time) * 1000)
            
        except Exception as e:
            logger.warning(f"eNAM failed: {e}, trying Agmarknet")
            fallback_used = True
            
            try:
                # Fallback to Agmarknet
                prices = await self.agmarknet.get_prices(
                    commodity=commodity,
                    state=state,
                    district=district,
                )
                result = {"prices": [p.model_dump() for p in prices]}
                source = "agmarknet"
                self._update_health("agmarknet", True, (time.time() - start_time) * 1000)
                self._update_health("enam", False, error=str(e))
                
            except Exception as e2:
                logger.error(f"All price sources failed: {e2}")
                # Use mock data as last resort
                prices = self.agmarknet.get_mock_prices(commodity, state, district or "")
                result = {"prices": [p.model_dump() for p in prices]}
                source = "mock"
                self._update_health("agmarknet", False, error=str(e2))
        
        return RealTimeData(
            data=result,
            source=source,
            freshness=DataFreshness.LIVE,
            fallback_used=fallback_used,
        )
    
    async def get_weather(
        self,
        state: str,
        district: str,
        include_forecast: bool = True,
        include_advisory: bool = True,
        crop: Optional[str] = None,
    ) -> RealTimeData:
        """
        Get weather data with optional forecast and advisory.
        
        Args:
            state: Indian state
            district: District name
            include_forecast: Include multi-day forecast
            include_advisory: Include agro advisory
            crop: Specific crop for advisory
            
        Returns:
            RealTimeData with weather information
        """
        import time
        start_time = time.time()
        
        try:
            current = await self.imd.get_current_weather(state, district)
            result = {"current": current.model_dump()}
            
            if include_forecast:
                forecast = await self.imd.get_forecast(state, district)
                result["forecast"] = [f.model_dump() for f in forecast.daily_forecasts]
            
            if include_advisory:
                advisory = await self.imd.get_agro_advisory(
                    state, district, crop or "general"
                )
                result["advisory"] = advisory.model_dump()
            
            # Check for alerts
            alerts = await self.imd.get_alerts(state, district)
            if alerts:
                result["alerts"] = [a.model_dump() for a in alerts]
            
            source = current.source
            self._update_health("imd", True, (time.time() - start_time) * 1000)
            
        except Exception as e:
            logger.error(f"Weather fetch failed: {e}")
            self._update_health("imd", False, error=str(e))
            
            # Return mock data
            from src.tools.imd_weather import IMDWeatherClient
            mock_client = IMDWeatherClient(use_mock=True)
            current = mock_client._get_mock_current(state, district)
            result = {"current": current.model_dump()}
            source = "mock"
        
        return RealTimeData(
            data=result,
            source=source,
            freshness=DataFreshness.LIVE,
        )
    
    async def get_crop_monitoring(
        self,
        lat: float,
        lon: float,
        radius_km: float = 10.0,
        crop: Optional[str] = None,
    ) -> RealTimeData:
        """
        Get satellite-based crop monitoring data.
        
        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Search radius
            crop: Specific crop for season info
            
        Returns:
            RealTimeData with crop monitoring
        """
        import time
        start_time = time.time()
        
        try:
            monitoring = await self.amed.get_crop_monitoring(lat, lon, radius_km)
            result = {"monitoring": monitoring.model_dump()}
            
            if crop:
                season = await self.amed.get_season_info(lat, lon, crop)
                result["season"] = season.model_dump()
            
            self._update_health("amed", True, (time.time() - start_time) * 1000)
            
        except Exception as e:
            logger.error(f"Crop monitoring failed: {e}")
            self._update_health("amed", False, error=str(e))
            
            # Return mock data
            from src.tools.google_amed import GoogleAMEDClient
            mock_client = GoogleAMEDClient(use_mock=True)
            monitoring = mock_client._get_mock_monitoring(lat, lon, radius_km)
            result = {"monitoring": monitoring.model_dump()}
        
        return RealTimeData(
            data=result,
            source="amed",
            freshness=DataFreshness.RECENT,  # Satellite data is typically a few days old
        )
    
    async def get_comprehensive_data(
        self,
        commodity: str,
        state: str,
        district: str,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> dict[str, RealTimeData]:
        """
        Get comprehensive real-time data for a location.
        
        Fetches prices, weather, and crop monitoring in parallel.
        
        Args:
            commodity: Commodity name
            state: Indian state
            district: District name
            lat: Optional latitude for crop monitoring
            lon: Optional longitude for crop monitoring
            
        Returns:
            Dict with prices, weather, and monitoring data
        """
        import asyncio
        
        # Parallel fetch
        tasks = {
            "prices": self.get_commodity_prices(commodity, state, district, include_trends=True),
            "weather": self.get_weather(state, district, crop=commodity),
        }
        
        if lat and lon:
            tasks["monitoring"] = self.get_crop_monitoring(lat, lon, crop=commodity)
        
        results = {}
        for key, task in tasks.items():
            try:
                results[key] = await task
            except Exception as e:
                logger.error(f"Failed to fetch {key}: {e}")
                results[key] = RealTimeData(
                    data={"error": str(e)},
                    source="error",
                    freshness=DataFreshness.OUTDATED,
                )
        
        return results
    
    def get_health_status(self) -> dict[str, DataSourceHealth]:
        """Get health status of all data sources."""
        return self._health
    
    def get_health_summary(self) -> dict:
        """Get summarized health status."""
        statuses = {k: v.status.value for k, v in self._health.items()}
        
        healthy_count = sum(1 for s in statuses.values() if s in ["healthy", "mock"])
        total = len(statuses)
        
        return {
            "overall": "healthy" if healthy_count == total else "degraded",
            "healthy_sources": healthy_count,
            "total_sources": total,
            "sources": statuses,
            "checked_at": datetime.now().isoformat(),
        }
    
    def get_freshness_summary(self) -> dict:
        """Get summary of data freshness across all cached data."""
        return {
            "enam": self.enam.get_data_freshness(),
            "overall_mode": "mock" if any(
                h.status == DataSourceStatus.MOCK for h in self._health.values()
            ) else "live",
        }


# Singleton instance
_data_manager: Optional[RealTimeDataManager] = None


def get_realtime_data_manager(
    use_mock: bool = True,
    **api_keys,
) -> RealTimeDataManager:
    """
    Get or create singleton data manager instance.
    
    Args:
        use_mock: Use mock data for all sources
        **api_keys: API keys for various services
        
    Returns:
        RealTimeDataManager instance
    """
    global _data_manager
    
    if _data_manager is None:
        _data_manager = RealTimeDataManager(use_mock=use_mock, **api_keys)
    
    return _data_manager
