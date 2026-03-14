"""
Real-Time Data Fetchers
=======================
Mixins detailing fetch workflows for prices, weather, and crop monitoring.
"""

import time
import asyncio
from typing import Optional
from loguru import logger

from .models import RealTimeData, DataFreshness


class FetchersMixin:
    """Provides methods to fetch unified data."""

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
        """
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
        """Get weather data with optional forecast and advisory."""
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
        """Get satellite-based crop monitoring data."""
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
        """
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
