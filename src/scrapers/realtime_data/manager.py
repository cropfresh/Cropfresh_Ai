"""
Real-Time Data Manager Core
===========================
Unified interface for all real-time agricultural data sources.
"""

from datetime import datetime
from typing import Optional

from loguru import logger

from .fetchers import FetchersMixin
from .health import HealthMixin
from .models import DataSourceHealth, DataSourceStatus


class RealTimeDataManager(HealthMixin, FetchersMixin):
    """
    Unified Real-Time Data Manager.
    Aggregates data from multiple sources.
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
        from src.tools.agmarknet import AgmarknetTool
        from src.tools.enam_client import get_enam_client
        from src.tools.google_amed import get_amed_client
        from src.tools.imd_weather import get_imd_client

        self.enam = get_enam_client(api_key=enam_api_key, use_mock=use_mock)
        self.imd = get_imd_client(
            imd_api_key=imd_api_key,
            owm_api_key=owm_api_key,
            use_mock=use_mock,
        )
        self.amed = get_amed_client(api_key=amed_api_key, use_mock=use_mock)
        self.agmarknet = AgmarknetTool(api_key=agmarknet_api_key)

        self._health: dict[str, DataSourceHealth] = {
            "enam": DataSourceHealth(source="enam", status=DataSourceStatus.MOCK if use_mock else DataSourceStatus.HEALTHY),
            "imd": DataSourceHealth(source="imd", status=DataSourceStatus.MOCK if use_mock else DataSourceStatus.HEALTHY),
            "amed": DataSourceHealth(source="amed", status=DataSourceStatus.MOCK if use_mock else DataSourceStatus.HEALTHY),
            "agmarknet": DataSourceHealth(source="agmarknet", status=DataSourceStatus.MOCK if use_mock else DataSourceStatus.HEALTHY),
        }

        self._request_counts: dict[str, int] = {}
        self._request_reset_time: datetime = datetime.now()

        logger.info("RealTimeDataManager initialized")


# Singleton instance
_data_manager: Optional[RealTimeDataManager] = None


def get_realtime_data_manager(
    use_mock: bool = True,
    **api_keys,
) -> RealTimeDataManager:
    """Get or create singleton data manager instance."""
    global _data_manager

    if _data_manager is None:
        _data_manager = RealTimeDataManager(use_mock=use_mock, **api_keys)

    return _data_manager
