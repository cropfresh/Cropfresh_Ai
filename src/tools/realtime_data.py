"""
Real-Time Data Manager Proxy (Tools)
====================================
Preserves backwards compatibility for old imports from `src.tools.realtime_data`.
"""

from src.scrapers.realtime_data import (
    DataFreshness,
    DataSourceHealth,
    DataSourceStatus,
    RealTimeData,
    RealTimeDataManager,
    get_realtime_data_manager,
)

__all__ = [
    "DataSourceStatus",
    "DataFreshness",
    "DataSourceHealth",
    "RealTimeData",
    "RealTimeDataManager",
    "get_realtime_data_manager",
]
