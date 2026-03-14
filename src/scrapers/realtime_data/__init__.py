"""
Real-Time Data Package
======================
Unified interface for caching, aggregating, and fetching real-time agricultural data.
"""

from .manager import RealTimeDataManager, get_realtime_data_manager
from .models import DataFreshness, DataSourceHealth, DataSourceStatus, RealTimeData

__all__ = [
    "DataSourceStatus",
    "DataFreshness",
    "DataSourceHealth",
    "RealTimeData",
    "RealTimeDataManager",
    "get_realtime_data_manager",
]
