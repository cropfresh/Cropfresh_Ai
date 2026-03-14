"""
Real-Time Data Models
=====================
Enums and data structures for real-time data integration.
"""

from datetime import datetime
from typing import Any, Optional
from enum import Enum
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
