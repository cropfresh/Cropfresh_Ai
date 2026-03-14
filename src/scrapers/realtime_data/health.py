"""
Real-Time Data Health Tracker
=============================
Mixin for tracking the health of data sources.
"""

from datetime import datetime
from typing import Optional

from .models import DataSourceStatus, DataFreshness, DataSourceHealth


class HealthMixin:
    """Mixin for managing data source health metrics."""

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
