"""APScheduler wrapper for ADCL weekly generation and refresh jobs."""

from __future__ import annotations

from typing import Any

from loguru import logger

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    _APSCHEDULER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _APSCHEDULER_AVAILABLE = False
    AsyncIOScheduler = None  # type: ignore[assignment]


class ADCLScheduler:
    """Refresh district reports weekly and source health daily."""

    def __init__(self, service: Any, districts: list[str]) -> None:
        self.service = service
        self.districts = districts
        self._running = False
        if _APSCHEDULER_AVAILABLE:
            self._scheduler: Any = AsyncIOScheduler(timezone="Asia/Kolkata")
        else:  # pragma: no cover
            self._scheduler = None

    def start(self) -> None:
        """Start APScheduler jobs when available."""
        if self._scheduler is None or not self.districts:
            logger.warning("ADCL scheduler unavailable or no districts configured")
            return
        self._scheduler.add_job(
            self._generate_all_reports,
            trigger="cron",
            id="adcl_weekly_reports",
            name="ADCL Weekly Reports",
            day_of_week="mon",
            hour=6,
            minute=0,
            replace_existing=True,
        )
        self._scheduler.add_job(
            self._refresh_source_health,
            trigger="cron",
            id="adcl_source_health_daily",
            name="ADCL Source Health Refresh",
            hour=5,
            minute=30,
            replace_existing=True,
        )
        self._scheduler.start()
        self._running = True
        logger.info("ADCL scheduler started for {} districts", len(self.districts))

    def stop(self) -> None:
        """Stop the scheduler if it is running."""
        if self._scheduler and self._running:
            self._scheduler.shutdown(wait=False)
            self._running = False

    async def _generate_all_reports(self) -> None:
        for district in self.districts:
            try:
                await self.service.generate_weekly_report(district=district, force_live=True)
            except Exception as exc:  # pragma: no cover
                logger.warning("ADCL weekly generation failed for {}: {}", district, exc)

    async def _refresh_source_health(self) -> None:
        for district in self.districts:
            try:
                await self.service.generate_weekly_report(district=district, force_live=True)
            except Exception as exc:  # pragma: no cover
                logger.warning("ADCL health refresh failed for {}: {}", district, exc)
