"""
Scraper Scheduler
=================
APScheduler-based daily job scheduler for CropFresh AI data scrapers.

Schedule (IST = UTC+5:30):
  - 10:00 AM IST  Agmarknet daily mandi prices
  - 11:00 AM IST  IMD weather data (when configured)

Uses AsyncIOScheduler so jobs share the running event loop.
"""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger

# APScheduler — already pulled in by the project's scheduling utilities
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger

    _APSCHEDULER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _APSCHEDULER_AVAILABLE = False
    AsyncIOScheduler = None  # type: ignore[assignment,misc]
    CronTrigger = None  # type: ignore[assignment]


# ============================================================================
# ScraperScheduler
# ============================================================================


class ScraperScheduler:
    """
    Schedules daily scraper jobs using APScheduler (AsyncIOScheduler).

    Jobs:
      - agmarknet_daily  10:00 AM IST  → AgmarknetScraper.scrape_and_store()
      - imd_daily        11:00 AM IST  → IMD weather scraper (optional)

    Usage:
        from src.scrapers.agmarknet import AgmarknetScraper

        scraper = AgmarknetScraper()
        scheduler = ScraperScheduler(agmarknet=scraper, db=db_client)
        scheduler.start()
        # ...app runs...
        scheduler.stop()
    """

    #: Timezone applied to all cron triggers
    TIMEZONE = "Asia/Kolkata"

    def __init__(
        self,
        agmarknet: Any = None,
        imd: Any = None,
        db: Any = None,
    ) -> None:
        """
        Args:
            agmarknet: AgmarknetScraper instance (or any object with
                       ``scrape_and_store(db)`` async method).
            imd:       IMD weather scraper instance (optional, for future use).
            db:        Database client passed through to scrapers.
        """
        self.agmarknet = agmarknet
        self.imd = imd
        self.db = db

        if _APSCHEDULER_AVAILABLE:
            self._scheduler: Any = AsyncIOScheduler(timezone=self.TIMEZONE)
        else:  # pragma: no cover
            logger.warning(
                "apscheduler not installed — ScraperScheduler running in no-op mode. "
                "Install with: pip install apscheduler"
            )
            self._scheduler = None

        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Register all jobs and start the scheduler.

        Jobs are idempotent — calling start() again after stop() re-registers.
        """
        if self._scheduler is None:
            logger.warning("[ScraperScheduler] APScheduler unavailable — skipping start")
            return

        self._register_jobs()
        self._scheduler.start()
        self._running = True
        logger.info("[ScraperScheduler] Started — jobs scheduled (IST timezone)")

    def stop(self) -> None:
        """Stop the scheduler and cancel pending jobs."""
        if self._scheduler and self._running:
            self._scheduler.shutdown(wait=False)
            self._running = False
            logger.info("[ScraperScheduler] Stopped")

    @property
    def is_running(self) -> bool:
        """Whether the scheduler is currently active."""
        return self._running

    # ── Job Registration ──────────────────────────────────────────────────

    def _register_jobs(self) -> None:
        """Register all cron jobs. Clears existing jobs first to stay idempotent."""
        self._scheduler.remove_all_jobs()

        # ── Job 1: Agmarknet daily prices at 10:00 AM IST ─────────────
        if self.agmarknet is not None:
            self._scheduler.add_job(
                self._run_agmarknet,
                trigger="cron",
                hour=10,
                minute=0,
                id="agmarknet_daily",
                replace_existing=True,
                misfire_grace_time=3600,  # 1-hour grace if scheduler was down
                name="Agmarknet Daily Price Scrape",
            )
            logger.debug("[ScraperScheduler] Registered agmarknet_daily job (10:00 AM IST)")
        else:
            logger.warning("[ScraperScheduler] No AgmarknetScraper — agmarknet_daily job skipped")

        # ── Job 2: IMD weather at 11:00 AM IST ────────────────────────
        if self.imd is not None:
            self._scheduler.add_job(
                self._run_imd,
                trigger="cron",
                hour=11,
                minute=0,
                id="imd_daily",
                replace_existing=True,
                misfire_grace_time=3600,
                name="IMD Daily Weather Scrape",
            )
            logger.debug("[ScraperScheduler] Registered imd_daily job (11:00 AM IST)")

    # ── Job Handlers ──────────────────────────────────────────────────────

    async def _run_agmarknet(self) -> None:
        """Async job: run Agmarknet daily scrape and store."""
        logger.info("[ScraperScheduler] Running agmarknet_daily job…")
        try:
            count = await self.agmarknet.scrape_and_store(db=self.db)
            logger.info(f"[ScraperScheduler] agmarknet_daily done — {count} records stored")
        except Exception as exc:
            logger.error(f"[ScraperScheduler] agmarknet_daily failed: {exc}")

    async def _run_imd(self) -> None:
        """Async job: run IMD weather data collection."""
        logger.info("[ScraperScheduler] Running imd_daily job…")
        try:
            result = await self.imd.scrape(state="Karnataka", include_advisory=True)
            logger.info(
                f"[ScraperScheduler] imd_daily done — {result.record_count} weather records"
            )
        except Exception as exc:
            logger.error(f"[ScraperScheduler] imd_daily failed: {exc}")

    # ── Introspection ─────────────────────────────────────────────────────

    def get_jobs(self) -> list[dict]:
        """
        Return a list of registered job metadata dicts.

        Useful for health checks and tests.
        """
        if self._scheduler is None:
            return []
        return [
            {
                "id": job.id,
                "name": job.name,
                "next_run": str(job.next_run_time) if job.next_run_time else None,
            }
            for job in self._scheduler.get_jobs()
        ]

    def get_job(self, job_id: str) -> Optional[Any]:
        """Return the APScheduler Job object for *job_id*, or None."""
        if self._scheduler is None:
            return None
        return self._scheduler.get_job(job_id)


# ============================================================================
# Factory
# ============================================================================


def get_scraper_scheduler(
    agmarknet: Any = None,
    imd: Any = None,
    db: Any = None,
) -> ScraperScheduler:
    """
    Factory for ScraperScheduler — allows zero-arg construction in tests.

    In production, wire real scrapers and a DB client; in tests,
    pass None for all to get a no-op scheduler.
    """
    if agmarknet is None:
        # Import inline to avoid circular imports at module level
        from src.scrapers.agmarknet import get_agmarknet_scraper

        agmarknet = get_agmarknet_scraper()

    return ScraperScheduler(agmarknet=agmarknet, imd=imd, db=db)
