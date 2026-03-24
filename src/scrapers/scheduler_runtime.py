"""Runtime implementation for scraper and rate-refresh scheduling."""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger

from src.rates.query_builder import normalize_rate_query
from src.rates.settings import get_agmarknet_api_key

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    _APSCHEDULER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _APSCHEDULER_AVAILABLE = False
    AsyncIOScheduler = None  # type: ignore[assignment]


RATE_JOB_SPECS = [
    {
        "id": "rates_official_mandi_morning",
        "name": "Rate Hub Official Mandi Refresh (Morning)",
        "hour": 6,
        "minute": 15,
        "targets": [
            {"rate_kinds": ["mandi_wholesale"], "commodity": "tomato", "market": "Kolar"},
            {"rate_kinds": ["mandi_wholesale"], "commodity": "onion", "market": "Hubballi"},
        ],
    },
    {
        "id": "rates_support_reference_daily",
        "name": "Rate Hub Support Price Refresh",
        "hour": 7,
        "minute": 0,
        "targets": [{"rate_kinds": ["support_price"], "commodity": "copra"}],
    },
    {
        "id": "rates_fuel_gold_morning",
        "name": "Rate Hub Fuel and Gold Refresh (Morning)",
        "hour": 9,
        "minute": 0,
        "targets": [{"rate_kinds": ["fuel"]}, {"rate_kinds": ["gold"]}],
    },
    {
        "id": "rates_official_mandi_midday",
        "name": "Rate Hub Official Mandi Refresh (Midday)",
        "hour": 12,
        "minute": 15,
        "targets": [
            {"rate_kinds": ["mandi_wholesale"], "commodity": "tomato", "market": "Kolar"},
            {"rate_kinds": ["mandi_wholesale"], "commodity": "potato", "market": "Bengaluru"},
        ],
    },
    {
        "id": "rates_validator_retail_daily",
        "name": "Rate Hub Validator and Retail Refresh",
        "hour": 13,
        "minute": 0,
        "targets": [{"rate_kinds": ["retail_produce"], "commodity": "tomato", "market": "Bengaluru"}],
    },
    {
        "id": "rates_fuel_gold_midday",
        "name": "Rate Hub Fuel and Gold Refresh (Midday)",
        "hour": 13,
        "minute": 0,
        "targets": [{"rate_kinds": ["fuel"]}, {"rate_kinds": ["gold"]}],
    },
    {
        "id": "rates_fuel_gold_evening",
        "name": "Rate Hub Fuel and Gold Refresh (Evening)",
        "hour": 17,
        "minute": 0,
        "targets": [{"rate_kinds": ["fuel"]}, {"rate_kinds": ["gold"]}],
    },
]

LEGACY_SCRAPER_JOB_IDS = ("agmarknet_daily", "imd_daily")
RATE_JOB_SPECS_BY_ID = {spec["id"]: spec for spec in RATE_JOB_SPECS}
ALL_SCRAPER_JOB_IDS = LEGACY_SCRAPER_JOB_IDS + tuple(RATE_JOB_SPECS_BY_ID)


class ScraperScheduler:
    """Schedules legacy scraper jobs and shared rate-hub refresh jobs."""

    TIMEZONE = "Asia/Kolkata"

    def __init__(self, agmarknet: Any = None, imd: Any = None, db: Any = None) -> None:
        self.agmarknet = agmarknet
        self.imd = imd
        self.db = db
        self._running = False
        if _APSCHEDULER_AVAILABLE:
            self._scheduler: Any = AsyncIOScheduler(timezone=self.TIMEZONE)
        else:  # pragma: no cover
            logger.warning("apscheduler not installed; ScraperScheduler running in no-op mode")
            self._scheduler = None

    def start(self) -> None:
        """Register all jobs and start the scheduler."""
        if self._scheduler is None:
            logger.warning("[ScraperScheduler] APScheduler unavailable; skipping start")
            return
        self._register_jobs()
        self._scheduler.start()
        self._running = True
        logger.info("[ScraperScheduler] Started with legacy and rate-hub jobs")

    def stop(self) -> None:
        """Stop the scheduler and cancel pending jobs."""
        if self._scheduler and self._running:
            self._scheduler.shutdown(wait=False)
            self._running = False
            logger.info("[ScraperScheduler] Stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    def get_jobs(self) -> list[dict[str, Optional[str]]]:
        """Return registered job metadata for health checks and tests."""
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
        """Return the underlying APScheduler job object when present."""
        if self._scheduler is None:
            return None
        return self._scheduler.get_job(job_id)

    async def run_job_once(self, job_id: str, *, raise_on_error: bool = True) -> bool:
        """Execute a named job once without starting APScheduler."""
        if job_id == "agmarknet_daily":
            return await self._run_agmarknet(raise_on_error=raise_on_error)
        if job_id == "imd_daily":
            return await self._run_imd(raise_on_error=raise_on_error)

        spec = RATE_JOB_SPECS_BY_ID.get(job_id)
        if spec is None:
            raise ValueError(f"Unknown scraper job_id '{job_id}'")

        return await self._run_rate_refresh(
            job_id=job_id,
            targets=spec["targets"],
            raise_on_error=raise_on_error,
        )

    async def run_jobs_once(
        self,
        job_ids: list[str],
        *,
        raise_on_error: bool = True,
    ) -> bool:
        """Execute multiple jobs in order, returning True only when all succeed."""
        succeeded = True
        for job_id in job_ids:
            job_succeeded = await self.run_job_once(job_id, raise_on_error=raise_on_error)
            succeeded = succeeded and job_succeeded
        return succeeded

    def _register_jobs(self) -> None:
        self._scheduler.remove_all_jobs()
        self._register_legacy_jobs()
        self._register_rate_jobs()

    def _register_legacy_jobs(self) -> None:
        # TODO: remove once legacy scheduler consumers migrate to rate-hub jobs.
        if self.agmarknet is not None:
            self._scheduler.add_job(
                self._run_agmarknet,
                trigger="cron",
                hour=10,
                minute=0,
                id="agmarknet_daily",
                replace_existing=True,
                misfire_grace_time=3600,
                name="Agmarknet Daily Price Scrape",
            )
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

    def _register_rate_jobs(self) -> None:
        for spec in RATE_JOB_SPECS:
            self._scheduler.add_job(
                self._run_rate_refresh,
                trigger="cron",
                hour=spec["hour"],
                minute=spec["minute"],
                id=spec["id"],
                replace_existing=True,
                misfire_grace_time=3600,
                name=spec["name"],
                kwargs={"job_id": spec["id"], "targets": spec["targets"]},
            )

    async def _run_agmarknet(self, *, raise_on_error: bool = False) -> bool:
        logger.info("[ScraperScheduler] Running agmarknet_daily job")
        if self.agmarknet is None:
            message = "[ScraperScheduler] agmarknet_daily requested without agmarknet scraper"
            logger.warning(message)
            if raise_on_error:
                raise RuntimeError(message)
            return False

        try:
            count = await self.agmarknet.scrape_and_store(db=self.db)
            logger.info("[ScraperScheduler] agmarknet_daily stored {} records", count)
            return True
        except Exception as exc:
            logger.error("[ScraperScheduler] agmarknet_daily failed: {}", exc)
            if raise_on_error:
                raise
            return False

    async def _run_imd(self, *, raise_on_error: bool = False) -> bool:
        logger.info("[ScraperScheduler] Running imd_daily job")
        if self.imd is None:
            message = "[ScraperScheduler] imd_daily requested without IMD scraper"
            logger.warning(message)
            if raise_on_error:
                raise RuntimeError(message)
            return False

        try:
            result = await self.imd.scrape(state="Karnataka", include_advisory=True)
            logger.info("[ScraperScheduler] imd_daily stored {} weather records", result.record_count)
            return True
        except Exception as exc:
            logger.error("[ScraperScheduler] imd_daily failed: {}", exc)
            if raise_on_error:
                raise
            return False

    async def _run_rate_refresh(
        self,
        job_id: str,
        targets: list[dict[str, Any]],
        *,
        raise_on_error: bool = False,
    ) -> bool:
        from src.rates.factory import get_rate_service

        service = await get_rate_service(db_client=self.db, agmarknet_api_key=get_agmarknet_api_key())
        logger.info("[ScraperScheduler] Running {} with {} targets", job_id, len(targets))
        succeeded = True
        for target in targets:
            try:
                query = normalize_rate_query(force_live=True, state="Karnataka", **target)
                await service.query(query)
            except Exception as exc:
                logger.error("[ScraperScheduler] {} target failed: {}", job_id, exc)
                if raise_on_error:
                    raise
                succeeded = False
        return succeeded


def get_scraper_scheduler(agmarknet: Any = None, imd: Any = None, db: Any = None) -> ScraperScheduler:
    """Factory for ScraperScheduler with compatibility defaults."""
    if agmarknet is None:
        from src.scrapers.agmarknet import get_agmarknet_scraper

        agmarknet = get_agmarknet_scraper()
    return ScraperScheduler(agmarknet=agmarknet, imd=imd, db=db)


def get_available_scraper_job_ids() -> tuple[str, ...]:
    """Return the supported one-shot job IDs for CLI and workflow wiring."""
    return ALL_SCRAPER_JOB_IDS
