import pytest

pytest.importorskip("apscheduler.schedulers.asyncio")

from src.scrapers.scheduler_runtime import RATE_JOB_SPECS, ScraperScheduler


@pytest.mark.asyncio
async def test_rate_scheduler_registers_new_jobs() -> None:
    scheduler = ScraperScheduler(agmarknet=None, imd=None, db=None)
    scheduler.start()
    try:
        jobs = {job["id"] for job in scheduler.get_jobs()}
    finally:
        scheduler.stop()

    expected = {spec["id"] for spec in RATE_JOB_SPECS}
    assert expected.issubset(jobs)


@pytest.mark.asyncio
async def test_rate_refresh_job_forces_live_queries(monkeypatch) -> None:
    scheduler = ScraperScheduler(agmarknet=None, imd=None, db=None)
    captured_queries = []

    class FakeService:
        async def query(self, query):
            captured_queries.append(query)

    async def fake_get_rate_service(**kwargs):
        return FakeService()

    monkeypatch.setattr("src.rates.factory.get_rate_service", fake_get_rate_service)
    await scheduler._run_rate_refresh(
        job_id="rates_fuel_gold_morning",
        targets=[{"rate_kinds": ["fuel"]}, {"rate_kinds": ["gold"]}],
    )

    assert len(captured_queries) == 2
    assert all(query.force_live is True for query in captured_queries)


@pytest.mark.asyncio
async def test_run_job_once_rejects_unknown_job_id() -> None:
    scheduler = ScraperScheduler(agmarknet=None, imd=None, db=None)

    with pytest.raises(ValueError):
        await scheduler.run_job_once("not-a-real-job")
