"""Tests for the one-shot scraper CLI entrypoint."""

from __future__ import annotations

import pytest

from src.scrapers import __main__ as scraper_cli


def test_scraper_cli_lists_supported_jobs(capsys) -> None:
    exit_code = scraper_cli.main(["--list-jobs"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "agmarknet_daily" in captured.out
    assert "rates_official_mandi_morning" in captured.out


def test_scraper_cli_requires_job_id_when_not_listing() -> None:
    with pytest.raises(SystemExit):
        scraper_cli.main([])


@pytest.mark.asyncio
async def test_scraper_cli_run_dispatches_requested_jobs(monkeypatch) -> None:
    captured_job_ids: list[str] = []

    class FakeScheduler:
        async def run_jobs_once(self, job_ids: list[str], *, raise_on_error: bool = True) -> bool:
            captured_job_ids.extend(job_ids)
            assert raise_on_error is True
            return True

    async def fake_create_db_client():
        return None

    monkeypatch.setattr(scraper_cli, "_maybe_create_db_client", fake_create_db_client)
    monkeypatch.setattr(scraper_cli, "_create_scheduler", lambda db_client=None: FakeScheduler())

    exit_code = await scraper_cli._run(
        ["agmarknet_daily", "rates_fuel_gold_morning", "rates_validator_retail_daily"]
    )

    assert exit_code == 0
    assert captured_job_ids == [
        "agmarknet_daily",
        "rates_fuel_gold_morning",
        "rates_validator_retail_daily",
    ]
