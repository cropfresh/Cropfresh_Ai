"""One-shot CLI for GitHub Actions scraper and rate-refresh jobs."""

from __future__ import annotations

import argparse
import asyncio
import os

from loguru import logger

from src.db.postgres_client import AuroraPostgresClient
from src.scrapers.agmarknet import get_agmarknet_scraper
from src.scrapers.agri_scrapers.imd import IMDWeatherScraper
from src.scrapers.scheduler_runtime import (
    ScraperScheduler,
    get_available_scraper_job_ids,
)


def _create_scheduler(db_client: AuroraPostgresClient | None = None) -> ScraperScheduler:
    """Create the default scheduler used by the scraper workflow CLI."""
    return ScraperScheduler(
        agmarknet=get_agmarknet_scraper(),
        imd=IMDWeatherScraper(),
        db=db_client,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CropFresh scraper jobs once and exit")
    parser.add_argument(
        "--job-id",
        action="append",
        dest="job_ids",
        choices=get_available_scraper_job_ids(),
        help="Named job to execute. Repeat for multiple jobs.",
    )
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="Print all supported job IDs and exit.",
    )
    return parser


async def _maybe_create_db_client() -> AuroraPostgresClient | None:
    """Create a DB client when PG settings are available in the environment."""
    if not os.getenv("PG_HOST"):
        logger.info("PG_HOST not configured; scraper CLI will run without DB persistence")
        return None

    client = AuroraPostgresClient(
        host=os.getenv("PG_HOST", ""),
        database=os.getenv("PG_DATABASE", "postgres"),
        port=int(os.getenv("PG_PORT", "5432")),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", ""),
        region=os.getenv("AWS_REGION", "ap-south-1"),
        use_iam_auth=os.getenv("PG_USE_IAM_AUTH", "false").lower() == "true",
    )
    try:
        await client.connect()
        return client
    except Exception as exc:
        logger.warning(
            "Scraper CLI could not connect to PostgreSQL; continuing without DB persistence: {}",
            exc,
        )
        return None


async def _run(job_ids: list[str]) -> int:
    db_client = await _maybe_create_db_client()
    scheduler = _create_scheduler(db_client=db_client)
    try:
        await scheduler.run_jobs_once(job_ids, raise_on_error=True)
        logger.info("Completed one-shot scraper jobs: {}", ", ".join(job_ids))
    finally:
        if db_client is not None:
            await db_client.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_jobs:
        for job_id in get_available_scraper_job_ids():
            print(job_id)
        return 0

    if not args.job_ids:
        parser.error("at least one --job-id is required unless --list-jobs is used")

    return asyncio.run(_run(args.job_ids))


if __name__ == "__main__":
    raise SystemExit(main())
