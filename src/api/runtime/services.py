"""Startup wiring for shared database, ADCL, listing, and voice services."""

from __future__ import annotations

import os
from typing import Any

from loguru import logger

from src.agents.adcl import get_adcl_service
from src.agents.adcl.scheduler import ADCLScheduler
from src.agents.price_prediction.agent import PricePredictionAgent
from src.api.runtime.voice_agent import build_voice_agent
from src.api.services.listing_service import get_listing_service
from src.api.services.order_service import get_order_service
from src.api.services.registration_service import get_registration_service
from src.db.postgres.client import get_postgres
from src.rates.factory import get_rate_service
from src.rates.settings import get_agmarknet_api_key
from src.scrapers.enam_client.client import get_enam_client
from src.scrapers.imd_weather.client import get_imd_client


async def initialize_runtime_services(app: Any, settings: Any) -> None:
    """Attach shared runtime services to `app.state`."""
    app.state.db = await _init_db(settings)
    app.state.rate_service = await get_rate_service(
        db_client=app.state.db,
        redis_client=getattr(app.state, "redis", None),
        llm_provider=getattr(app.state, "llm", None),
        agmarknet_api_key=get_agmarknet_api_key(),
    )

    app.state.pricing_agent = PricePredictionAgent(llm=getattr(app.state, "llm", None))
    await app.state.pricing_agent.initialize()
    app.state.order_service = get_order_service(
        db=app.state.db,
        pricing_agent=app.state.pricing_agent,
    )

    app.state.registration_service = get_registration_service(db=app.state.db)
    imd_client = get_imd_client(
        imd_api_key=os.getenv("IMD_API_KEY", ""),
        owm_api_key=settings.weather_api_key,
        use_mock=False,
    )
    enam_api_key = os.getenv("ENAM_API_KEY", "")
    enable_enam = _feature_flag("ENABLE_ENAM") and bool(enam_api_key)
    enam_client = get_enam_client(api_key=enam_api_key, use_mock=False)

    app.state.adcl_service = get_adcl_service(
        db=app.state.db,
        rate_service=app.state.rate_service,
        llm=getattr(app.state, "llm", None),
        imd_client=imd_client,
        enam_client=enam_client,
        enable_enam=enable_enam,
    )
    app.state.adcl_agent = app.state.adcl_service

    app.state.listing_service = get_listing_service(
        db=app.state.db,
        pricing_agent=app.state.pricing_agent,
        quality_agent=_supervisor_agent(app, "quality_assessment_agent"),
        adcl_agent=app.state.adcl_service,
    )
    app.state.voice_agent = build_voice_agent(
        llm_provider=getattr(app.state, "llm", None),
        listing_service=app.state.listing_service,
        order_service=app.state.order_service,
        matching_agent=_supervisor_agent(app, "buyer_matching_agent"),
        quality_agent=_supervisor_agent(app, "quality_assessment_agent"),
        agronomy_agent=_supervisor_agent(app, "agronomy_agent"),
        adcl_agent=app.state.adcl_service,
        registration_service=app.state.registration_service,
        weather_api_key=settings.weather_api_key,
    )
    app.state.adcl_scheduler = _build_adcl_scheduler(app.state.adcl_service)
    if app.state.adcl_scheduler is not None:
        app.state.adcl_scheduler.start()

    logger.info(
        "Runtime services ready: db={} adcl={} listing={} voice={}",
        bool(app.state.db),
        bool(app.state.adcl_service),
        bool(app.state.listing_service),
        bool(app.state.voice_agent),
    )


async def shutdown_runtime_services(app: Any) -> None:
    """Close runtime services that own external resources."""
    scheduler = getattr(app.state, "adcl_scheduler", None)
    if scheduler is not None:
        scheduler.stop()

    db = getattr(app.state, "db", None)
    if db is not None:
        await db.close()


async def _init_db(settings: Any) -> Any | None:
    if not getattr(settings, "pg_host", ""):
        logger.warning("PG_HOST not configured; database-backed ADCL features are disabled")
        return None
    try:
        return await get_postgres()
    except Exception as exc:
        logger.warning("Database initialization failed: {}", exc)
        return None


def _build_adcl_scheduler(service: Any) -> ADCLScheduler | None:
    districts = [
        district.strip()
        for district in os.getenv(
            "ADCL_DISTRICTS",
            "Bangalore,Kolar,Mysore,Hubli",
        ).split(",")
        if district.strip()
    ]
    if not districts:
        return None
    return ADCLScheduler(service=service, districts=districts)


def _feature_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _supervisor_agent(app: Any, name: str) -> Any | None:
    supervisor = getattr(app.state, "supervisor", None)
    agents = getattr(supervisor, "_agents", {}) if supervisor is not None else {}
    return agents.get(name)
