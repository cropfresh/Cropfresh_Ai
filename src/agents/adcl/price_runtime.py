"""Runtime helpers for collecting ADCL price signals from history and live rates."""

from __future__ import annotations

from typing import Any

from src.agents.adcl.price_context import build_price_signal
from src.agents.adcl.report_utils import rate_status
from src.rates.query_builder import normalize_rate_query


async def build_price_signals(
    repository: Any,
    rate_service: Any | None,
    district: str,
    commodities: list[str],
    force_live: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Build price signals, freshness, and health from shared rate infrastructure."""
    signals: dict[str, dict[str, Any]] = {}
    freshness: dict[str, Any] = {}
    rate_health = {"status": "disabled", "sources": []}

    for commodity in commodities[:10]:
        history = await repository.get_price_history(commodity, district, days=30)
        live_rate = await get_live_rate(rate_service, commodity, district, force_live)
        signals[commodity] = build_price_signal(history, live_rate)
        freshness[commodity] = signals[commodity]["latest_price_date"]

    if rate_service is not None:
        health = rate_service.get_source_health()
        rate_health = {
            "status": rate_status(health),
            "sources": [item.model_dump(mode="json") for item in health],
        }

    return signals, rate_health, freshness


async def get_live_rate(
    rate_service: Any | None,
    commodity: str,
    district: str,
    force_live: bool,
) -> dict[str, Any] | None:
    """Fetch one canonical live rate from the shared rate hub."""
    if rate_service is None:
        return None

    result = await rate_service.query(
        normalize_rate_query(
            rate_kinds=["mandi_wholesale"],
            commodity=commodity,
            district=district,
            state="Karnataka",
            force_live=force_live,
            comparison_depth="official_plus_validators",
        )
    )
    if not result.canonical_rates:
        return None
    return result.canonical_rates[0].model_dump(mode="json")
