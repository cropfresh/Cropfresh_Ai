"""Tool entry points for the shared multi-source Karnataka rate hub."""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.rates.factory import get_rate_service
from src.rates.query_builder import normalize_rate_query
from src.rates.settings import get_agmarknet_api_key
from src.tools.registry import get_tool_registry


async def multi_source_rates_tool(
    rate_kinds: list[str],
    commodity: str | None = None,
    state: str = "Karnataka",
    district: str | None = None,
    market: str | None = None,
    date: str | None = None,
    include_reference: bool = True,
    force_live: bool = False,
    comparison_depth: str = "all_sources",
) -> dict[str, Any]:
    """Fetch evidence-backed rates across supported sources."""
    query = normalize_rate_query(
        rate_kinds=rate_kinds,
        commodity=commodity,
        state=state,
        district=district,
        market=market,
        date=date,
        include_reference=include_reference,
        force_live=force_live,
        comparison_depth=comparison_depth,
    )
    service = await get_rate_service(agmarknet_api_key=get_agmarknet_api_key())
    result = await service.query(query)
    return result.model_dump(mode="json")


async def price_api_tool(
    commodity: str,
    state: str = "Karnataka",
    district: str | None = None,
    market: str | None = None,
    location: str | None = None,
    date: str | None = None,
    force_live: bool = False,
) -> dict[str, Any]:
    """Backward-compatible mandi-only alias over the rate hub."""
    return await multi_source_rates_tool(
        rate_kinds=["mandi_wholesale"],
        commodity=commodity,
        state=state,
        district=district,
        market=market or location,
        date=date,
        include_reference=True,
        force_live=force_live,
        comparison_depth="all_sources",
    )


try:
    registry = get_tool_registry()
    registry.add_tool(
        multi_source_rates_tool,
        name="multi_source_rates",
        description=(
            "Fetch Karnataka mandi, retail produce, fuel, gold, or support prices "
            "from multiple sources with official-first comparison evidence."
        ),
        category="rates",
    )
    registry.add_tool(
        price_api_tool,
        name="price_api",
        description="Backward-compatible mandi-only alias for multi-source rate queries.",
        category="rates",
    )
except Exception as exc:
    logger.debug("Rate tool registration deferred: {}", exc)
