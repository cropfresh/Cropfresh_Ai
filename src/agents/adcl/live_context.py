"""Live-source helpers for IMD and eNAM ADCL enrichment."""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.agents.adcl.time_utils import utc_now_iso


async def fetch_imd_context(
    imd_client: Any | None,
    district: str,
    commodities: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Fetch live IMD advisories for the top commodities when configured."""
    if imd_client is None or getattr(imd_client, "use_mock", False):
        return {}, _disabled_source("imd", "Live IMD client not configured"), {}

    advisories: dict[str, dict[str, Any]] = {}
    freshness: dict[str, Any] = {}
    try:
        for commodity in commodities[:5]:
            advisory = await imd_client.get_agro_advisory(
                state="Karnataka",
                district=district,
                crop=commodity,
            )
            advisories[commodity] = advisory.model_dump(mode="json")
            freshness[commodity] = advisories[commodity].get("valid_until", "")
        return advisories, {
            "status": "healthy",
            "checked_at": utc_now_iso(),
            "crop_count": len(advisories),
        }, freshness
    except Exception as exc:
        logger.warning("ADCL IMD enrichment failed: {}", exc)
        return {}, {
            "status": "degraded",
            "checked_at": utc_now_iso(),
            "error": str(exc),
        }, {}


async def fetch_enam_context(
    enam_client: Any | None,
    district: str,
    commodities: list[str],
    enabled: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Fetch gated eNAM pricing evidence when credentials are available."""
    if not enabled:
        return {}, _disabled_source("enam", "Feature flag disabled"), {}
    if enam_client is None or getattr(enam_client, "use_mock", False):
        return {}, _disabled_source("enam", "Live eNAM credentials unavailable"), {}

    snapshots: dict[str, dict[str, Any]] = {}
    freshness: dict[str, Any] = {}
    try:
        for commodity in commodities[:5]:
            prices = await enam_client.get_live_prices(
                commodity=commodity,
                state="Karnataka",
                district=district,
                limit=1,
            )
            if prices:
                payload = prices[0].model_dump(mode="json")
                snapshots[commodity] = payload
                freshness[commodity] = payload.get("last_updated", "")
        return snapshots, {
            "status": "healthy",
            "checked_at": utc_now_iso(),
            "crop_count": len(snapshots),
        }, freshness
    except Exception as exc:
        logger.warning("ADCL eNAM enrichment failed: {}", exc)
        return {}, {
            "status": "degraded",
            "checked_at": utc_now_iso(),
            "error": str(exc),
        }, {}


def _disabled_source(source: str, reason: str) -> dict[str, Any]:
    return {
        "status": "disabled" if source == "imd" else "gated",
        "checked_at": utc_now_iso(),
        "reason": reason,
    }
