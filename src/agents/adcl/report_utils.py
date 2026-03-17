"""Helpers for shaping ADCL report metadata, evidence, and empty states."""

from __future__ import annotations

from datetime import date
from typing import Any

from src.agents.adcl.models import ADCLCrop, WeeklyReport
from src.agents.adcl.time_utils import utc_now_iso


def base_source_health(order_count: int) -> dict[str, Any]:
    """Build the base source-health block from live marketplace orders."""
    return {
        "orders": {
            "status": "healthy" if order_count else "unavailable",
            "checked_at": utc_now_iso(),
            "order_count": order_count,
        },
    }


def rate_status(health_items: list[Any]) -> str:
    """Collapse rate-hub source health into one top-level status."""
    if not health_items:
        return "disabled"
    statuses = {item.status for item in health_items}
    if statuses == {"healthy"}:
        return "healthy"
    if "healthy" in statuses:
        return "degraded"
    return "unavailable"


def attach_crop_context(
    crops: list[ADCLCrop],
    price_signals: dict[str, dict[str, Any]],
    source_health: dict[str, Any],
    imd_context: dict[str, dict[str, Any]],
    enam_context: dict[str, dict[str, Any]],
) -> None:
    """Attach evidence, freshness, and source health to each crop."""
    for crop in crops:
        signal = price_signals.get(crop.commodity, {})
        crop.predicted_price_per_kg = signal.get(
            "predicted_price_per_kg",
            crop.predicted_price_per_kg,
        )
        crop.price_trend = signal.get("price_trend", crop.price_trend)
        crop.evidence = build_crop_evidence(
            crop=crop,
            price_signal=signal,
            imd_advisory=imd_context.get(crop.commodity),
            enam_snapshot=enam_context.get(crop.commodity),
        )
        crop.freshness = {
            "price": signal.get("latest_price_date", ""),
            "generated_at": utc_now_iso(),
        }
        crop.source_health = source_health


def build_empty_report(
    district: str,
    week_start: date,
    source_health: dict[str, Any],
    metadata: dict[str, Any],
) -> WeeklyReport:
    """Return the persisted empty-report state for districts with no live orders."""
    return WeeklyReport(
        week_start=week_start,
        district=district,
        crops=[],
        summary_en="No live order data is available for this district this week.",
        summary_hi="Is zille ke liye is hafte live order data uplabdh nahin hai.",
        summary_kn="Ee jillege ee vara live order data labhyavilla.",
        freshness={"generated_at": utc_now_iso()},
        source_health=source_health,
        metadata=metadata,
    )


def report_metadata(
    force_live: bool,
    farmer_id: str | None,
    language: str | None,
    crop_count: int,
    green_count: int = 0,
) -> dict[str, Any]:
    """Return report metadata used by callers and diagnostics."""
    coverage = round(green_count / crop_count, 3) if crop_count else 0.0
    return {
        "force_live": force_live,
        "farmer_id": farmer_id,
        "language": language,
        "crop_count": crop_count,
        "green_count": green_count,
        "recommendation_coverage": coverage,
    }


def build_crop_evidence(
    crop: ADCLCrop,
    price_signal: dict[str, Any],
    imd_advisory: dict[str, Any] | None,
    enam_snapshot: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Create the per-crop evidence list returned in the canonical payload."""
    evidence = [
        {
            "source": "orders",
            "detail": f"{crop.buyer_count} buyers ordered {crop.total_demand_kg:.1f} kg in the last 90 days.",
        },
        {
            "source": "price_context",
            "detail": (
                "Predicted price "
                f"{price_signal.get('predicted_price_per_kg', 0.0):.2f} Rs/kg "
                f"with {price_signal.get('price_trend', 'stable')} trend."
            ),
        },
    ]
    if imd_advisory:
        evidence.append(
            {
                "source": "imd",
                "detail": imd_advisory.get("sowing_advisory")
                or imd_advisory.get("weather_summary", ""),
            }
        )
    if enam_snapshot:
        evidence.append(
            {
                "source": "enam",
                "detail": (
                    f"{enam_location(enam_snapshot)}: "
                    f"{enam_snapshot.get('modal_price', 0.0):.0f} Rs/quintal"
                ),
            }
        )
    return evidence


def enam_location(snapshot: dict[str, Any]) -> str:
    """Return the most useful human-readable eNAM location label."""
    return str(snapshot.get("market") or snapshot.get("district") or "eNAM")
