"""
ADCL Agent — Scoring & Green-Label
====================================
Merges price forecasts + seasonal fit into demand records and
applies the green-label business rule.

Green-label rule (task12.md AC 2):
    demand_score > 0.6
    AND price_trend in ('rising', 'stable')
    AND seasonal_fit != 'off_season'
"""

# * ADCL SCORING MODULE
# NOTE: Pure function — no I/O, fully testable.

from __future__ import annotations

from typing import Any

from src.agents.adcl.models import ADCLCrop
from src.agents.adcl.seasonal import SeasonalCalendar

_CALENDAR = SeasonalCalendar()


def score_and_label(
    demand_records: list[dict[str, Any]],
    price_forecasts: dict[str, float],
    current_month: int,
) -> list[ADCLCrop]:
    """
    Convert raw demand records into labelled ADCLCrop objects.

    Args:
        demand_records  : Output of aggregate_demand().
        price_forecasts : {commodity_lower: predicted_price_per_kg}
        current_month   : Month number 1–12 (used for seasonal fit).

    Returns:
        List of ADCLCrop sorted: green-labelled first, then demand_score desc.
    """
    crops: list[ADCLCrop] = []

    for rec in demand_records:
        commodity = rec["commodity"]
        demand_score = rec["demand_score"]
        price_trend = rec["price_trend"]
        buyer_count = rec["buyer_count"]
        total_demand_kg = rec["total_demand_kg"]

        # Seasonal fit
        seasonal_fit = _CALENDAR.get_fit(commodity, current_month)

        # Price forecast (₹/kg); 0.0 when price agent not available
        predicted_price = price_forecasts.get(commodity, 0.0)

        # Green-label business rule
        green_label = (
            demand_score > 0.6
            and price_trend in ("rising", "stable")
            and seasonal_fit != "off_season"
        )

        # Human-readable recommendation
        recommendation = _build_recommendation(
            commodity, green_label, demand_score, price_trend, seasonal_fit, predicted_price
        )

        crops.append(ADCLCrop(
            commodity=commodity,
            demand_score=demand_score,
            predicted_price_per_kg=predicted_price,
            price_trend=price_trend,
            seasonal_fit=seasonal_fit,
            green_label=green_label,
            buyer_count=buyer_count,
            total_demand_kg=total_demand_kg,
            recommendation=recommendation,
        ))

    # Sort: green first, then demand_score descending
    crops.sort(key=lambda c: (not c.green_label, -c.demand_score))
    return crops


def _build_recommendation(
    commodity: str,
    green_label: bool,
    demand_score: float,
    price_trend: str,
    seasonal_fit: str,
    predicted_price: float,
) -> str:
    """Build a short English recommendation sentence."""
    name = commodity.title()
    price_str = f"₹{predicted_price:.0f}/kg" if predicted_price > 0 else "price TBD"

    if green_label:
        return (
            f"✅ Grow {name} — high buyer demand (score {demand_score:.0%}), "
            f"{price_trend} prices ({price_str}), {seasonal_fit.replace('_', ' ')}."
        )
    if seasonal_fit == "off_season":
        return f"⚠️ {name} is off-season this month — consider alternative crops."
    if price_trend == "falling":
        return f"📉 {name} prices are falling — evaluate carefully before planting."
    return (
        f"ℹ️ {name} has moderate demand (score {demand_score:.0%}); "
        f"monitor prices ({price_str}) before committing."
    )
