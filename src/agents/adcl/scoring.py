"""
ADCL Agent — Scoring & Green-Label
====================================
Merges price forecasts + seasonal fit into demand records and
applies the green-label business rule.

Green-label rule (task35 upgrade):
    demand_score > 0.6
    AND demand_trend in ('rising', 'stable')
    AND sow_season_fit != 'not_sow_season'
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
        #! Fixed: now reads "demand_trend" (volume-based) from aggregate_demand()
        demand_trend = rec["demand_trend"]
        buyer_count = rec["buyer_count"]
        total_demand_kg = rec["total_demand_kg"]

        # Harvest-season fit (backward-compatible)
        seasonal_fit = _CALENDAR.get_fit(commodity, current_month)

        # * Sow-season fit — is NOW a good time to plant this crop?
        sow_season_fit = _CALENDAR.get_sow_fit(commodity, current_month)

        # Price forecast (₹/kg); 0.0 when price agent not available
        predicted_price = price_forecasts.get(commodity, 0.0)

        # * Actual price trend — derived from price forecast vs baseline
        # TODO: Replace with real historical price comparison when data available
        price_trend = _derive_price_trend(predicted_price, demand_trend)

        #! Updated green-label rule: uses demand_trend + sow_season_fit
        green_label = (
            demand_score > 0.6
            and demand_trend in ("rising", "stable")
            and sow_season_fit != "not_sow_season"
        )

        # Human-readable recommendation
        recommendation = _build_recommendation(
            commodity, green_label, demand_score,
            demand_trend, sow_season_fit, predicted_price,
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
            demand_trend=demand_trend,
            sow_season_fit=sow_season_fit,
        ))

    # Sort: green first, then demand_score descending
    crops.sort(key=lambda c: (not c.green_label, -c.demand_score))
    return crops


def _derive_price_trend(
    predicted_price: float,
    demand_trend: str,
) -> str:
    """
    Derive an approximate price trend.

    When full price history is available this should compare predicted
    price vs recent average price. For now, we use demand_trend as a
    proxy — rising demand typically correlates with rising prices.

    TODO: Integrate real price history from PricePredictionAgent.
    """
    if predicted_price <= 0:
        # No price data — fall back to demand trend as proxy
        return demand_trend
    # ? When we have price history, replace this with:
    # ?   if predicted > avg_last_30d * 1.10: return "rising"
    return demand_trend


def _build_recommendation(
    commodity: str,
    green_label: bool,
    demand_score: float,
    demand_trend: str,
    sow_season_fit: str,
    predicted_price: float,
) -> str:
    """Build a short English recommendation sentence."""
    name = commodity.title()
    price_str = f"₹{predicted_price:.0f}/kg" if predicted_price > 0 else "price TBD"

    if green_label:
        sow_note = (
            "ideal time to sow"
            if sow_season_fit == "ideal_sow"
            else "still within sowing window"
        )
        return (
            f"✅ Grow {name} — high buyer demand (score {demand_score:.0%}), "
            f"{demand_trend} demand ({price_str}), {sow_note}."
        )
    if sow_season_fit == "not_sow_season":
        return f"⚠️ {name} is not in its sowing season this month — consider alternative crops."
    if demand_trend == "falling":
        return f"📉 {name} demand is falling — evaluate carefully before planting."
    return (
        f"ℹ️ {name} has moderate demand (score {demand_score:.0%}); "
        f"monitor prices ({price_str}) before committing."
    )
