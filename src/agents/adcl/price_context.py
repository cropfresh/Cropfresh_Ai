"""Pure helpers for deriving ADCL price context."""

from __future__ import annotations

from statistics import mean
from typing import Any


def build_price_signal(
    history: list[dict[str, Any]],
    live_rate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine historical price rows and an optional live rate into one signal."""
    history_prices = [_history_price_per_kg(row) for row in history]
    history_prices = [price for price in history_prices if price > 0]
    live_price = _live_price_per_kg(live_rate)
    latest_price = live_price or (history_prices[0] if history_prices else 0.0)

    if history_prices:
        recent_window = history_prices[:7]
        prior_window = history_prices[7:21] or history_prices[:7]
        recent_avg = mean(recent_window)
        prior_avg = mean(prior_window)
        trend = derive_trend(recent_avg, prior_avg)
        predicted_price = round((latest_price + recent_avg) / 2, 2)
        latest_date = str(history[0].get("date", ""))
    elif live_price:
        trend = "stable"
        predicted_price = round(live_price, 2)
        latest_date = str(live_rate.get("price_date", ""))
    else:
        trend = "stable"
        predicted_price = 0.0
        latest_date = ""

    return {
        "predicted_price_per_kg": predicted_price,
        "price_trend": trend,
        "latest_price_per_kg": round(latest_price, 2),
        "history_count": len(history_prices),
        "latest_price_date": latest_date,
        "live_rate_used": bool(live_price),
    }


def derive_trend(recent_avg: float, prior_avg: float) -> str:
    """Return a simple rising/stable/falling trend label."""
    if prior_avg <= 0:
        return "stable"
    if recent_avg >= prior_avg * 1.08:
        return "rising"
    if recent_avg <= prior_avg * 0.92:
        return "falling"
    return "stable"


def _history_price_per_kg(row: dict[str, Any]) -> float:
    value = float(row.get("modal_price", 0.0) or 0.0)
    return round(value / 100, 2) if value else 0.0


def _live_price_per_kg(rate: dict[str, Any] | None) -> float:
    if not rate:
        return 0.0
    modal_price = float(rate.get("modal_price", 0.0) or 0.0)
    unit = str(rate.get("unit", "")).lower()
    if not modal_price:
        return 0.0
    if "quintal" in unit:
        return round(modal_price / 100, 2)
    return round(modal_price, 2)
