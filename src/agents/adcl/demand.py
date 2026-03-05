"""
ADCL Agent — Demand Aggregation
================================
Aggregates raw buyer order data into per-commodity demand records
with 30/60/90-day trend analysis.

Input  : list of order dicts (commodity, quantity_kg, buyer_id, created_at)
Output : list of demand dicts ready for scoring.score_and_label()
"""

# * ADCL DEMAND MODULE
# NOTE: Pure functions — no I/O, fully testable in isolation.

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any


def aggregate_demand(
    orders: list[dict[str, Any]],
    reference_date: date | None = None,
) -> list[dict[str, Any]]:
    """
    Aggregate buyer orders into per-commodity demand records.

    Algorithm:
    1. Group orders by commodity.
    2. Sum quantity_kg and count unique buyers.
    3. Compare 30-day demand vs 60–90-day daily average to compute trend.
    4. Normalise demand_score using percentile ranking so multiple
       crops can independently exceed the 0.6 green-label threshold.

    Args:
        orders        : List of order dicts with keys:
                        - commodity    (str)
                        - quantity_kg  (float)
                        - buyer_id     (str)
                        - created_at   (str ISO-8601 or date)
        reference_date: Treat as "today" (default: today).

    Returns:
        List of demand dicts sorted by total_demand_kg descending. Each dict:
        {
            "commodity"      : str,
            "total_demand_kg": float,
            "buyer_count"    : int,
            "demand_score"   : float,   # 0–1, percentile-rank normalised
            "demand_trend"   : str,     # 'rising' | 'stable' | 'falling'
        }
    """
    if not orders:
        return []

    today = reference_date or date.today()
    cutoff_30 = today - timedelta(days=30)
    cutoff_60 = today - timedelta(days=60)
    cutoff_90 = today - timedelta(days=90)

    # Per-commodity buckets
    total_kg: dict[str, float] = defaultdict(float)
    buyers: dict[str, set[str]] = defaultdict(set)
    kg_last_30: dict[str, float] = defaultdict(float)
    kg_30_to_90: dict[str, float] = defaultdict(float)

    for order in orders:
        commodity = str(order.get("commodity", "")).strip().lower()
        if not commodity:
            continue

        qty = float(order.get("quantity_kg", 0.0))
        buyer = str(order.get("buyer_id", ""))

        # Parse date
        created_raw = order.get("created_at")
        if isinstance(created_raw, datetime):
            order_date = created_raw.date()
        elif isinstance(created_raw, date):
            order_date = created_raw
        else:
            try:
                order_date = date.fromisoformat(str(created_raw)[:10])
            except (ValueError, TypeError):
                order_date = today  # fallback: treat as today

        # Only consider last 90 days
        if order_date < cutoff_90:
            continue

        total_kg[commodity] += qty
        buyers[commodity].add(buyer)

        if order_date >= cutoff_30:
            kg_last_30[commodity] += qty
        elif order_date >= cutoff_60:
            kg_30_to_90[commodity] += qty
        else:
            kg_30_to_90[commodity] += qty

    if not total_kg:
        return []

    # * Percentile-rank normalisation (replaces simple max-normalisation)
    # Each crop's score = fraction of crops it beats or ties on total_demand_kg
    sorted_totals = sorted(total_kg.values())
    n = len(sorted_totals)

    def _percentile_score(value: float) -> float:
        """Percentile rank in [0, 1]. Crops at the top get 1.0."""
        if n <= 1:
            return 1.0
        # Count how many values are strictly less than this one
        rank = sum(1 for v in sorted_totals if v < value)
        return rank / (n - 1)

    results: list[dict[str, Any]] = []
    for commodity, total in total_kg.items():
        demand_score = _percentile_score(total)

        # Trend: compare last-30 daily rate vs prior-60 daily rate
        last30_rate = kg_last_30[commodity] / 30.0
        prior_rate = (
            kg_30_to_90[commodity] / 60.0
            if kg_30_to_90[commodity] > 0
            else 0.0
        )

        if prior_rate == 0:
            # New commodity — treat as rising
            trend = "rising"
        elif last30_rate >= prior_rate * 1.15:
            trend = "rising"
        elif last30_rate <= prior_rate * 0.85:
            trend = "falling"
        else:
            trend = "stable"

        results.append({
            "commodity": commodity,
            "total_demand_kg": round(total, 2),
            "buyer_count": len(buyers[commodity]),
            "demand_score": round(demand_score, 4),
            #! Fixed: was "price_trend" — now correctly named "demand_trend"
            "demand_trend": trend,
        })

    # Sort by total_demand_kg descending
    results.sort(key=lambda r: r["total_demand_kg"], reverse=True)
    return results
