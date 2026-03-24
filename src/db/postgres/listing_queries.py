"""Shared SQL builders for listing persistence and search."""

from __future__ import annotations

from typing import Any


def build_listing_insert_payload(listing_data: dict[str, Any]) -> tuple[list[str], list[Any]]:
    """Return column names and ordered params for listing insertion."""
    columns = [
        "farmer_id",
        "commodity",
        "variety",
        "quantity_kg",
        "asking_price_per_kg",
        "grade",
        "cv_confidence",
        "hitl_required",
        "status",
        "harvest_date",
        "pickup_window_start",
        "pickup_window_end",
        "batch_qr_code",
        "adcl_tagged",
        "expires_at",
    ]
    values = [
        listing_data["farmer_id"],
        listing_data["commodity"],
        listing_data.get("variety"),
        float(listing_data["quantity_kg"]),
        float(listing_data["asking_price_per_kg"]),
        listing_data.get("grade", "Unverified"),
        listing_data.get("cv_confidence"),
        bool(listing_data.get("hitl_required", False)),
        listing_data.get("status", "active"),
        listing_data.get("harvest_date"),
        listing_data.get("pickup_window_start"),
        listing_data.get("pickup_window_end"),
        listing_data.get("batch_qr_code"),
        bool(listing_data.get("adcl_tagged", False)),
        listing_data.get("expires_at"),
    ]
    return columns, values


def build_listing_search(filters: dict[str, Any]) -> tuple[str, list[Any]]:
    """Build a listing search SQL statement and bound params."""
    conditions = ["l.status = 'active'"]
    params: list[Any] = []
    idx = 1

    if commodity := filters.get("commodity"):
        conditions.append(f"l.commodity ILIKE ${idx}")
        params.append(f"%{commodity}%")
        idx += 1

    if grade := filters.get("grade"):
        conditions.append(f"l.grade = ${idx}")
        params.append(grade)
        idx += 1

    if farmer_id := filters.get("farmer_id"):
        conditions.append(f"l.farmer_id = ${idx}::uuid")
        params.append(farmer_id)
        idx += 1

    if district := filters.get("district"):
        conditions.append(f"f.district ILIKE ${idx}")
        params.append(f"%{district}%")
        idx += 1

    if min_qty := filters.get("min_qty_kg"):
        conditions.append(f"l.quantity_kg >= ${idx}")
        params.append(float(min_qty))
        idx += 1

    if max_price := filters.get("max_price_per_kg"):
        conditions.append(f"l.asking_price_per_kg <= ${idx}")
        params.append(float(max_price))
        idx += 1

    if "adcl_tagged" in filters:
        conditions.append(f"l.adcl_tagged = ${idx}")
        params.append(bool(filters["adcl_tagged"]))
        idx += 1

    if status := filters.get("status"):
        conditions[0] = f"l.status = ${idx}"
        params.append(status)
        idx += 1

    limit = int(filters.get("limit", 50))
    params.append(limit)
    where = " AND ".join(conditions)
    sql = f"""
        SELECT l.*, f.name AS farmer_name, f.district AS farmer_district
        FROM listings l
        JOIN farmers f ON f.id = l.farmer_id
        WHERE {where}
        ORDER BY l.created_at DESC
        LIMIT ${idx}
    """
    return sql, params
