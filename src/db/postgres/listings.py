"""
Produce and listing CRUD mixin for Aurora PostgreSQL.
"""

import json
from typing import Any, Optional

from loguru import logger

from src.db.postgres.listing_queries import (
    build_listing_insert_payload,
    build_listing_search,
)


class ListingOperationsMixin:
    """
    Mixin providing produce listing and search operations.

    Requires the parent class to expose a `pool` property
    returning an asyncpg connection pool.
    """

    async def list_produce(
        self,
        crop_name: Optional[str] = None,
        quality_grade: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List available produce with optional filters."""
        conditions = ["status = 'available'"]
        params: list[Any] = []
        idx = 1

        if crop_name:
            conditions.append(f"crop_name ILIKE ${idx}")
            params.append(f"%{crop_name}%")
            idx += 1

        if quality_grade:
            conditions.append(f"quality_grade = ${idx}")
            params.append(quality_grade)
            idx += 1

        where = " AND ".join(conditions)
        params.append(limit)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM produce WHERE {where} LIMIT ${idx}",
                *params,
            )

        return [dict(row) for row in rows]

    async def create_produce_listing(
        self,
        farmer_id: str,
        crop_name: str,
        quantity_kg: float,
        price_per_kg: float,
        quality_grade: str = "B",
        location: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Create a new produce listing."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO produce
                    (farmer_id, crop_name, quantity_kg, price_per_kg, quality_grade, location, status)
                VALUES ($1::uuid, $2, $3, $4, $5, $6::jsonb, 'available')
                RETURNING *
                """,
                farmer_id, crop_name, quantity_kg, price_per_kg,
                quality_grade, json.dumps(location or {}),
            )

        logger.info(f"Created listing: {quantity_kg}kg {crop_name}")
        return dict(row) if row else {}

    async def get_listing(self, listing_id: str) -> Optional[dict[str, Any]]:
        """Fetch a single listing by UUID, joined with farmer details."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT l.*, f.name AS farmer_name, f.district AS farmer_district
                FROM listings l
                JOIN farmers f ON f.id = l.farmer_id
                WHERE l.id = $1::uuid
                """,
                listing_id,
            )
        return dict(row) if row else None

    async def create_listing(self, listing_data: dict[str, Any]) -> str:
        """
        Create a new produce listing.

        Args:
            listing_data: Dict with keys: farmer_id, commodity, quantity_kg,
                          asking_price_per_kg, and optional grade, variety,
                          harvest_date, pickup_window_start, pickup_window_end,
                          batch_qr_code, expires_at.

        Returns:
            New listing UUID as string.
        """
        columns, values = build_listing_insert_payload(listing_data)
        placeholders = ", ".join(
            f"${index}::uuid" if column == "farmer_id" else f"${index}"
            for index, column in enumerate(columns, start=1)
        )
        async with self.pool.acquire() as conn:
            listing_id = await conn.fetchval(
                f"""
                INSERT INTO listings ({', '.join(columns)})
                VALUES ({placeholders})
                RETURNING id
                """,
                *values,
            )
        logger.info(
            f"Created listing: {listing_data['quantity_kg']}kg "
            f"{listing_data['commodity']} by farmer {listing_data['farmer_id']}"
        )
        return str(listing_id)

    async def search_listings(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Search active listings with optional filters.

        Args:
            filters: Dict with optional keys: commodity, grade, district,
                     min_qty_kg, max_price_per_kg, limit (default 50).

        Returns:
            List of matching listing dicts.
        """
        sql, params = build_listing_search(filters)
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [dict(row) for row in rows]
