"""
Price history CRUD mixin for Aurora PostgreSQL.
"""

from typing import Any

from loguru import logger


class PriceOperationsMixin:
    """
    Mixin providing market price history operations.

    Requires the parent class to expose a `pool` property
    returning an asyncpg connection pool.
    """

    async def insert_price_history(self, records: list[dict[str, Any]]) -> int:
        """
        Bulk insert market price records (upsert on conflict).

        Args:
            records: List of dicts with keys: commodity, district, date,
                     modal_price, and optional min_price, max_price, source, state.

        Returns:
            Number of rows inserted/updated.
        """
        rows = [
            (
                rec["commodity"],
                rec["district"],
                rec.get("state", "Karnataka"),
                rec["date"],
                float(rec["modal_price"]),
                float(rec["min_price"]) if rec.get("min_price") is not None else None,
                float(rec["max_price"]) if rec.get("max_price") is not None else None,
                rec.get("source", "agmarknet"),
            )
            for rec in records
        ]

        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO price_history
                    (commodity, district, state, date, modal_price,
                     min_price, max_price, source)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (commodity, district, date, source)
                DO UPDATE SET
                    modal_price = EXCLUDED.modal_price,
                    min_price   = EXCLUDED.min_price,
                    max_price   = EXCLUDED.max_price
                """,
                rows,
            )
        logger.info(f"Upserted {len(rows)} price history records")
        return len(rows)

    async def insert_mandi_prices(self, records: list[dict[str, Any]]) -> int:
        """Backward-compatible alias for legacy scraper callers."""
        return await self.insert_price_history(records)

    async def get_price_history(
        self,
        commodity: str,
        district: str,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """
        Fetch recent market prices for a commodity + district.

        Args:
            commodity: Crop name (case-insensitive match).
            district: Karnataka district name.
            days: Number of calendar days to look back (default 30).

        Returns:
            List of price records sorted newest-first.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT commodity, district, state, date,
                       modal_price, min_price, max_price, source
                FROM price_history
                WHERE commodity ILIKE $1
                  AND district ILIKE $2
                  AND date >= CURRENT_DATE - ($3 * INTERVAL '1 day')
                ORDER BY date DESC
                """,
                f"%{commodity}%",
                f"%{district}%",
                days,
            )
        return [dict(row) for row in rows]
