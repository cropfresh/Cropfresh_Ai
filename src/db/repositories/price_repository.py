"""
Repository for price records (raw and normalized).
"""
import json
from datetime import date
from typing import Any, List

from loguru import logger

from src.db.models.price_records import NormalizedPriceRecord, RawPriceRecord


class PriceRepository:
    """Handles data ingestion and basic retrieval queries for prices."""

    def __init__(self, db_client: Any):
        """Initialize with an AuroraPostgresClient instance."""
        self.db = db_client
        logger.info("Initializing PriceRepository")

    async def initialize_schema(self):
        """Create price tables if they don't exist."""
        from pathlib import Path
        schema_path = Path(__file__).parent.parent / "schema_prices.sql"
        if schema_path.exists():
            schema_sql = schema_path.read_text(encoding="utf-8")
            async with self.db.pool.acquire() as conn:
                await conn.execute(schema_sql)
            logger.info("Initialized price tables schema from schema_prices.sql")
        else:
            logger.warning("schema_prices.sql not found; skip init.")

    async def save_raw_record(self, record: RawPriceRecord) -> str:
        """Save a raw record to source_data and return its generated ID."""
        async with self.db.pool.acquire() as conn:
            row_id = await conn.fetchval(
                """
                INSERT INTO source_data (source, raw_data, url, scraped_at)
                VALUES ($1, $2::jsonb, $3, $4)
                RETURNING id
                """,
                record.source, json.dumps(record.raw_data), record.url, record.scraped_at
            )
        return str(row_id)

    async def save_normalized_record(self, record: NormalizedPriceRecord) -> str:
        """Save a canonical price record to normalized_prices."""
        async with self.db.pool.acquire() as conn:
            row_id = await conn.fetchval(
                """
                INSERT INTO normalized_prices
                (commodity, variety, state, market, price_date, min_price, max_price, modal_price, unit, source, raw_record_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::uuid)
                RETURNING id
                """,
                record.commodity, record.variety, record.state, record.market,
                record.price_date, record.min_price, record.max_price, record.modal_price,
                record.unit, record.source, record.raw_record_id
            )
        return str(row_id)

    async def get_prices(
        self,
        commodity: str,
        market: str,
        start_date: date,
        end_date: date
    ) -> List[NormalizedPriceRecord]:
        """Fetch normalized prices for a specific commodity/market/date range."""
        async with self.db.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM normalized_prices
                WHERE commodity = $1 AND market = $2 AND price_date >= $3 AND price_date <= $4
                ORDER BY price_date DESC, created_at DESC
                """,
                commodity, market, start_date, end_date
            )
        return [NormalizedPriceRecord(**dict(row)) for row in rows]
