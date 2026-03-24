"""Repository for the multi-source rate hub."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.db.models.price_records import NormalizedPriceRecord, RawPriceRecord
from src.db.repositories.price_repository import PriceRepository
from src.rates.enums import RateKind
from src.rates.models import NormalizedRateRecord, RateQuery


class RateRepository:
    """Persistence helpers for rate-hub raw and normalized data."""

    def __init__(self, db_client: Any):
        self.db = db_client
        self.price_repository = PriceRepository(db_client) if db_client is not None else None

    async def initialize_schema(self) -> None:
        """Create the normalized_rates table when a DB client is available."""
        if self.db is None:
            return
        schema_path = Path(__file__).resolve().parents[1] / "db" / "schema_rates.sql"
        async with self.db.pool.acquire() as conn:
            await conn.execute(schema_path.read_text(encoding="utf-8"))

    async def save_raw_record(self, record: RawPriceRecord) -> str | None:
        """Persist raw connector data into source_data."""
        if self.db is None:
            return None
        async with self.db.pool.acquire() as conn:
            return str(
                await conn.fetchval(
                    """
                    INSERT INTO source_data (source, raw_data, url, scraped_at)
                    VALUES ($1, $2::jsonb, $3, $4)
                    RETURNING id
                    """,
                    record.source,
                    json.dumps(record.raw_data),
                    record.url,
                    record.scraped_at,
                )
            )

    async def save_normalized_rate(self, record: NormalizedRateRecord) -> str | None:
        """Persist one normalized rate and dual-write mandi data to normalized_prices."""
        if self.db is None:
            return None
        async with self.db.pool.acquire() as conn:
            row_id = await conn.fetchval(
                """
                INSERT INTO normalized_rates (
                    rate_kind, commodity, variety, state, district, market, location_label,
                    price_date, unit, currency, price_value, min_price, max_price,
                    modal_price, source, authority_tier, source_url, freshness,
                    fetched_at, raw_record_id
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7,
                    $8, $9, $10, $11, $12, $13,
                    $14, $15, $16, $17, $18,
                    $19, $20::uuid
                )
                RETURNING id
                """,
                record.rate_kind.value,
                record.commodity,
                record.variety,
                record.state,
                record.district,
                record.market,
                record.location_label,
                record.price_date,
                record.unit,
                record.currency,
                record.price_value,
                record.min_price,
                record.max_price,
                record.modal_price,
                record.source,
                record.authority_tier.value,
                record.source_url,
                record.freshness,
                record.fetched_at,
                record.raw_record_id,
            )
        if record.rate_kind is RateKind.MANDI_WHOLESALE and self.price_repository is not None:
            await self.price_repository.save_normalized_record(
                NormalizedPriceRecord(
                    commodity=record.commodity or "",
                    variety=record.variety,
                    state=record.state,
                    market=record.market or record.location_label,
                    price_date=record.price_date,
                    min_price=record.min_price,
                    max_price=record.max_price,
                    modal_price=record.modal_price or record.price_value,
                    unit=record.unit,
                    source=record.source,
                    raw_record_id=record.raw_record_id,
                )
            )
        return str(row_id)

    async def get_rates(self, query: RateQuery) -> list[NormalizedRateRecord]:
        """Load normalized rates for a query target from persistent storage."""
        if self.db is None:
            return []
        clauses = ["rate_kind = ANY($1::text[])", "price_date = $2"]
        values: list[Any] = [[kind.value for kind in query.rate_kinds], query.target_date]
        if query.commodity:
            clauses.append(f"commodity ILIKE ${len(values) + 1}")
            values.append(f"%{query.commodity}%")
        if query.market:
            clauses.append(f"location_label ILIKE ${len(values) + 1}")
            values.append(f"%{query.market}%")
        elif query.district:
            clauses.append(f"district ILIKE ${len(values) + 1}")
            values.append(f"%{query.district}%")

        sql = f"SELECT * FROM normalized_rates WHERE {' AND '.join(clauses)} ORDER BY fetched_at DESC"
        async with self.db.pool.acquire() as conn:
            rows = await conn.fetch(sql, *values)
        return [NormalizedRateRecord.model_validate(dict(row)) for row in rows]
