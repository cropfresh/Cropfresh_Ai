"""ADCL-specific Aurora PostgreSQL operations."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from loguru import logger


class ADCLOperationsMixin:
    """Mixin providing district-scoped ADCL read/write helpers."""

    async def get_recent_orders(
        self,
        district: str,
        days: int = 90,
    ) -> list[dict[str, Any]]:
        """Return recent confirmed marketplace orders for a district."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    l.commodity,
                    o.quantity_kg,
                    o.buyer_id,
                    o.order_status,
                    o.created_at,
                    b.district AS buyer_district,
                    f.district AS farmer_district
                FROM orders o
                JOIN listings l ON l.id = o.listing_id
                JOIN buyers b ON b.id = o.buyer_id
                JOIN farmers f ON f.id = l.farmer_id
                WHERE o.created_at >= now() - ($2 * INTERVAL '1 day')
                  AND (
                      b.district ILIKE $1
                      OR f.district ILIKE $1
                  )
                  AND o.order_status IN (
                      'confirmed',
                      'pickup_scheduled',
                      'in_transit',
                      'delivered',
                      'disputed',
                      'settled'
                  )
                ORDER BY o.created_at DESC
                """,
                f"%{district}%",
                days,
            )
        return [dict(row) for row in rows]

    async def insert_adcl_report(self, report: dict[str, Any]) -> None:
        """Insert or update one persisted ADCL report."""
        week_start = report["week_start"]
        generated_at = report.get("generated_at")
        if isinstance(week_start, str):
            week_start = date.fromisoformat(week_start)
        if isinstance(generated_at, str):
            generated_at = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        crops = json.dumps(report.get("crops", []))
        freshness = json.dumps(report.get("freshness", {}))
        source_health = json.dumps(report.get("source_health", {}))
        metadata = json.dumps(report.get("metadata", {}))
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO adcl_reports (
                    week_start,
                    district,
                    crops,
                    generated_by,
                    generated_at,
                    summary_en,
                    summary_hi,
                    summary_kn,
                    freshness,
                    source_health,
                    metadata
                )
                VALUES (
                    $1,
                    $2,
                    $3::jsonb,
                    $4,
                    $5,
                    $6,
                    $7,
                    $8,
                    $9::jsonb,
                    $10::jsonb,
                    $11::jsonb
                )
                ON CONFLICT (week_start, district)
                DO UPDATE SET
                    crops = EXCLUDED.crops,
                    generated_by = EXCLUDED.generated_by,
                    generated_at = EXCLUDED.generated_at,
                    summary_en = EXCLUDED.summary_en,
                    summary_hi = EXCLUDED.summary_hi,
                    summary_kn = EXCLUDED.summary_kn,
                    freshness = EXCLUDED.freshness,
                    source_health = EXCLUDED.source_health,
                    metadata = EXCLUDED.metadata
                """,
                week_start,
                report.get("district", "Bangalore"),
                crops,
                report.get("generated_by", "adcl_service"),
                generated_at,
                report.get("summary_en", ""),
                report.get("summary_hi", ""),
                report.get("summary_kn", ""),
                freshness,
                source_health,
                metadata,
            )
        logger.info(
            "Persisted ADCL report for {} ({})",
            report.get("district", "Bangalore"),
            report["week_start"],
        )

    async def get_latest_adcl_report(
        self,
        district: str,
        week_start: date | None = None,
    ) -> dict[str, Any] | None:
        """Fetch the latest ADCL report for a district."""
        query = """
            SELECT
                week_start,
                district,
                crops,
                generated_by,
                generated_at,
                summary_en,
                summary_hi,
                summary_kn,
                freshness,
                source_health,
                metadata
            FROM adcl_reports
            WHERE district ILIKE $1
        """
        params: list[Any] = [district]
        if week_start is not None:
            query += " AND week_start = $2"
            params.append(week_start)
        query += " ORDER BY week_start DESC, generated_at DESC LIMIT 1"

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)
        return dict(row) if row else None
