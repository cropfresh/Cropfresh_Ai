"""
Digital twin and dispute CRUD mixin for Aurora PostgreSQL.
"""

import json
from typing import Any, Optional

from loguru import logger


class DigitalTwinOperationsMixin:
    """
    Mixin providing digital twin and dispute operations.

    Requires the parent class to expose a `pool` property
    returning an asyncpg connection pool.
    """

    async def create_digital_twin(self, twin_data: dict[str, Any]) -> str:
        """
        Create a digital twin record for a listing.

        Args:
            twin_data: Dict with keys: listing_id, and optional farmer_photos,
                       agent_photos, ai_annotations, agent_id, grade,
                       confidence, defect_types, shelf_life_days.

        Returns:
            New digital_twin UUID as string.
        """
        async with self.pool.acquire() as conn:
            twin_id = await conn.fetchval(
                """
                INSERT INTO digital_twins
                    (listing_id, farmer_photos, agent_photos,
                     ai_annotations, agent_id, grade,
                     confidence, defect_types, shelf_life_days)
                VALUES ($1::uuid, $2, $3, $4::jsonb, $5::uuid, $6, $7, $8, $9)
                RETURNING id
                """,
                twin_data["listing_id"],
                twin_data.get("farmer_photos", []),
                twin_data.get("agent_photos", []),
                json.dumps(twin_data.get("ai_annotations", {})),
                twin_data.get("agent_id"),
                twin_data.get("grade"),
                twin_data.get("confidence"),
                twin_data.get("defect_types", []),
                twin_data.get("shelf_life_days"),
            )
        logger.info(f"Created digital twin {twin_id} for listing {twin_data['listing_id']}")
        return str(twin_id)

    async def get_digital_twin(self, twin_id: str) -> Optional[dict[str, Any]]:
        """Fetch a digital twin record by UUID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, listing_id, farmer_photos, agent_photos,
                       ai_annotations, agent_id, grade, confidence,
                       defect_types, shelf_life_days, created_at
                FROM digital_twins
                WHERE id = $1::uuid
                """,
                twin_id,
            )
        if row is None:
            return None
        return dict(row)

    async def update_dispute_diff_report(
        self,
        dispute_id: str,
        diff_report: dict[str, Any],
        liability: Optional[str] = None,
        claim_percent: Optional[float] = None,
    ) -> bool:
        """
        Update a dispute record with the AI diff report results.

        Args:
            dispute_id:   Dispute UUID string.
            diff_report:  DiffReport.to_dict() payload.
            liability:    Liable party ('farmer' | 'hauler' | 'buyer' | 'shared').
            claim_percent: Recommended claim percentage [0, 100].

        Returns:
            True if a row was updated, False if dispute not found.
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE disputes
                SET diff_report   = $2::jsonb,
                    liability     = COALESCE($3, liability),
                    claim_percent = COALESCE($4, claim_percent),
                    status        = 'ai_analysed',
                    updated_at    = NOW()
                WHERE id = $1::uuid
                """,
                dispute_id,
                json.dumps(diff_report),
                liability,
                claim_percent,
            )
        updated = result.split()[-1] != "0"
        if updated:
            logger.info(f"Dispute {dispute_id} diff report saved (liability={liability})")
        return updated

    async def create_dispute(self, dispute_data: dict[str, Any]) -> str:
        """
        Open a new dispute for an order.

        Args:
            dispute_data: Dict with keys: order_id, raised_by, reason,
                          and optional departure_twin_id, arrival_photos.

        Returns:
            New dispute UUID as string.
        """
        async with self.pool.acquire() as conn:
            dispute_id = await conn.fetchval(
                """
                INSERT INTO disputes
                    (order_id, raised_by, reason,
                     departure_twin_id, arrival_photos)
                VALUES ($1::uuid, $2, $3, $4::uuid, $5)
                RETURNING id
                """,
                dispute_data["order_id"],
                dispute_data["raised_by"],
                dispute_data["reason"],
                dispute_data.get("departure_twin_id"),
                dispute_data.get("arrival_photos", []),
            )
        logger.info(f"Opened dispute {dispute_id} for order {dispute_data['order_id']}")
        return str(dispute_id)

    async def update_dispute(
        self,
        dispute_id: str,
        updates: dict[str, Any],
    ) -> bool:
        """
        Update a dispute record (status, diff_report, liability, claim_percent).

        Returns:
            True if the row was updated, False if not found.
        """
        allowed = {"status", "diff_report", "liability", "claim_percent", "resolution_notes"}
        fields = {k: v for k, v in updates.items() if k in allowed}
        if not fields:
            return False

        set_clauses = []
        params: list[Any] = []
        for i, (col, val) in enumerate(fields.items(), start=1):
            if col == "diff_report":
                set_clauses.append(f"{col} = ${i}::jsonb")
                params.append(json.dumps(val) if isinstance(val, dict) else val)
            else:
                set_clauses.append(f"{col} = ${i}")
                params.append(val)

        params.append(dispute_id)
        set_sql = ", ".join(set_clauses)

        async with self.pool.acquire() as conn:
            result = await conn.execute(
                f"UPDATE disputes SET {set_sql} WHERE id = ${len(params)}::uuid",
                *params,
            )
        updated = int(result.split()[-1]) > 0
        if updated:
            logger.info(f"Dispute {dispute_id} updated: {list(fields.keys())}")
        return updated

    async def get_dispute_status(self, dispute_id: str) -> Optional[dict[str, Any]]:
        """Fetch a dispute record with order and listing info."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT d.*, o.order_status, o.aisp_total, l.commodity
                FROM disputes d
                JOIN orders o ON o.id = d.order_id
                JOIN listings l ON l.id = o.listing_id
                WHERE d.id = $1::uuid
                """,
                dispute_id,
            )
        return dict(row) if row else None
