"""
User, farmer, and buyer CRUD mixin for Aurora PostgreSQL.

Replaces SupabaseClient user operations.
"""

import json
from typing import Any, Optional

from loguru import logger


class UserOperationsMixin:
    """
    Mixin providing user, farmer, and buyer CRUD operations.

    Requires the parent class to expose a `pool` property
    returning an asyncpg connection pool.
    """

    async def get_user(self, user_id: str) -> Optional[dict[str, Any]]:
        """Get user by ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM users WHERE id = $1::uuid",
                user_id,
            )
        return dict(row) if row else None

    async def create_user(
        self,
        phone: str,
        name: str,
        user_type: str = "farmer",
        location: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Create a new user."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO users (phone, name, user_type, location)
                VALUES ($1, $2, $3, $4::jsonb)
                RETURNING *
                """,
                phone, name, user_type, json.dumps(location or {}),
            )

        logger.info(f"Created user: {name} ({user_type})")
        return dict(row) if row else {}

    async def create_farmer(self, farmer_data: dict[str, Any]) -> str:
        """
        Insert a new farmer record.

        Args:
            farmer_data: Dict with keys: name, phone, district, and optional
                         village, language_pref, aadhaar_hash, onboarded_by.

        Returns:
            New farmer UUID as string.
        """
        async with self.pool.acquire() as conn:
            farmer_id = await conn.fetchval(
                """
                INSERT INTO farmers
                    (name, phone, district, village, language_pref,
                     aadhaar_hash, onboarded_by)
                VALUES ($1, $2, $3, $4, $5, $6, $7::uuid)
                RETURNING id
                """,
                farmer_data["name"],
                farmer_data["phone"],
                farmer_data["district"],
                farmer_data.get("village"),
                farmer_data.get("language_pref", "kn"),
                farmer_data.get("aadhaar_hash"),
                farmer_data.get("onboarded_by"),
            )
        logger.info(f"Created farmer: {farmer_data['name']} ({farmer_data['phone']})")
        return str(farmer_id)

    async def get_farmer(self, farmer_id: str) -> Optional[dict[str, Any]]:
        """Fetch a farmer record by UUID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM farmers WHERE id = $1::uuid AND is_active = TRUE",
                farmer_id,
            )
        return dict(row) if row else None

    async def get_farmer_by_phone(self, phone: str) -> Optional[dict[str, Any]]:
        """Fetch a farmer record by phone number."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM farmers WHERE phone = $1 AND is_active = TRUE LIMIT 1",
                phone,
            )
        return dict(row) if row else None

    async def update_farmer(
        self,
        farmer_id: str,
        updates: dict[str, Any],
    ) -> bool:
        """
        Update a farmer's profile fields.

        Args:
            farmer_id: Farmer UUID string.
            updates: Dict with any subset of: name, district, village,
                     language_pref, aadhaar_hash, quality_score.

        Returns:
            True if the row was updated, False if not found.
        """
        allowed = {
            "name", "district", "village", "language_pref",
            "aadhaar_hash", "quality_score",
        }
        fields = {k: v for k, v in updates.items() if k in allowed}
        if not fields:
            return False

        set_clauses = [f"{col} = ${i}" for i, col in enumerate(fields.keys(), start=1)]
        params: list[Any] = list(fields.values())
        params.append(farmer_id)

        async with self.pool.acquire() as conn:
            result = await conn.execute(
                f"UPDATE farmers SET {', '.join(set_clauses)} WHERE id = ${len(params)}::uuid",
                *params,
            )
        updated = int(result.split()[-1]) > 0
        if updated:
            logger.info(f"Farmer {farmer_id} updated: {list(fields.keys())}")
        return updated

    async def create_buyer(self, buyer_data: dict[str, Any]) -> str:
        """
        Create a new buyer record.

        Args:
            buyer_data: Dict with keys: name, phone, type, district,
                        and optional credit_limit, subscription_tier.

        Returns:
            New buyer UUID as string.
        """
        async with self.pool.acquire() as conn:
            buyer_id = await conn.fetchval(
                """
                INSERT INTO buyers (name, phone, type, district, credit_limit, subscription_tier)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
                """,
                buyer_data["name"],
                buyer_data["phone"],
                buyer_data["type"],
                buyer_data["district"],
                float(buyer_data.get("credit_limit", 0.0)),
                buyer_data.get("subscription_tier", "free"),
            )
        logger.info(f"Created buyer: {buyer_data['name']} ({buyer_data['type']})")
        return str(buyer_id)

    async def get_buyer(self, buyer_id: str) -> Optional[dict[str, Any]]:
        """Fetch a single buyer record by UUID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM buyers WHERE id = $1::uuid AND is_active = TRUE",
                buyer_id,
            )
        return dict(row) if row else None

    async def get_buyer_by_phone(self, phone: str) -> Optional[dict[str, Any]]:
        """Fetch a buyer record by phone number."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM buyers WHERE phone = $1 AND is_active = TRUE LIMIT 1",
                phone,
            )
        return dict(row) if row else None

    async def update_buyer(
        self,
        buyer_id: str,
        updates: dict[str, Any],
    ) -> bool:
        """
        Update a buyer's profile fields.

        Args:
            buyer_id: Buyer UUID string.
            updates: Dict with any subset of: name, district, type,
                     credit_limit, subscription_tier.

        Returns:
            True if the row was updated, False if not found.
        """
        allowed = {"name", "district", "type", "credit_limit", "subscription_tier"}
        fields = {k: v for k, v in updates.items() if k in allowed}
        if not fields:
            return False

        set_clauses = [f"{col} = ${i}" for i, col in enumerate(fields.keys(), start=1)]
        params: list[Any] = list(fields.values())
        params.append(buyer_id)

        async with self.pool.acquire() as conn:
            result = await conn.execute(
                f"UPDATE buyers SET {', '.join(set_clauses)} WHERE id = ${len(params)}::uuid",
                *params,
            )
        updated = int(result.split()[-1]) > 0
        if updated:
            logger.info(f"Buyer {buyer_id} updated: {list(fields.keys())}")
        return updated
