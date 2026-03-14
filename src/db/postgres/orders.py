"""
Order CRUD mixin for Aurora PostgreSQL.
"""

from typing import Any, Optional

from loguru import logger


class OrderOperationsMixin:
    """
    Mixin providing order lifecycle operations.

    Requires the parent class to expose a `pool` property
    returning an asyncpg connection pool.
    """

    async def create_order(self, order_data: dict[str, Any]) -> str:
        """
        Create a new order with escrow pending.

        Args:
            order_data: Dict with keys: listing_id, buyer_id, quantity_kg,
                        farmer_payout, logistics_cost, platform_margin,
                        risk_buffer, aisp_total, aisp_per_kg.
                        Optional: hauler_id.

        Returns:
            New order UUID as string.
        """
        async with self.pool.acquire() as conn:
            order_id = await conn.fetchval(
                """
                INSERT INTO orders
                    (listing_id, buyer_id, hauler_id, quantity_kg,
                     farmer_payout, logistics_cost, platform_margin,
                     risk_buffer, aisp_total, aisp_per_kg)
                VALUES ($1::uuid, $2::uuid, $3::uuid, $4, $5, $6, $7, $8, $9, $10)
                RETURNING id
                """,
                order_data["listing_id"],
                order_data["buyer_id"],
                order_data.get("hauler_id"),
                float(order_data["quantity_kg"]),
                float(order_data["farmer_payout"]),
                float(order_data["logistics_cost"]),
                float(order_data["platform_margin"]),
                float(order_data["risk_buffer"]),
                float(order_data["aisp_total"]),
                float(order_data["aisp_per_kg"]),
            )
        logger.info(
            f"Created order {order_id}: "
            f"{order_data['quantity_kg']}kg, ₹{order_data['aisp_total']}"
        )
        return str(order_id)

    async def update_order_status(
        self,
        order_id: str,
        status: str,
        escrow_status: Optional[str] = None,
    ) -> bool:
        """
        Update order lifecycle status.

        Args:
            order_id: Order UUID.
            status: New order_status value.
            escrow_status: Optional new escrow_status value.

        Returns:
            True if the row was updated, False if order not found.
        """
        if escrow_status:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    """
                    UPDATE orders
                    SET order_status = $1, escrow_status = $2
                    WHERE id = $3::uuid
                    """,
                    status, escrow_status, order_id,
                )
        else:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    "UPDATE orders SET order_status = $1 WHERE id = $2::uuid",
                    status, order_id,
                )

        updated = int(result.split()[-1]) > 0
        if updated:
            logger.info(f"Order {order_id} status → {status}")
        return updated

    async def get_order(self, order_id: str) -> Optional[dict[str, Any]]:
        """Fetch a single order with listing and buyer details."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT o.*,
                       l.commodity, l.farmer_id,
                       b.name AS buyer_name
                FROM orders o
                JOIN listings l ON l.id = o.listing_id
                JOIN buyers   b ON b.id = o.buyer_id
                WHERE o.id = $1::uuid
                """,
                order_id,
            )
        return dict(row) if row else None

    async def get_orders_by_farmer(
        self,
        farmer_id: str,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch all orders belonging to a farmer's listings."""
        params: list[Any] = [farmer_id, limit]
        status_clause = ""
        if status:
            status_clause = "AND o.order_status = $3"
            params = [farmer_id, limit, status]

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT o.*, l.commodity
                FROM orders o
                JOIN listings l ON l.id = o.listing_id
                WHERE l.farmer_id = $1::uuid
                  {status_clause}
                ORDER BY o.created_at DESC
                LIMIT $2
                """,
                *params,
            )
        return [dict(row) for row in rows]

    async def get_orders_by_buyer(
        self,
        buyer_id: str,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch all orders placed by a buyer."""
        params: list[Any] = [buyer_id, limit]
        status_clause = ""
        if status:
            status_clause = "AND o.order_status = $3"
            params = [buyer_id, limit, status]

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT o.*, l.commodity
                FROM orders o
                JOIN listings l ON l.id = o.listing_id
                WHERE o.buyer_id = $1::uuid
                  {status_clause}
                ORDER BY o.created_at DESC
                LIMIT $2
                """,
                *params,
            )
        return [dict(row) for row in rows]
