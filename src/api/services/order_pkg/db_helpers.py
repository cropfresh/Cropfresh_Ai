"""
Database helper methods for OrderService — extracted from the private methods.
"""

import uuid
from typing import Any, Optional

from loguru import logger

from src.api.services.order_pkg.models import AISPBreakdown, OrderResponse


class OrderDBMixin:
    """
    Mixin providing all DB interaction methods for OrderService.

    Expects `self.db` to be set by the main class.
    """

    async def _fetch_listing(self, listing_id: str) -> Optional[dict[str, Any]]:
        """Fetch listing row by UUID; returns None when DB unavailable."""
        if self.db and hasattr(self.db, "get_listing"):
            return await self.db.get_listing(listing_id)
        return None

    async def _fetch_order(self, order_id: str) -> Optional[dict[str, Any]]:
        """Fetch order row by UUID."""
        if self.db and hasattr(self.db, "get_order"):
            return await self.db.get_order(order_id)
        return None

    async def _fetch_farmer_orders(
        self, farmer_id: str, status: Optional[str]
    ) -> list[dict[str, Any]]:
        """Fetch all orders for a farmer's listings."""
        if self.db and hasattr(self.db, "get_orders_by_farmer"):
            return await self.db.get_orders_by_farmer(farmer_id, status=status)
        return []

    async def _fetch_buyer_orders(
        self, buyer_id: str, status: Optional[str]
    ) -> list[dict[str, Any]]:
        """Fetch all orders placed by a buyer."""
        if self.db and hasattr(self.db, "get_orders_by_buyer"):
            return await self.db.get_orders_by_buyer(buyer_id, status=status)
        return []

    async def _persist_order(self, order_data: dict[str, Any]) -> str:
        """Persist order to DB; return order UUID."""
        if self.db and hasattr(self.db, "create_order"):
            return await self.db.create_order(order_data)
        return str(uuid.uuid4())

    async def _persist_dispute(self, dispute_data: dict[str, Any]) -> str:
        """Persist dispute to DB; return dispute UUID."""
        if self.db and hasattr(self.db, "create_dispute"):
            return await self.db.create_dispute(dispute_data)
        return str(uuid.uuid4())

    async def _apply_status(
        self,
        order_id: str,
        status: str,
        escrow_status: Optional[str] = None,
    ) -> None:
        """Write status transition to DB."""
        if self.db and hasattr(self.db, "update_order_status"):
            await self.db.update_order_status(order_id, status, escrow_status)

    async def _update_listing_status(self, listing_id: str, status: str) -> None:
        """Update listing status after order creation (e.g., → 'matched')."""
        if self.db and hasattr(self.db, "update_listing"):
            await self.db.update_listing(listing_id, {"status": status})

    async def _save_diff_report(
        self,
        dispute_id: str,
        diff_report: dict[str, Any],
        liability: Optional[str] = None,
        claim_percent: Optional[float] = None,
    ) -> None:
        """Persist diff_report + liability to dispute record."""
        if not self.db:
            return
        if hasattr(self.db, "update_dispute_diff_report"):
            await self.db.update_dispute_diff_report(
                dispute_id, diff_report,
                liability=liability, claim_percent=claim_percent,
            )
        elif hasattr(self.db, "update_dispute"):
            await self.db.update_dispute(
                dispute_id,
                {
                    "diff_report": diff_report,
                    "liability": liability,
                    "claim_percent": claim_percent,
                    "status": "ai_analysed",
                },
            )

    # ── Row converters ────────────────────────────────────────

    @staticmethod
    def _row_to_aisp(row: dict[str, Any]) -> AISPBreakdown:
        """Extract AISP fields from a DB order row."""
        return AISPBreakdown(
            farmer_payout=float(row.get("farmer_payout", 0)),
            logistics_cost=float(row.get("logistics_cost", 0)),
            platform_margin=float(row.get("platform_margin", 0)),
            risk_buffer=float(row.get("risk_buffer", 0)),
            aisp_total=float(row.get("aisp_total", 0)),
            aisp_per_kg=float(row.get("aisp_per_kg", 0)),
        )

    @staticmethod
    def _row_to_order(row: dict[str, Any]) -> OrderResponse:
        """Convert a DB row dict to an OrderResponse."""
        return OrderResponse(
            id=str(row.get("id", "")),
            listing_id=str(row.get("listing_id", "")),
            buyer_id=str(row.get("buyer_id", "")),
            hauler_id=str(row["hauler_id"]) if row.get("hauler_id") else None,
            quantity_kg=float(row.get("quantity_kg", 0)),
            order_status=row.get("order_status", "confirmed"),
            escrow_status=row.get("escrow_status", "held"),
            aisp=OrderDBMixin._row_to_aisp(row),
            commodity=row.get("commodity"),
            farmer_id=str(row["farmer_id"]) if row.get("farmer_id") else None,
            buyer_name=row.get("buyer_name"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )
