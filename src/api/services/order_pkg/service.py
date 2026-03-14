"""
Order Management Service — main lifecycle operations.

Responsibilities:
- Create orders from matched listings with AISP price breakdown
- Enforce the order state machine (valid transitions only)
- Manage escrow flow: held → released / refunded
- Raise disputes with Digital Twin comparison trigger
- Settle orders with escrow release and stats update
"""

from datetime import UTC, datetime
from typing import Any, Optional

from loguru import logger

from src.api.services.order_pkg.db_helpers import OrderDBMixin
from src.api.services.order_pkg.models import (
    AISP_FARMER_RATIO,
    AISP_LOGISTICS_RATIO,
    AISP_PLATFORM_RATIO,
    AISP_RISK_RATIO,
    ESCROW_ON_TRANSITION,
    VALID_TRANSITIONS,
    AISPBreakdown,
    CreateOrderRequest,
    DisputeResponse,
    OrderResponse,
    RaiseDisputeRequest,
)


class OrderService(OrderDBMixin):
    """Full lifecycle management for CropFresh marketplace orders."""

    def __init__(
        self,
        db: Optional[Any] = None,
        pricing_agent: Optional[Any] = None,
        quality_agent: Optional[Any] = None,
        twin_engine: Optional[Any] = None,
        notification_stub: Optional[Any] = None,
    ) -> None:
        self.db = db
        self.pricing_agent = pricing_agent
        self.quality_agent = quality_agent
        self.twin_engine = twin_engine
        self.notification_stub = notification_stub

    # ── Create ────────────────────────────────────────────────

    async def create_order(self, request: CreateOrderRequest) -> OrderResponse:
        """Create a new order from a matched listing."""
        listing = await self._fetch_listing(request.listing_id)
        if listing is None:
            raise ValueError(f"Listing {request.listing_id} not found")

        if listing.get("status") not in ("active", "matched"):
            raise ValueError(
                f"Listing {request.listing_id} is not available "
                f"(status={listing.get('status')})"
            )

        price_per_kg = request.override_price_per_kg or float(
            listing.get("asking_price_per_kg", 0)
        )
        aisp = await self._calculate_aisp(
            quantity_kg=request.quantity_kg,
            price_per_kg=price_per_kg,
            commodity=listing.get("commodity", ""),
        )

        hauler_id = request.hauler_id or await self._assign_hauler(listing)

        order_data: dict[str, Any] = {
            "listing_id": request.listing_id,
            "buyer_id": request.buyer_id,
            "hauler_id": hauler_id,
            "quantity_kg": request.quantity_kg,
            "farmer_payout": aisp.farmer_payout,
            "logistics_cost": aisp.logistics_cost,
            "platform_margin": aisp.platform_margin,
            "risk_buffer": aisp.risk_buffer,
            "aisp_total": aisp.aisp_total,
            "aisp_per_kg": aisp.aisp_per_kg,
        }
        order_id = await self._persist_order(order_data)
        await self._update_listing_status(request.listing_id, "matched")
        await self._notify_order_created(order_id, listing, request.buyer_id)

        logger.info(
            f"Created order {order_id}: {request.quantity_kg}kg "
            f"{listing.get('commodity')} @ ₹{price_per_kg}/kg "
            f"(aisp_total=₹{aisp.aisp_total:.2f}, escrow=held)"
        )

        return OrderResponse(
            id=order_id, listing_id=request.listing_id,
            buyer_id=request.buyer_id, hauler_id=hauler_id,
            quantity_kg=request.quantity_kg,
            order_status="confirmed", escrow_status="held",
            aisp=aisp, commodity=listing.get("commodity"),
            farmer_id=str(listing.get("farmer_id", "")),
            created_at=datetime.now(UTC).replace(tzinfo=None),
        )

    # ── State machine ─────────────────────────────────────────

    async def update_status(
        self, order_id: str, new_status: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> OrderResponse:
        """Advance an order through the state machine."""
        order = await self._fetch_order(order_id)
        if order is None:
            raise ValueError(f"Order {order_id} not found")

        current = order.get("order_status", "")
        self._assert_valid_transition(current, new_status)

        escrow_status = ESCROW_ON_TRANSITION.get(new_status)
        await self._apply_status(order_id, new_status, escrow_status)
        await self._notify_status_change(order_id, new_status, metadata or {})

        logger.info(
            f"Order {order_id}: {current} → {new_status}"
            + (f" (escrow → {escrow_status})" if escrow_status else "")
        )

        updated = {**order, "order_status": new_status}
        if escrow_status:
            updated["escrow_status"] = escrow_status
        return self._row_to_order(updated)

    # ── Dispute ───────────────────────────────────────────────

    async def raise_dispute(
        self, order_id: str, dispute_data: RaiseDisputeRequest,
    ) -> DisputeResponse:
        """Open a dispute for an in-transit or delivered order."""
        order = await self._fetch_order(order_id)
        if order is None:
            raise ValueError(f"Order {order_id} not found")

        self._assert_valid_transition(order.get("order_status", ""), "disputed")
        await self._apply_status(order_id, "disputed")

        record: dict[str, Any] = {
            "order_id": order_id,
            "raised_by": dispute_data.raised_by,
            "reason": dispute_data.reason,
            "arrival_photos": dispute_data.arrival_photos or [],
            "departure_twin_id": dispute_data.departure_twin_id,
        }
        dispute_id = await self._persist_dispute(record)

        diff_report: Optional[dict[str, Any]] = None
        if dispute_data.departure_twin_id and dispute_data.arrival_photos:
            diff_report = await self._trigger_twin_diff(
                departure_twin_id=dispute_data.departure_twin_id,
                arrival_photos=dispute_data.arrival_photos,
                dispute_id=dispute_id,
            )

        logger.info(
            f"Dispute {dispute_id} raised for order {order_id} "
            f"by {dispute_data.raised_by}: {dispute_data.reason}"
        )

        return DisputeResponse(
            id=dispute_id, order_id=order_id,
            raised_by=dispute_data.raised_by, reason=dispute_data.reason,
            status="open",
            arrival_photos=dispute_data.arrival_photos,
            departure_twin_id=dispute_data.departure_twin_id,
            diff_report=diff_report,
            created_at=datetime.now(UTC).replace(tzinfo=None),
        )

    # ── Settlement ────────────────────────────────────────────

    async def settle_order(self, order_id: str) -> OrderResponse:
        """Release escrow to farmer and settle the order."""
        order = await self._fetch_order(order_id)
        if order is None:
            raise ValueError(f"Order {order_id} not found")

        self._assert_valid_transition(order.get("order_status", ""), "settled")
        await self._apply_status(order_id, "settled", "released")
        await self._update_participant_stats(order)
        await self._notify_status_change(order_id, "settled", {})

        logger.info(
            f"Order {order_id} settled — "
            f"escrow released ₹{order.get('farmer_payout', 0):.2f} to farmer"
        )
        settled = {**order, "order_status": "settled", "escrow_status": "released"}
        return self._row_to_order(settled)

    # ── Queries ───────────────────────────────────────────────

    async def get_order(self, order_id: str) -> Optional[OrderResponse]:
        row = await self._fetch_order(order_id)
        return self._row_to_order(row) if row else None

    async def get_orders_by_farmer(
        self, farmer_id: str, status: Optional[str] = None,
    ) -> list[OrderResponse]:
        rows = await self._fetch_farmer_orders(farmer_id, status)
        return [self._row_to_order(r) for r in rows]

    async def get_orders_by_buyer(
        self, buyer_id: str, status: Optional[str] = None,
    ) -> list[OrderResponse]:
        rows = await self._fetch_buyer_orders(buyer_id, status)
        return [self._row_to_order(r) for r in rows]

    async def get_aisp_breakdown(self, order_id: str) -> Optional[AISPBreakdown]:
        row = await self._fetch_order(order_id)
        return self._row_to_aisp(row) if row else None

    # ── Private helpers ───────────────────────────────────────

    @staticmethod
    def _assert_valid_transition(current: str, target: str) -> None:
        allowed = VALID_TRANSITIONS.get(current, [])
        if target not in allowed:
            raise ValueError(
                f"Invalid transition: '{current}' → '{target}'. "
                f"Allowed from '{current}': {allowed or ['(terminal)']}"
            )

    async def _calculate_aisp(
        self, quantity_kg: float, price_per_kg: float, commodity: str,
    ) -> AISPBreakdown:
        if self.pricing_agent and hasattr(self.pricing_agent, "calculate_aisp"):
            try:
                result = await self.pricing_agent.calculate_aisp(
                    quantity_kg=quantity_kg, price_per_kg=price_per_kg,
                    commodity=commodity,
                )
                return AISPBreakdown(**result)
            except Exception as exc:
                logger.warning(f"PricingAgent AISP calculation failed: {exc}")

        total = round(quantity_kg * price_per_kg, 2)
        return AISPBreakdown(
            farmer_payout=round(total * AISP_FARMER_RATIO, 2),
            logistics_cost=round(total * AISP_LOGISTICS_RATIO, 2),
            platform_margin=round(total * AISP_PLATFORM_RATIO, 2),
            risk_buffer=round(total * AISP_RISK_RATIO, 2),
            aisp_total=total,
            aisp_per_kg=round(price_per_kg, 2),
        )

    async def _assign_hauler(self, listing: dict[str, Any]) -> Optional[str]:
        return None  # TODO: Replace with LogisticsRouterAgent

    async def _trigger_twin_diff(
        self, departure_twin_id: str, arrival_photos: list[str], dispute_id: str,
    ) -> Optional[dict[str, Any]]:
        diff_result = None
        if self.twin_engine and hasattr(self.twin_engine, "compare_arrival"):
            try:
                diff_result = await self.twin_engine.compare_arrival(
                    twin_id=departure_twin_id, arrival_photos=arrival_photos,
                )
            except Exception as exc:
                logger.warning(f"DigitalTwinEngine diff failed: {exc}")

        if diff_result is None and self.quality_agent and hasattr(self.quality_agent, "compare_twin"):
            try:
                diff_result = await self.quality_agent.compare_twin(
                    twin_id=departure_twin_id, arrival_photos=arrival_photos,
                )
            except Exception as exc:
                logger.warning(f"QualityAgent twin diff failed: {exc}")

        if diff_result is None:
            pending: dict[str, Any] = {
                "status": "pending",
                "message": "AI diff engine not available — manual review required",
                "dispute_id": dispute_id,
            }
            await self._save_diff_report(dispute_id, pending)
            return pending

        report_dict: dict[str, Any] = (
            diff_result.to_dict() if hasattr(diff_result, "to_dict") else dict(diff_result)
        )
        await self._save_diff_report(
            dispute_id, report_dict,
            liability=report_dict.get("liability"),
            claim_percent=report_dict.get("claim_percent"),
        )
        return report_dict

    async def _update_participant_stats(self, order: dict[str, Any]) -> None:
        logger.debug(f"Stats update stub: farmer {order.get('farmer_id')}")

    async def _notify_order_created(
        self, order_id: str, listing: dict[str, Any], buyer_id: str,
    ) -> None:
        logger.info(f"[NOTIFY-STUB] Order {order_id} created")

    async def _notify_status_change(
        self, order_id: str, new_status: str, metadata: dict[str, Any],
    ) -> None:
        logger.info(f"[NOTIFY-STUB] Order {order_id} → {new_status}")


def get_order_service(
    db=None, pricing_agent=None, quality_agent=None,
    twin_engine=None, notification_stub=None,
) -> OrderService:
    """Factory for creating an OrderService with injected dependencies."""
    return OrderService(
        db=db, pricing_agent=pricing_agent,
        quality_agent=quality_agent, twin_engine=twin_engine,
        notification_stub=notification_stub,
    )
