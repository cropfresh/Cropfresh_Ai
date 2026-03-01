"""
Order Management Service
========================
Full order lifecycle management for the CropFresh marketplace.

Responsibilities:
- Create orders from matched listings with AISP price breakdown
- Enforce the order state machine (valid transitions only)
- Manage escrow flow: held → released / refunded
- Raise disputes with Digital Twin comparison trigger
- Settle orders with escrow release and stats update
- Notification stubs for every status change
- Query order history by farmer or buyer
"""

# * ORDER SERVICE MODULE
# NOTE: All DB writes go through AuroraPostgresClient CRUD methods (Task 6)
# NOTE: PricingAgent + QualityAgent deps injected; service degrades gracefully if absent

import uuid
from datetime import UTC, datetime
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
# * Order State Machine
# ═══════════════════════════════════════════════════════════════

# * Legal forward-transitions for every order_status value.
# ! ALERT: Any transition not listed here will be rejected with a 422.
VALID_TRANSITIONS: dict[str, list[str]] = {
    "confirmed": ["pickup_scheduled", "cancelled"],
    "pickup_scheduled": ["in_transit", "cancelled"],
    "in_transit": ["delivered", "disputed"],
    "delivered": ["settled", "disputed"],
    "disputed": ["ai_analysed"],
    "ai_analysed": ["resolved", "escalated"],
    "resolved": ["settled"],
    "escalated": ["settled", "refunded"],
    "settled": [],       # Terminal — no further transitions
    "refunded": [],      # Terminal — escrow fully returned
    "cancelled": [],     # Terminal — order abandoned
}

# * Escrow status that should be applied for each terminal/key transition
ESCROW_ON_TRANSITION: dict[str, str] = {
    "confirmed": "held",         # Escrow locked at order creation
    "settled": "released",       # Farmer paid on delivery confirmation
    "refunded": "refunded",      # Buyer refunded on dispute resolution
    "cancelled": "refunded",     # Cancelled before pickup — refund buyer
}

# * AISP cost breakdown ratios (of total transaction value)
# NOTE: AISP = AI-Set Price breakdown covering all stakeholders
AISP_FARMER_RATIO: float = 0.80       # 80% to farmer
AISP_LOGISTICS_RATIO: float = 0.10    # 10% logistics cost
AISP_PLATFORM_RATIO: float = 0.06     # 6% platform margin
AISP_RISK_RATIO: float = 0.04         # 4% risk buffer


# ═══════════════════════════════════════════════════════════════
# * Pydantic Models
# ═══════════════════════════════════════════════════════════════

class AISPBreakdown(BaseModel):
    """AI-Set Price breakdown showing how the transaction value is distributed."""
    farmer_payout: float
    logistics_cost: float
    platform_margin: float
    risk_buffer: float
    aisp_total: float
    aisp_per_kg: float


class CreateOrderRequest(BaseModel):
    """Request body for creating a new order from a matched listing."""
    listing_id: str
    buyer_id: str
    quantity_kg: float = Field(gt=0, description="Must be positive")
    hauler_id: Optional[str] = None          # Optional — can be auto-assigned
    override_price_per_kg: Optional[float] = Field(
        default=None, ge=0,
        description="Override AISP price (e.g., negotiated price)"
    )


class UpdateStatusRequest(BaseModel):
    """Request body for advancing an order through the state machine."""
    status: str
    metadata: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description="Context for the transition (e.g., GPS coordinates, notes)"
    )


class RaiseDisputeRequest(BaseModel):
    """Request body for raising a dispute with supporting evidence."""
    raised_by: str = Field(description="'buyer' or 'farmer'")
    reason: str
    arrival_photos: Optional[list[str]] = None    # S3 URLs of arrival photos
    departure_twin_id: Optional[str] = None       # Digital twin at departure


class OrderResponse(BaseModel):
    """Complete order representation returned to callers."""
    id: str
    listing_id: str
    buyer_id: str
    hauler_id: Optional[str] = None
    quantity_kg: float
    order_status: str
    escrow_status: str
    aisp: AISPBreakdown
    commodity: Optional[str] = None
    farmer_id: Optional[str] = None
    buyer_name: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class DisputeResponse(BaseModel):
    """Dispute record returned to callers."""
    id: str
    order_id: str
    raised_by: str
    reason: str
    status: str
    arrival_photos: Optional[list[str]] = None
    departure_twin_id: Optional[str] = None
    diff_report: Optional[dict[str, Any]] = None
    liability: Optional[str] = None
    claim_percent: Optional[float] = None
    created_at: Optional[datetime] = None


# ═══════════════════════════════════════════════════════════════
# * OrderService
# ═══════════════════════════════════════════════════════════════

class OrderService:
    """
    Full lifecycle management for CropFresh marketplace orders.

    Dependencies (all optional — service degrades gracefully):
        db:               AuroraPostgresClient — for persistence
        pricing_agent:    PricingAgent — for AISP calculation
        quality_agent:    QualityAssessmentAgent — for dispute diff analysis
        twin_engine:      DigitalTwinEngine — direct twin access (preferred over quality_agent)
        notification_stub: Any — for SMS/WhatsApp notifications

    NOTE: twin_engine takes priority over quality_agent for diff analysis.
    Both paths produce a DiffReport; twin_engine bypasses the QA agent wrapper.

    Usage:
        service = OrderService(db=client, pricing_agent=agent, twin_engine=engine)
        order = await service.create_order(CreateOrderRequest(...))
    """

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
        # NOTE: twin_engine preferred; quality_agent.compare_twin() used as fallback
        self.twin_engine = twin_engine
        self.notification_stub = notification_stub

    # ─────────────────────────────────────────────────────────
    # * Create
    # ─────────────────────────────────────────────────────────

    async def create_order(self, request: CreateOrderRequest) -> OrderResponse:
        """
        Create a new order from a matched listing.

        Steps:
        1. Validate listing is active
        2. Calculate AISP via PricingAgent (or ratio-based fallback)
        3. Assign hauler via LogisticsRouter stub (or use provided hauler_id)
        4. Create order record with escrow = 'held'
        5. Update listing status to 'matched'
        6. Notify farmer + buyer

        Args:
            request: Validated CreateOrderRequest.

        Returns:
            OrderResponse with full AISP breakdown.
        """
        listing = await self._fetch_listing(request.listing_id)
        if listing is None:
            raise ValueError(f"Listing {request.listing_id} not found")

        if listing.get("status") not in ("active", "matched"):
            raise ValueError(
                f"Listing {request.listing_id} is not available "
                f"(status={listing.get('status')})"
            )

        # * Step 2: calculate AISP breakdown
        price_per_kg = request.override_price_per_kg or float(
            listing.get("asking_price_per_kg", 0)
        )
        aisp = await self._calculate_aisp(
            quantity_kg=request.quantity_kg,
            price_per_kg=price_per_kg,
            commodity=listing.get("commodity", ""),
        )

        # * Step 3: hauler assignment stub
        hauler_id = request.hauler_id or await self._assign_hauler(listing)

        # * Step 4: persist order with escrow = held
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

        # * Step 5: mark listing as matched
        await self._update_listing_status(request.listing_id, "matched")

        # * Step 6: fire-and-forget notifications
        await self._notify_order_created(order_id, listing, request.buyer_id)

        logger.info(
            f"Created order {order_id}: {request.quantity_kg}kg "
            f"{listing.get('commodity')} @ ₹{price_per_kg}/kg "
            f"(aisp_total=₹{aisp.aisp_total:.2f}, escrow=held)"
        )

        return OrderResponse(
            id=order_id,
            listing_id=request.listing_id,
            buyer_id=request.buyer_id,
            hauler_id=hauler_id,
            quantity_kg=request.quantity_kg,
            order_status="confirmed",
            escrow_status="held",
            aisp=aisp,
            commodity=listing.get("commodity"),
            farmer_id=str(listing.get("farmer_id", "")),
            created_at=datetime.now(UTC).replace(tzinfo=None),
        )

    # ─────────────────────────────────────────────────────────
    # * State machine
    # ─────────────────────────────────────────────────────────

    async def update_status(
        self,
        order_id: str,
        new_status: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> OrderResponse:
        """
        Advance an order through the state machine.

        Validates the transition is legal before applying it.
        Automatically updates escrow_status for terminal transitions.

        Args:
            order_id: Order UUID.
            new_status: Target status string.
            metadata: Optional context dict (GPS, notes, etc.).

        Returns:
            Updated OrderResponse.

        Raises:
            ValueError: If order not found or transition is illegal.
        """
        order = await self._fetch_order(order_id)
        if order is None:
            raise ValueError(f"Order {order_id} not found")

        current = order.get("order_status", "")
        self._assert_valid_transition(current, new_status)

        escrow_status = ESCROW_ON_TRANSITION.get(new_status)
        await self._apply_status(order_id, new_status, escrow_status)

        # * Notify on status change
        await self._notify_status_change(order_id, new_status, metadata or {})

        logger.info(
            f"Order {order_id}: {current} → {new_status}"
            + (f" (escrow → {escrow_status})" if escrow_status else "")
        )

        updated = {**order, "order_status": new_status}
        if escrow_status:
            updated["escrow_status"] = escrow_status
        return self._row_to_order(updated)

    # ─────────────────────────────────────────────────────────
    # * Dispute
    # ─────────────────────────────────────────────────────────

    async def raise_dispute(
        self,
        order_id: str,
        dispute_data: RaiseDisputeRequest,
    ) -> DisputeResponse:
        """
        Open a dispute for an in-transit or delivered order.

        Triggers:
        - Advances order_status to 'disputed'
        - Creates dispute record
        - Triggers Digital Twin AI diff if departure_twin_id provided

        Args:
            order_id: Order UUID.
            dispute_data: Validated RaiseDisputeRequest.

        Returns:
            DisputeResponse with initial diff_report if available.

        Raises:
            ValueError: If order not found or transition to 'disputed' is illegal.
        """
        order = await self._fetch_order(order_id)
        if order is None:
            raise ValueError(f"Order {order_id} not found")

        current = order.get("order_status", "")
        self._assert_valid_transition(current, "disputed")

        # * Transition order to disputed state
        await self._apply_status(order_id, "disputed")

        # * Create dispute record
        record: dict[str, Any] = {
            "order_id": order_id,
            "raised_by": dispute_data.raised_by,
            "reason": dispute_data.reason,
            "arrival_photos": dispute_data.arrival_photos or [],
            "departure_twin_id": dispute_data.departure_twin_id,
        }
        dispute_id = await self._persist_dispute(record)

        # * Trigger Digital Twin AI diff (fire-and-forget)
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
            id=dispute_id,
            order_id=order_id,
            raised_by=dispute_data.raised_by,
            reason=dispute_data.reason,
            status="open",
            arrival_photos=dispute_data.arrival_photos,
            departure_twin_id=dispute_data.departure_twin_id,
            diff_report=diff_report,
            created_at=datetime.now(UTC).replace(tzinfo=None),
        )

    # ─────────────────────────────────────────────────────────
    # * Settlement
    # ─────────────────────────────────────────────────────────

    async def settle_order(self, order_id: str) -> OrderResponse:
        """
        Release escrow to farmer and settle the order.

        Steps:
        1. Validate transition to 'settled' is legal
        2. Set escrow_status = 'released'
        3. Update farmer + buyer stats (stub)
        4. Calculate actual platform margin

        Args:
            order_id: Order UUID.

        Returns:
            Settled OrderResponse with escrow_status = 'released'.

        Raises:
            ValueError: If order not found or settlement transition is illegal.
        """
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

    # ─────────────────────────────────────────────────────────
    # * Queries
    # ─────────────────────────────────────────────────────────

    async def get_order(self, order_id: str) -> Optional[OrderResponse]:
        """
        Fetch a single order by UUID.

        Returns:
            OrderResponse, or None if not found.
        """
        row = await self._fetch_order(order_id)
        return self._row_to_order(row) if row else None

    async def get_orders_by_farmer(
        self,
        farmer_id: str,
        status: Optional[str] = None,
    ) -> list[OrderResponse]:
        """
        Return all orders for a farmer's listings.

        Args:
            farmer_id: Farmer UUID.
            status: Optional order_status filter.

        Returns:
            List of OrderResponse objects ordered newest-first.
        """
        rows = await self._fetch_farmer_orders(farmer_id, status)
        return [self._row_to_order(r) for r in rows]

    async def get_orders_by_buyer(
        self,
        buyer_id: str,
        status: Optional[str] = None,
    ) -> list[OrderResponse]:
        """
        Return all orders placed by a buyer.

        Args:
            buyer_id: Buyer UUID.
            status: Optional order_status filter.

        Returns:
            List of OrderResponse objects ordered newest-first.
        """
        rows = await self._fetch_buyer_orders(buyer_id, status)
        return [self._row_to_order(r) for r in rows]

    async def get_aisp_breakdown(self, order_id: str) -> Optional[AISPBreakdown]:
        """
        Return the AISP price breakdown for an existing order.

        Args:
            order_id: Order UUID.

        Returns:
            AISPBreakdown, or None if order not found.
        """
        row = await self._fetch_order(order_id)
        if row is None:
            return None
        return self._row_to_aisp(row)

    # ─────────────────────────────────────────────────────────
    # * Private — state machine helper
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _assert_valid_transition(current: str, target: str) -> None:
        """
        Assert that moving from current → target is a legal transition.

        Args:
            current: Current order_status.
            target: Requested new status.

        Raises:
            ValueError: With a descriptive message if transition is illegal.
        """
        allowed = VALID_TRANSITIONS.get(current, [])
        if target not in allowed:
            raise ValueError(
                f"Invalid transition: '{current}' → '{target}'. "
                f"Allowed from '{current}': {allowed or ['(terminal — no transitions)']}"
            )

    # ─────────────────────────────────────────────────────────
    # * Private — AISP calculation
    # ─────────────────────────────────────────────────────────

    async def _calculate_aisp(
        self,
        quantity_kg: float,
        price_per_kg: float,
        commodity: str,
    ) -> AISPBreakdown:
        """
        Calculate AISP breakdown, delegating to PricingAgent when available.

        Falls back to fixed ratio allocation if no pricing agent is injected.

        Returns:
            AISPBreakdown with all cost components.
        """
        if self.pricing_agent and hasattr(self.pricing_agent, "calculate_aisp"):
            try:
                result = await self.pricing_agent.calculate_aisp(
                    quantity_kg=quantity_kg,
                    price_per_kg=price_per_kg,
                    commodity=commodity,
                )
                return AISPBreakdown(**result)
            except Exception as exc:
                logger.warning(f"PricingAgent AISP calculation failed: {exc}")

        # * Ratio-based fallback
        total = round(quantity_kg * price_per_kg, 2)
        return AISPBreakdown(
            farmer_payout=round(total * AISP_FARMER_RATIO, 2),
            logistics_cost=round(total * AISP_LOGISTICS_RATIO, 2),
            platform_margin=round(total * AISP_PLATFORM_RATIO, 2),
            risk_buffer=round(total * AISP_RISK_RATIO, 2),
            aisp_total=total,
            aisp_per_kg=round(price_per_kg, 2),
        )

    # ─────────────────────────────────────────────────────────
    # * Private — hauler assignment stub
    # ─────────────────────────────────────────────────────────

    async def _assign_hauler(self, listing: dict[str, Any]) -> Optional[str]:
        """
        Stub for LogisticsRouter hauler assignment.

        # TODO: Replace with real LogisticsRouterAgent.assign() call
        Returns None (manual assignment) until LogisticsRouter is implemented.
        """
        return None

    # ─────────────────────────────────────────────────────────
    # * Private — Digital Twin diff trigger
    # ─────────────────────────────────────────────────────────

    async def _trigger_twin_diff(
        self,
        departure_twin_id: str,
        arrival_photos: list[str],
        dispute_id: str,
    ) -> Optional[dict[str, Any]]:
        """
        Trigger AI diff between departure Digital Twin and arrival photos.

        Resolution order:
          1. DigitalTwinEngine (direct — preferred)
          2. QualityAssessmentAgent.compare_twin() (wrapper — legacy fallback)
          3. Placeholder pending report (no AI available)

        Persists liability + claim_percent alongside diff_report to the dispute record.

        Args:
            departure_twin_id: Digital twin UUID from the dispute record.
            arrival_photos:    S3 URLs submitted by the buyer.
            dispute_id:        UUID of the dispute record to update.

        Returns:
            diff_report dict if analysis completes, else None.
        """
        diff_result = None

        # * Path 1 — DigitalTwinEngine direct injection (preferred)
        if self.twin_engine and hasattr(self.twin_engine, "compare_arrival"):
            try:
                diff_result = await self.twin_engine.compare_arrival(
                    twin_id=departure_twin_id,
                    arrival_photos=arrival_photos,
                )
            except Exception as exc:
                logger.warning(
                    f"DigitalTwinEngine diff failed for dispute {dispute_id}: {exc}"
                )

        # * Path 2 — QualityAssessmentAgent.compare_twin() fallback
        if diff_result is None and self.quality_agent and hasattr(
            self.quality_agent, "compare_twin"
        ):
            try:
                diff_result = await self.quality_agent.compare_twin(
                    twin_id=departure_twin_id,
                    arrival_photos=arrival_photos,
                )
            except Exception as exc:
                logger.warning(
                    f"QualityAgent twin diff failed for dispute {dispute_id}: {exc}"
                )

        if diff_result is None:
            # ! No AI engine available — return pending placeholder
            pending: dict[str, Any] = {
                "status": "pending",
                "message": "AI diff engine not available — manual review required",
                "dispute_id": dispute_id,
            }
            await self._save_diff_report(dispute_id, pending)
            return pending

        # * Convert DiffReport dataclass → dict (handles both dataclass and dict returns)
        report_dict: dict[str, Any] = (
            diff_result.to_dict() if hasattr(diff_result, "to_dict") else dict(diff_result)
        )

        # * Persist diff report + liability to dispute record
        await self._save_diff_report(
            dispute_id,
            report_dict,
            liability=report_dict.get("liability"),
            claim_percent=report_dict.get("claim_percent"),
        )
        logger.info(
            f"Digital Twin diff complete for dispute {dispute_id}: "
            f"liability={report_dict.get('liability')} "
            f"claim={report_dict.get('claim_percent')}%"
        )
        return report_dict

    # ─────────────────────────────────────────────────────────
    # * Private — participant stats update stub
    # ─────────────────────────────────────────────────────────

    async def _update_participant_stats(self, order: dict[str, Any]) -> None:
        """
        Update farmer quality_score and buyer order stats after settlement.

        # TODO: Implement actual stats update once user stats schema is finalised
        Currently a no-op stub.
        """
        logger.debug(
            f"Stats update stub: farmer {order.get('farmer_id')}, "
            f"buyer {order.get('buyer_id')}"
        )

    # ─────────────────────────────────────────────────────────
    # * Private — notification stubs
    # ─────────────────────────────────────────────────────────

    async def _notify_order_created(
        self,
        order_id: str,
        listing: dict[str, Any],
        buyer_id: str,
    ) -> None:
        """
        Stub: notify farmer and buyer when a new order is created.

        # TODO: Wire to WhatsAppBotAgent.send_message() when ready
        """
        logger.info(
            f"[NOTIFY-STUB] Order {order_id} created — "
            f"farmer {listing.get('farmer_id')}, buyer {buyer_id}"
        )

    async def _notify_status_change(
        self,
        order_id: str,
        new_status: str,
        metadata: dict[str, Any],
    ) -> None:
        """
        Stub: notify relevant parties on any status transition.

        # TODO: Wire to WhatsAppBotAgent.send_status_update() when ready
        """
        logger.info(f"[NOTIFY-STUB] Order {order_id} → {new_status}")

    # ─────────────────────────────────────────────────────────
    # * Private — DB helpers
    # ─────────────────────────────────────────────────────────

    async def _fetch_listing(self, listing_id: str) -> Optional[dict[str, Any]]:
        """Fetch listing row by UUID; returns None when DB unavailable (dev/test mode)."""
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
        """
        Persist diff_report + liability to dispute record.

        Prefers update_dispute_diff_report() when available (Task 10 DB method),
        falls back to generic update_dispute() for backwards compatibility.
        """
        if not self.db:
            return
        if hasattr(self.db, "update_dispute_diff_report"):
            await self.db.update_dispute_diff_report(
                dispute_id,
                diff_report,
                liability=liability,
                claim_percent=claim_percent,
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

    # ─────────────────────────────────────────────────────────
    # * Private — row converters
    # ─────────────────────────────────────────────────────────

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
            aisp=OrderService._row_to_aisp(row),
            commodity=row.get("commodity"),
            farmer_id=str(row["farmer_id"]) if row.get("farmer_id") else None,
            buyer_name=row.get("buyer_name"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )


# ═══════════════════════════════════════════════════════════════
# * Module-level factory
# ═══════════════════════════════════════════════════════════════

def get_order_service(
    db: Optional[Any] = None,
    pricing_agent: Optional[Any] = None,
    quality_agent: Optional[Any] = None,
    twin_engine: Optional[Any] = None,
    notification_stub: Optional[Any] = None,
) -> OrderService:
    """
    Factory for creating an OrderService with injected dependencies.

    Args:
        db:               AuroraPostgresClient instance.
        pricing_agent:    PricingAgent instance for AISP calculation.
        quality_agent:    QualityAssessmentAgent for dispute diff analysis (fallback).
        twin_engine:      DigitalTwinEngine for direct twin access (preferred).
        notification_stub: Notification service for SMS/WhatsApp alerts.

    Returns:
        Configured OrderService.
    """
    return OrderService(
        db=db,
        pricing_agent=pricing_agent,
        quality_agent=quality_agent,
        twin_engine=twin_engine,
        notification_stub=notification_stub,
    )
