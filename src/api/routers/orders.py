"""
Orders API Router
=================
REST endpoints for the CropFresh order lifecycle management.

Endpoints:
    POST   /api/v1/orders                  — Create order from matched listing
    GET    /api/v1/orders/{id}             — Get order details
    PATCH  /api/v1/orders/{id}/status      — Advance through state machine
    GET    /api/v1/orders                  — List orders (farmer_id or buyer_id filter)
    POST   /api/v1/orders/{id}/dispute     — Raise a dispute with evidence
    POST   /api/v1/orders/{id}/settle      — Release escrow and settle
    GET    /api/v1/orders/{id}/aisp        — Get AISP price breakdown
"""

# * ORDERS ROUTER MODULE
# NOTE: OrderService is resolved from app.state or instantiated fresh per request

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from loguru import logger

from src.api.services.order_service import (
    AISPBreakdown,
    CreateOrderRequest,
    DisputeResponse,
    OrderResponse,
    RaiseDisputeRequest,
    UpdateStatusRequest,
    get_order_service,
)

router = APIRouter()


# ─────────────────────────────────────────────────────────────
# * Dependency helper
# ─────────────────────────────────────────────────────────────

def _service(request: Request):
    """Resolve OrderService from app.state or create a bare instance."""
    if hasattr(request.app.state, "order_service"):
        return request.app.state.order_service
    return get_order_service(
        db=getattr(request.app.state, "db", None),
        pricing_agent=getattr(request.app.state, "pricing_agent", None),
        quality_agent=getattr(request.app.state, "quality_agent", None),
    )


# ─────────────────────────────────────────────────────────────
# * POST /orders — Create order
# ─────────────────────────────────────────────────────────────

@router.post(
    "/orders",
    response_model=OrderResponse,
    status_code=201,
    summary="Create a new order from a matched listing",
    tags=["orders"],
)
async def create_order(
    body: CreateOrderRequest,
    request: Request,
) -> OrderResponse:
    """
    Create a new order from a matched listing + buyer.

    - AISP price breakdown is calculated automatically.
    - Escrow is set to `held` on creation.
    - The listing status is updated to `matched`.
    - Farmer and buyer receive notification stubs.
    """
    svc = _service(request)
    try:
        return await svc.create_order(body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"POST /orders error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────
# * GET /orders — List by farmer or buyer
# NOTE: Must be registered BEFORE /orders/{id} to avoid route collision
# ─────────────────────────────────────────────────────────────

@router.get(
    "/orders",
    response_model=list[OrderResponse],
    summary="List orders filtered by farmer or buyer",
    tags=["orders"],
)
async def list_orders(
    request: Request,
    farmer_id: Optional[str] = Query(default=None, description="Filter by farmer UUID"),
    buyer_id: Optional[str] = Query(default=None, description="Filter by buyer UUID"),
    status: Optional[str] = Query(default=None, description="Filter by order_status"),
) -> list[OrderResponse]:
    """
    Retrieve order history.

    Provide either `farmer_id` or `buyer_id`. At least one is required.
    Optionally filter by `status` (e.g., `confirmed`, `in_transit`, `settled`).
    """
    if not farmer_id and not buyer_id:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one of: farmer_id, buyer_id",
        )

    svc = _service(request)
    try:
        if farmer_id:
            return await svc.get_orders_by_farmer(farmer_id=farmer_id, status=status)
        return await svc.get_orders_by_buyer(buyer_id=buyer_id, status=status)  # type: ignore[arg-type]
    except Exception as exc:
        logger.error(f"GET /orders error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────
# * GET /orders/{id} — Get by ID
# ─────────────────────────────────────────────────────────────

@router.get(
    "/orders/{order_id}",
    response_model=OrderResponse,
    summary="Get an order by ID",
    tags=["orders"],
)
async def get_order(
    order_id: str,
    request: Request,
) -> OrderResponse:
    """Fetch a single order with full AISP breakdown by its UUID."""
    svc = _service(request)
    try:
        result = await svc.get_order(order_id)
    except Exception as exc:
        logger.error(f"GET /orders/{order_id} error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    if result is None:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    return result


# ─────────────────────────────────────────────────────────────
# * PATCH /orders/{id}/status — Advance state machine
# ─────────────────────────────────────────────────────────────

@router.patch(
    "/orders/{order_id}/status",
    response_model=OrderResponse,
    summary="Advance an order through the state machine",
    tags=["orders"],
)
async def update_order_status(
    order_id: str,
    body: UpdateStatusRequest,
    request: Request,
) -> OrderResponse:
    """
    Advance an order to the next valid state.

    Valid transitions:
    ```
    confirmed → pickup_scheduled → in_transit → delivered → settled
                                       ↓
                                   disputed → ai_analysed → resolved → settled
                                                          → escalated → settled / refunded
    ```

    Returns `422` if the requested transition is not allowed from the current state.
    """
    svc = _service(request)
    try:
        return await svc.update_status(order_id, body.status, body.metadata)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"PATCH /orders/{order_id}/status error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────
# * POST /orders/{id}/dispute — Raise dispute
# ─────────────────────────────────────────────────────────────

@router.post(
    "/orders/{order_id}/dispute",
    response_model=DisputeResponse,
    status_code=201,
    summary="Raise a dispute with arrival evidence",
    tags=["orders"],
)
async def raise_dispute(
    order_id: str,
    body: RaiseDisputeRequest,
    request: Request,
) -> DisputeResponse:
    """
    Open a dispute for an order in `in_transit` or `delivered` status.

    - Supply `arrival_photos` (S3 URLs) as evidence.
    - Supply `departure_twin_id` to trigger AI Digital Twin diff analysis.
    - The order status is automatically advanced to `disputed`.
    """
    svc = _service(request)
    try:
        return await svc.raise_dispute(order_id, body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"POST /orders/{order_id}/dispute error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────
# * POST /orders/{id}/settle — Settle and release escrow
# ─────────────────────────────────────────────────────────────

@router.post(
    "/orders/{order_id}/settle",
    response_model=OrderResponse,
    summary="Settle order and release escrow to farmer",
    tags=["orders"],
)
async def settle_order(
    order_id: str,
    request: Request,
) -> OrderResponse:
    """
    Settle an order and release escrow funds to the farmer.

    - Valid only from `delivered` or `resolved` status.
    - Sets `escrow_status = released`.
    - Updates farmer and buyer stats (stub).
    """
    svc = _service(request)
    try:
        return await svc.settle_order(order_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"POST /orders/{order_id}/settle error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────
# * GET /orders/{id}/aisp — AISP breakdown
# ─────────────────────────────────────────────────────────────

@router.get(
    "/orders/{order_id}/aisp",
    response_model=AISPBreakdown,
    summary="Get AISP price breakdown for an order",
    tags=["orders"],
)
async def get_aisp_breakdown(
    order_id: str,
    request: Request,
) -> AISPBreakdown:
    """
    Retrieve the AI-Set Price (AISP) breakdown for an existing order.

    Returns the split between farmer payout, logistics cost,
    platform margin, and risk buffer.
    """
    svc = _service(request)
    try:
        result = await svc.get_aisp_breakdown(order_id)
    except Exception as exc:
        logger.error(f"GET /orders/{order_id}/aisp error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    if result is None:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    return result
