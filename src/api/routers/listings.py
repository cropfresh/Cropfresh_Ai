"""
Listings API Router
===================
REST endpoints for the CropFresh produce listing marketplace.

Endpoints:
    POST   /api/v1/listings              — Create a new listing
    GET    /api/v1/listings              — Search / paginated list
    GET    /api/v1/listings/farmer/{id}  — All listings for a farmer
    GET    /api/v1/listings/{id}         — Get listing by ID
    PATCH  /api/v1/listings/{id}         — Update price / quantity / status
    DELETE /api/v1/listings/{id}         — Soft-cancel a listing
    POST   /api/v1/listings/{id}/grade   — Attach quality grade
"""

# * LISTINGS ROUTER MODULE
# NOTE: ListingService is resolved from app.state or instantiated fresh per request

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from loguru import logger

from src.api.services.listing_service import (
    CreateListingRequest,
    GradeAttachRequest,
    ListingResponse,
    PaginatedListings,
    UpdateListingRequest,
    get_listing_service,
)

router = APIRouter()


# ─────────────────────────────────────────────────────────────
# * Dependency helper
# ─────────────────────────────────────────────────────────────

def _service(request: Request):
    """Resolve ListingService from app.state or create a bare instance."""
    if hasattr(request.app.state, "listing_service"):
        return request.app.state.listing_service
    return get_listing_service(
        db=getattr(request.app.state, "db", None),
        pricing_agent=getattr(request.app.state, "pricing_agent", None),
        quality_agent=getattr(request.app.state, "quality_agent", None),
        adcl_agent=getattr(request.app.state, "adcl_agent", None),
    )


# ─────────────────────────────────────────────────────────────
# * POST /listings — Create listing
# ─────────────────────────────────────────────────────────────

@router.post(
    "/listings",
    response_model=ListingResponse,
    status_code=201,
    summary="Create a new produce listing",
    tags=["listings"],
)
async def create_listing(
    body: CreateListingRequest,
    request: Request,
) -> ListingResponse:
    """
    Create a new produce listing with auto-enrichment.

    - If `asking_price_per_kg` is omitted, a price is suggested by the Pricing Agent.
    - If `photos` are provided, quality assessment is triggered.
    - `expires_at` is set automatically based on commodity shelf life.
    - A unique `batch_qr_code` is generated.
    """
    svc = _service(request)
    try:
        return await svc.create_listing(body)
    except Exception as exc:
        logger.error(f"POST /listings error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────
# * GET /listings — Search / paginated list
# ─────────────────────────────────────────────────────────────

@router.get(
    "/listings",
    response_model=PaginatedListings,
    summary="Search listings with filters",
    tags=["listings"],
)
async def search_listings(
    request: Request,
    commodity: Optional[str] = Query(default=None, description="Filter by crop name"),
    district: Optional[str] = Query(default=None, description="Filter by district"),
    min_grade: Optional[str] = Query(default=None, description="Minimum grade: A+/A/B/C"),
    max_price: Optional[float] = Query(default=None, ge=0, description="Max ₹/kg"),
    adcl_tagged: Optional[bool] = Query(default=None, description="Only ADCL demand list crops"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> PaginatedListings:
    """
    Search active produce listings with optional filters.

    Results are sorted by creation date (newest first).
    Use `offset` + `limit` for pagination.
    """
    filters: dict = {}
    if commodity:
        filters["commodity"] = commodity
    if district:
        filters["district"] = district
    if min_grade:
        filters["min_grade"] = min_grade
    if max_price is not None:
        filters["max_price_per_kg"] = max_price
    if adcl_tagged is not None:
        filters["adcl_tagged"] = adcl_tagged

    svc = _service(request)
    try:
        return await svc.search_listings(filters=filters, limit=limit, offset=offset)
    except Exception as exc:
        logger.error(f"GET /listings error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────
# * GET /listings/farmer/{farmer_id} — Farmer's own listings
# NOTE: Must be registered BEFORE /listings/{id} to avoid route collision
# ─────────────────────────────────────────────────────────────

@router.get(
    "/listings/farmer/{farmer_id}",
    response_model=list[ListingResponse],
    summary="Get all listings for a farmer",
    tags=["listings"],
)
async def get_farmer_listings(
    farmer_id: str,
    request: Request,
    status: str = Query(default="active", description="Filter by status"),
) -> list[ListingResponse]:
    """Return all produce listings created by the specified farmer."""
    svc = _service(request)
    try:
        return await svc.get_farmer_listings(farmer_id=farmer_id, status=status)
    except Exception as exc:
        logger.error(f"GET /listings/farmer/{farmer_id} error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────
# * GET /listings/{id} — Get by ID
# ─────────────────────────────────────────────────────────────

@router.get(
    "/listings/{listing_id}",
    response_model=ListingResponse,
    summary="Get a listing by ID",
    tags=["listings"],
)
async def get_listing(
    listing_id: str,
    request: Request,
) -> ListingResponse:
    """Fetch a single listing by its UUID."""
    svc = _service(request)
    try:
        result = await svc.get_listing(listing_id)
    except Exception as exc:
        logger.error(f"GET /listings/{listing_id} error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    if result is None:
        raise HTTPException(status_code=404, detail=f"Listing {listing_id} not found")
    return result


# ─────────────────────────────────────────────────────────────
# * PATCH /listings/{id} — Partial update
# ─────────────────────────────────────────────────────────────

@router.patch(
    "/listings/{listing_id}",
    response_model=ListingResponse,
    summary="Update a listing (price / quantity / status)",
    tags=["listings"],
)
async def update_listing(
    listing_id: str,
    body: UpdateListingRequest,
    request: Request,
) -> ListingResponse:
    """
    Partially update an existing listing.

    Allowed fields: `asking_price_per_kg`, `quantity_kg`, `status`.
    Only provided (non-null) fields are changed.
    """
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No update fields provided")

    svc = _service(request)
    try:
        result = await svc.update_listing(listing_id, updates)
    except Exception as exc:
        logger.error(f"PATCH /listings/{listing_id} error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    if result is None:
        raise HTTPException(status_code=404, detail=f"Listing {listing_id} not found")
    return result


# ─────────────────────────────────────────────────────────────
# * DELETE /listings/{id} — Soft cancel
# ─────────────────────────────────────────────────────────────

@router.delete(
    "/listings/{listing_id}",
    status_code=204,
    summary="Cancel (soft-delete) a listing",
    tags=["listings"],
)
async def cancel_listing(
    listing_id: str,
    request: Request,
) -> None:
    """
    Soft-cancel a listing by setting its status to `cancelled`.

    Returns 204 No Content on success, 404 if not found.
    """
    svc = _service(request)
    try:
        ok = await svc.cancel_listing(listing_id)
    except Exception as exc:
        logger.error(f"DELETE /listings/{listing_id} error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    if not ok:
        raise HTTPException(status_code=404, detail=f"Listing {listing_id} not found")


# ─────────────────────────────────────────────────────────────
# * POST /listings/{id}/grade — Attach quality grade
# ─────────────────────────────────────────────────────────────

@router.post(
    "/listings/{listing_id}/grade",
    response_model=ListingResponse,
    summary="Attach a quality grade to a listing",
    tags=["listings"],
)
async def attach_grade(
    listing_id: str,
    body: GradeAttachRequest,
    request: Request,
) -> ListingResponse:
    """
    Attach a quality assessment result to an existing listing.

    If `cv_confidence` < 0.70 or grade is `A+`, `hitl_required` is
    automatically set to `True` (field agent verification required).
    """
    svc = _service(request)
    try:
        result = await svc.attach_grade(listing_id, body)
    except Exception as exc:
        logger.error(f"POST /listings/{listing_id}/grade error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    if result is None:
        raise HTTPException(status_code=404, detail=f"Listing {listing_id} not found")
    return result
