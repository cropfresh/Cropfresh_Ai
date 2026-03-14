"""
Listing Service API
===================
Orchestrator for produce-listing lifecycle management.
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Optional

from loguru import logger

from .constants import GRADE_ORDER
from .enrichment import ListingEnrichmentMixin
from .models import (
    CreateListingRequest,
    GradeAttachRequest,
    ListingResponse,
    PaginatedListings,
)
from .storage import ListingStorageMixin


class ListingService(ListingStorageMixin, ListingEnrichmentMixin):
    """
    Full lifecycle management for produce listings.

    Dependencies (all optional — service degrades gracefully):
        db: AuroraPostgresClient — for persistence
        pricing_agent: PricingAgent — for auto-price suggestions
        quality_agent: QualityAssessmentAgent — for photo assessment
        adcl_agent: ADCLAgent — to check weekly demand tags
    """

    def __init__(
        self,
        db: Optional[Any] = None,
        pricing_agent: Optional[Any] = None,
        quality_agent: Optional[Any] = None,
        adcl_agent: Optional[Any] = None,
    ) -> None:
        self.db = db
        self.pricing_agent = pricing_agent
        self.quality_agent = quality_agent
        self.adcl_agent = adcl_agent

    async def create_listing(self, request: CreateListingRequest) -> ListingResponse:
        """Create a new produce listing with auto-enrichment."""
        suggested_price = None
        final_price = request.asking_price_per_kg

        if final_price is None:
            suggested_price = await self._suggest_price(request.commodity)
            final_price = suggested_price or 25.0

        shelf_days = self._get_shelf_life(request.commodity)
        expires_at = datetime.now(UTC).replace(tzinfo=None) + timedelta(days=shelf_days)
        batch_qr = f"CF-{request.commodity[:3].upper()}-{uuid.uuid4().hex[:8].upper()}"
        adcl_tagged = await self._check_adcl_tag(request.commodity)

        listing_data: dict[str, Any] = {
            "farmer_id": request.farmer_id,
            "commodity": request.commodity,
            "variety": request.variety,
            "quantity_kg": request.quantity_kg,
            "asking_price_per_kg": final_price,
            "grade": "Unverified",
            "harvest_date": request.harvest_date,
            "batch_qr_code": batch_qr,
            "adcl_tagged": adcl_tagged,
            "expires_at": expires_at,
        }

        hitl_required = False
        if request.photos:
            hitl_required = await self._trigger_quality_assessment(
                listing_data, request.photos
            )

        listing_data["hitl_required"] = hitl_required
        listing_id = await self._persist_listing(listing_data)

        logger.info(
            f"Created listing {listing_id}: {request.quantity_kg}kg "
            f"{request.commodity} @ ₹{final_price}/kg "
            f"(expires {expires_at.date()}, adcl={adcl_tagged})"
        )

        return ListingResponse(
            id=listing_id,
            farmer_id=request.farmer_id,
            commodity=request.commodity,
            variety=request.variety,
            quantity_kg=request.quantity_kg,
            asking_price_per_kg=final_price,
            suggested_price=suggested_price,
            grade="Unverified",
            hitl_required=hitl_required,
            status="active",
            adcl_tagged=adcl_tagged,
            batch_qr_code=batch_qr,
            expires_at=expires_at,
            created_at=datetime.now(UTC).replace(tzinfo=None),
        )

    async def create_listing_from_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convenience wrapper accepting a plain dict (for agent + voice use)."""
        request = CreateListingRequest(**data)
        response = await self.create_listing(request)
        return response.model_dump()

    async def search_listings(
        self,
        filters: Optional[dict[str, Any]] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> PaginatedListings:
        """Search active listings with optional filters."""
        filters = filters or {}
        limit = min(limit, 100)
        search_params = {**filters, "limit": limit + 1, "offset": offset}

        rows = await self._fetch_listings(search_params)
        has_more = len(rows) > limit
        page_items = rows[:limit]

        if min_grade := filters.get("min_grade"):
            min_score = GRADE_ORDER.get(min_grade, 0)
            page_items = [
                r for r in page_items
                if GRADE_ORDER.get(r.get("grade", "Unverified"), 0) >= min_score
            ]

        items = [self._row_to_listing(r) for r in page_items]
        return PaginatedListings(
            items=items,
            total=len(items),
            limit=limit,
            offset=offset,
            has_more=has_more,
        )

    async def get_listing(self, listing_id: str) -> Optional[ListingResponse]:
        """Fetch a single listing by UUID."""
        row = await self._fetch_single(listing_id)
        return self._row_to_listing(row) if row else None

    async def update_listing(
        self,
        listing_id: str,
        updates: dict[str, Any],
    ) -> Optional[ListingResponse]:
        """Apply partial updates to an existing listing."""
        row = await self._apply_update(listing_id, updates)
        return self._row_to_listing(row) if row else None

    async def cancel_listing(self, listing_id: str) -> bool:
        """Soft-delete a listing by setting status = 'cancelled'."""
        return await self._set_status(listing_id, "cancelled")

    async def get_farmer_listings(
        self,
        farmer_id: str,
        status: str = "active",
    ) -> list[ListingResponse]:
        """Return all listings for a specific farmer."""
        rows = await self._fetch_farmer_listings(farmer_id, status)
        return [self._row_to_listing(r) for r in rows]

    async def attach_grade(
        self,
        listing_id: str,
        grade_request: GradeAttachRequest,
    ) -> Optional[ListingResponse]:
        """Attach a quality grade to a listing, updating hitl_required flag."""
        hitl_required = (
            grade_request.cv_confidence is not None
            and grade_request.cv_confidence < 0.70
        ) or grade_request.grade == "A+"

        updates = {
            "grade": grade_request.grade,
            "cv_confidence": grade_request.cv_confidence,
            "hitl_required": hitl_required,
        }
        logger.info(f"Attaching grade {grade_request.grade} to listing {listing_id}")
        return await self.update_listing(listing_id, updates)

    async def expire_stale_listings(self) -> int:
        """Background job: expire all listings whose expiry date has passed."""
        count = await self._expire_past_expiry()
        if count:
            logger.info(f"Expired {count} stale listing(s)")
        return count


def get_listing_service(
    db: Optional[Any] = None,
    pricing_agent: Optional[Any] = None,
    quality_agent: Optional[Any] = None,
    adcl_agent: Optional[Any] = None,
) -> ListingService:
    """Factory for creating a ListingService with injected dependencies."""
    return ListingService(
        db=db,
        pricing_agent=pricing_agent,
        quality_agent=quality_agent,
        adcl_agent=adcl_agent,
    )
