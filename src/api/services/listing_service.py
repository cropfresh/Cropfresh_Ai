"""
Listing Service
===============
Full produce-listing lifecycle management.

Responsibilities:
- Create listings with auto-price suggestion and shelf-life expiry
- Trigger quality assessment when photos are attached
- Generate batch QR codes
- Tag listings against ADCL weekly demand list
- Search with commodity / grade / price / district filters
- Update, cancel (soft-delete), and expire stale listings
- Attach quality grades from QualityAssessmentAgent
"""

# * LISTING SERVICE MODULE
# NOTE: All DB writes go through AuroraPostgresClient CRUD methods (Task 6)
# NOTE: Pricing + quality deps are injected; service degrades gracefully if absent

import uuid
from datetime import UTC, date, datetime, timedelta
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
# * Constants — commodity shelf-life calendar (days)
# ═══════════════════════════════════════════════════════════════

SHELF_LIFE_DAYS: dict[str, int] = {
    "tomato": 7,
    "onion": 60,
    "potato": 90,
    "beans": 5,
    "okra": 4,
    "carrot": 21,
    "cauliflower": 7,
    "cucumber": 6,
    "chilli": 14,
    "leafy greens": 3,
    "spinach": 3,
    "coriander": 5,
    "default": 14,
}

# * Grade ordering for min_grade filter comparisons
GRADE_ORDER: dict[str, int] = {"A+": 4, "A": 3, "B": 2, "C": 1, "Unverified": 0}


# ═══════════════════════════════════════════════════════════════
# * Pydantic Models
# ═══════════════════════════════════════════════════════════════

class CreateListingRequest(BaseModel):
    """Request body for creating a new produce listing."""
    farmer_id: str
    commodity: str
    variety: Optional[str] = None
    quantity_kg: float = Field(gt=0, description="Must be positive")
    asking_price_per_kg: Optional[float] = Field(default=None, ge=0)
    harvest_date: Optional[date] = None
    pickup_lat: Optional[float] = None
    pickup_lon: Optional[float] = None
    photos: Optional[list[str]] = None          # S3 URLs


class UpdateListingRequest(BaseModel):
    """Partial update request for a listing."""
    asking_price_per_kg: Optional[float] = Field(default=None, ge=0)
    quantity_kg: Optional[float] = Field(default=None, gt=0)
    status: Optional[str] = None


class GradeAttachRequest(BaseModel):
    """Attach a quality grade to a listing."""
    grade: str = Field(description="One of: A+, A, B, C")
    cv_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    defect_types: Optional[list[str]] = None
    agent_id: Optional[str] = None             # Field agent who verified


class ListingResponse(BaseModel):
    """Complete listing representation returned to callers."""
    id: str
    farmer_id: str
    commodity: str
    variety: Optional[str] = None
    quantity_kg: float
    asking_price_per_kg: float
    suggested_price: Optional[float] = None    # Auto-suggested price if none given
    grade: str = "Unverified"
    cv_confidence: Optional[float] = None
    hitl_required: bool = False
    status: str = "active"
    adcl_tagged: bool = False
    batch_qr_code: Optional[str] = None
    expires_at: Optional[datetime] = None
    created_at: Optional[datetime] = None


class PaginatedListings(BaseModel):
    """Paginated listing search results."""
    items: list[ListingResponse]
    total: int
    limit: int
    offset: int
    has_more: bool


# ═══════════════════════════════════════════════════════════════
# * ListingService
# ═══════════════════════════════════════════════════════════════

class ListingService:
    """
    Full lifecycle management for produce listings.

    Dependencies (all optional — service degrades gracefully):
        db: AuroraPostgresClient — for persistence
        pricing_agent: PricingAgent — for auto-price suggestions
        quality_agent: QualityAssessmentAgent — for photo assessment
        adcl_agent: ADCLAgent — to check weekly demand tags

    Usage:
        service = ListingService(db=client, pricing_agent=agent)
        listing = await service.create_listing(CreateListingRequest(...))
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

    # ─────────────────────────────────────────────────────────
    # * Create
    # ─────────────────────────────────────────────────────────

    async def create_listing(self, request: CreateListingRequest) -> ListingResponse:
        """
        Create a new produce listing with auto-enrichment.

        Enrichment pipeline:
        1. Auto-suggest price if none given (PricingAgent → PricePrediction fallback)
        2. Set expiry date based on commodity shelf life
        3. Generate batch QR code (UUID-based)
        4. Tag ADCL flag if crop is on weekly demand list
        5. Trigger quality assessment if photos provided
        6. Persist to database

        Args:
            request: Validated CreateListingRequest.

        Returns:
            ListingResponse with all enriched fields.
        """
        # * Step 1: auto-price suggestion
        suggested_price: Optional[float] = None
        final_price = request.asking_price_per_kg

        if final_price is None:
            suggested_price = await self._suggest_price(request.commodity)
            final_price = suggested_price or 25.0     # hard fallback ₹25/kg

        # * Step 2: shelf-life expiry
        shelf_days = self._get_shelf_life(request.commodity)
        expires_at = datetime.now(UTC).replace(tzinfo=None) + timedelta(days=shelf_days)

        # * Step 3: batch QR code
        batch_qr = f"CF-{request.commodity[:3].upper()}-{uuid.uuid4().hex[:8].upper()}"

        # * Step 4: ADCL tag
        adcl_tagged = await self._check_adcl_tag(request.commodity)

        # * Build listing record
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

        # * Step 5: quality assessment (fire-and-forget when photos present)
        hitl_required = False
        if request.photos:
            hitl_required = await self._trigger_quality_assessment(
                listing_data, request.photos
            )

        listing_data["hitl_required"] = hitl_required

        # * Step 6: persist
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
        """
        Convenience wrapper accepting a plain dict (for agent + voice use).

        Args:
            data: Dict with same keys as CreateListingRequest.

        Returns:
            ListingResponse serialized as dict.
        """
        request = CreateListingRequest(**data)
        response = await self.create_listing(request)
        return response.model_dump()

    # ─────────────────────────────────────────────────────────
    # * Search
    # ─────────────────────────────────────────────────────────

    async def search_listings(
        self,
        filters: Optional[dict[str, Any]] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> PaginatedListings:
        """
        Search active listings with optional filters.

        Args:
            filters: Dict with optional keys: commodity, district, min_grade,
                     max_price_per_kg, adcl_tagged.
            limit: Page size (default 20, max 100).
            offset: Page offset.

        Returns:
            PaginatedListings with items + total + pagination info.
        """
        filters = filters or {}
        limit = min(limit, 100)
        search_params = {**filters, "limit": limit + 1, "offset": offset}

        rows = await self._fetch_listings(search_params)

        has_more = len(rows) > limit
        page_items = rows[:limit]

        # * Apply min_grade post-filter (grade ordering)
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

    # ─────────────────────────────────────────────────────────
    # * Get / Update / Cancel
    # ─────────────────────────────────────────────────────────

    async def get_listing(self, listing_id: str) -> Optional[ListingResponse]:
        """
        Fetch a single listing by UUID.

        Returns:
            ListingResponse, or None if not found.
        """
        row = await self._fetch_single(listing_id)
        return self._row_to_listing(row) if row else None

    async def update_listing(
        self,
        listing_id: str,
        updates: dict[str, Any],
    ) -> Optional[ListingResponse]:
        """
        Apply partial updates to an existing listing.

        Args:
            listing_id: Listing UUID.
            updates: Dict of fields to change (price, quantity, status).

        Returns:
            Updated ListingResponse, or None if not found.
        """
        row = await self._apply_update(listing_id, updates)
        return self._row_to_listing(row) if row else None

    async def cancel_listing(self, listing_id: str) -> bool:
        """
        Soft-delete a listing by setting status = 'cancelled'.

        Args:
            listing_id: Listing UUID.

        Returns:
            True if the listing was found and cancelled.
        """
        return await self._set_status(listing_id, "cancelled")

    async def get_farmer_listings(
        self,
        farmer_id: str,
        status: str = "active",
    ) -> list[ListingResponse]:
        """
        Return all listings for a specific farmer.

        Args:
            farmer_id: Farmer UUID.
            status: Status filter (default 'active').

        Returns:
            List of ListingResponse objects.
        """
        rows = await self._fetch_farmer_listings(farmer_id, status)
        return [self._row_to_listing(r) for r in rows]

    # ─────────────────────────────────────────────────────────
    # * Grade attachment
    # ─────────────────────────────────────────────────────────

    async def attach_grade(
        self,
        listing_id: str,
        grade_request: GradeAttachRequest,
    ) -> Optional[ListingResponse]:
        """
        Attach a quality grade to a listing, updating hitl_required flag.

        Args:
            listing_id: Listing UUID.
            grade_request: Grade, confidence, defect types.

        Returns:
            Updated ListingResponse, or None if listing not found.
        """
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

    # ─────────────────────────────────────────────────────────
    # * Background jobs
    # ─────────────────────────────────────────────────────────

    async def expire_stale_listings(self) -> int:
        """
        Background job: expire all listings whose expiry date has passed.

        Returns:
            Number of listings expired.
        """
        count = await self._expire_past_expiry()
        if count:
            logger.info(f"Expired {count} stale listing(s)")
        return count

    # ─────────────────────────────────────────────────────────
    # * Private enrichment helpers
    # ─────────────────────────────────────────────────────────

    async def _suggest_price(self, commodity: str) -> Optional[float]:
        """Fetch price recommendation from pricing/prediction agent."""
        if not self.pricing_agent:
            return None
        try:
            if hasattr(self.pricing_agent, "predict"):
                prediction = await self.pricing_agent.predict(commodity=commodity)
                return round(prediction.current_price / 100, 2)   # quintal → kg
            if hasattr(self.pricing_agent, "get_recommendation"):
                rec = await self.pricing_agent.get_recommendation(commodity=commodity)
                return rec.get("recommended_price_per_kg")
        except Exception as exc:
            logger.warning(f"Price suggestion failed for {commodity}: {exc}")
        return None

    async def _check_adcl_tag(self, commodity: str) -> bool:
        """Check if commodity appears in current ADCL weekly demand list."""
        if not self.adcl_agent:
            return False
        try:
            result = await self.adcl_agent.get_weekly_demand()
            crops = result.get("crops", [])
            return any(
                c.get("crop", "").lower() == commodity.lower()
                for c in crops
            )
        except Exception as exc:
            logger.warning(f"ADCL tag check failed: {exc}")
        return False

    async def _trigger_quality_assessment(
        self, listing_data: dict, photos: list[str]
    ) -> bool:
        """
        Trigger quality assessment for a listing with photos.

        Returns:
            hitl_required flag from assessment result.
        """
        if not self.quality_agent:
            return True     # ! Default to HITL when no agent available

        try:
            result = await self.quality_agent.assess(
                photos=photos,
                commodity=listing_data.get("commodity", ""),
            )
            return result.get("hitl_required", True)
        except Exception as exc:
            logger.warning(f"Quality assessment trigger failed: {exc}")
            return True

    @staticmethod
    def _get_shelf_life(commodity: str) -> int:
        """Return shelf life in days for a commodity (case-insensitive)."""
        return SHELF_LIFE_DAYS.get(commodity.lower(), SHELF_LIFE_DAYS["default"])

    # ─────────────────────────────────────────────────────────
    # * Private DB helpers (wrap postgres_client or simulate in-memory)
    # ─────────────────────────────────────────────────────────

    async def _persist_listing(self, listing_data: dict) -> str:
        """Persist listing to DB; return listing ID."""
        if self.db and hasattr(self.db, "create_listing"):
            return await self.db.create_listing(listing_data)
        # * In-memory fallback for dev/test without DB
        return str(uuid.uuid4())

    async def _fetch_listings(self, params: dict) -> list[dict]:
        """Fetch filtered listings from DB."""
        if self.db and hasattr(self.db, "search_listings"):
            return await self.db.search_listings(params)
        return []

    async def _fetch_single(self, listing_id: str) -> Optional[dict]:
        """Fetch a single listing row by UUID."""
        if self.db and hasattr(self.db, "get_listing"):
            return await self.db.get_listing(listing_id)
        return None

    async def _apply_update(
        self, listing_id: str, updates: dict
    ) -> Optional[dict]:
        """Apply updates to a listing row, return updated row."""
        if self.db and hasattr(self.db, "update_listing"):
            return await self.db.update_listing(listing_id, updates)
        # * Fallback: return a mock updated row for test/dev
        return {"id": listing_id, **updates}

    async def _set_status(self, listing_id: str, status: str) -> bool:
        """Set listing status (soft-delete / expire)."""
        if self.db and hasattr(self.db, "update_listing"):
            result = await self.db.update_listing(listing_id, {"status": status})
            return result is not None
        return False

    async def _fetch_farmer_listings(
        self, farmer_id: str, status: str
    ) -> list[dict]:
        """Fetch all listings for a farmer."""
        if self.db and hasattr(self.db, "search_listings"):
            return await self.db.search_listings(
                {"farmer_id": farmer_id, "status": status, "limit": 100}
            )
        return []

    async def _expire_past_expiry(self) -> int:
        """Expire all listings past their expiry timestamp."""
        if self.db and hasattr(self.db, "expire_stale_listings"):
            return await self.db.expire_stale_listings()
        return 0

    @staticmethod
    def _row_to_listing(row: dict) -> ListingResponse:
        """Convert a DB row dict to a ListingResponse."""
        return ListingResponse(
            id=str(row.get("id", "")),
            farmer_id=str(row.get("farmer_id", "")),
            commodity=row.get("commodity", ""),
            variety=row.get("variety"),
            quantity_kg=float(row.get("quantity_kg", 0)),
            asking_price_per_kg=float(row.get("asking_price_per_kg", 0)),
            suggested_price=row.get("suggested_price"),
            grade=row.get("grade", "Unverified"),
            cv_confidence=row.get("cv_confidence"),
            hitl_required=bool(row.get("hitl_required", False)),
            status=row.get("status", "active"),
            adcl_tagged=bool(row.get("adcl_tagged", False)),
            batch_qr_code=row.get("batch_qr_code"),
            expires_at=row.get("expires_at"),
            created_at=row.get("created_at"),
        )


# ═══════════════════════════════════════════════════════════════
# * Module-level factory
# ═══════════════════════════════════════════════════════════════

def get_listing_service(
    db: Optional[Any] = None,
    pricing_agent: Optional[Any] = None,
    quality_agent: Optional[Any] = None,
    adcl_agent: Optional[Any] = None,
) -> ListingService:
    """
    Factory for creating a ListingService with injected dependencies.

    Args:
        db: AuroraPostgresClient instance.
        pricing_agent: PricingAgent or PricePredictionAgent instance.
        quality_agent: QualityAssessmentAgent instance.
        adcl_agent: ADCLAgent instance.

    Returns:
        Configured ListingService.
    """
    return ListingService(
        db=db,
        pricing_agent=pricing_agent,
        quality_agent=quality_agent,
        adcl_agent=adcl_agent,
    )
