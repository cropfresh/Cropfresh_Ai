# Task 7: Build Crop Listing Service

> **Priority:** 🟠 P1 | **Phase:** 2 | **Effort:** 3–4 days  
> **Files:** `src/agents/crop_listing/agent.py`, `src/api/services/listing_service.py` [NEW], `src/api/routers/listings.py` [NEW]  
> **Score Target:** 9/10 — Full CRUD with auto-pricing, quality triggers, and expiry management

---

## 📌 Problem Statement

`src/agents/crop_listing/agent.py` is a stub. No listing service or REST endpoints exist. Farmers need to create, view, update, and manage produce listings via voice or API.

---

## 🏗️ Implementation Spec

### Listing Service (`listing_service.py`)
```python
class ListingService:
    """
    Complete listing lifecycle management.
    
    Features:
    - Create from voice input or API
    - Auto-suggest price from PricingAgent
    - Trigger quality assessment when photos attached
    - Auto-expire based on commodity shelf life
    - Search with filters (commodity, location, grade, price range)
    """
    
    async def create_listing(self, data: CreateListingRequest) -> Listing:
        """
        Create a new produce listing.
        
        Auto-enrichment:
        1. If no price → fetch recommendation from PricingAgent
        2. If photos → trigger QualityAssessmentAgent  
        3. Set expiry based on commodity shelf life
        4. Generate batch QR code
        5. Check ADCL tag (is this on weekly demand list?)
        """
    
    async def search_listings(
        self,
        commodity: Optional[str] = None,
        district: Optional[str] = None,
        min_grade: Optional[str] = None,
        max_price: Optional[float] = None,
        status: str = 'active',
        limit: int = 20,
        offset: int = 0,
    ) -> PaginatedResult[Listing]:
        """Search with filtering, sorting, and pagination."""
    
    async def expire_stale_listings(self) -> int:
        """Background job: expire listings past their shelf life."""
```

### REST API Endpoints (`routers/listings.py`)
```
POST   /api/v1/listings              → Create listing
GET    /api/v1/listings              → Search/list (with query params)
GET    /api/v1/listings/{id}         → Get by ID
PATCH  /api/v1/listings/{id}         → Update (price, quantity, status)
DELETE /api/v1/listings/{id}         → Soft delete (set status=cancelled)
GET    /api/v1/listings/farmer/{id}  → Get farmer's listings
POST   /api/v1/listings/{id}/grade   → Attach quality grade
```

### Pydantic Models
```python
class CreateListingRequest(BaseModel):
    farmer_id: str
    commodity: str
    variety: Optional[str] = None
    quantity_kg: float = Field(gt=0)
    asking_price_per_kg: Optional[float] = None  # Auto-suggested if None
    harvest_date: Optional[date] = None
    pickup_lat: Optional[float] = None
    pickup_lon: Optional[float] = None
    photos: Optional[list[str]] = None  # S3 URLs

class ListingResponse(BaseModel):
    id: str
    farmer_id: str
    commodity: str
    quantity_kg: float
    asking_price_per_kg: float
    suggested_price: Optional[float] = None
    grade: str = 'Unverified'
    status: str = 'active'
    expires_at: datetime
    created_at: datetime
```

---

## ✅ Acceptance Criteria

| # | Criterion | Weight |
|---|-----------|--------|
| 1 | CRUD endpoints work with proper validation | 25% |
| 2 | Auto-price suggestion when no price given | 20% |
| 3 | Quality assessment triggered on photo upload | 15% |
| 4 | Shelf-life-based auto-expiry | 15% |
| 5 | Search with commodity + location + grade filters | 15% |
| 6 | Voice agent `create_listing` intent creates real DB record | 10% |
