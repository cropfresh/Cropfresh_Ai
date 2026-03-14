"""
Listing Models
==============
Pydantic schemas for produce-listing lifecycle management.
"""

from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel, Field


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
