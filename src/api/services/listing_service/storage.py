"""
Listing Storage Mixin
=====================
Handles low-level persistence for the Listing Service.
"""

import uuid
from typing import Any, Optional

from .models import ListingResponse


class ListingStorageMixin:
    """Mixin containing database operations for listings."""

    db: Optional[Any]

    async def _persist_listing(self, listing_data: dict) -> str:
        """Persist listing to DB; return listing ID."""
        if self.db and hasattr(self.db, "create_listing"):
            return await self.db.create_listing(listing_data)
        # In-memory fallback for dev/test without DB
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

    async def _apply_update(self, listing_id: str, updates: dict) -> Optional[dict]:
        """Apply updates to a listing row, return updated row."""
        if self.db and hasattr(self.db, "update_listing"):
            return await self.db.update_listing(listing_id, updates)
        # Fallback: return a mock updated row for test/dev
        return {"id": listing_id, **updates}

    async def _set_status(self, listing_id: str, status: str) -> bool:
        """Set listing status (soft-delete / expire)."""
        if self.db and hasattr(self.db, "update_listing"):
            result = await self.db.update_listing(listing_id, {"status": status})
            return result is not None
        return False

    async def _fetch_farmer_listings(self, farmer_id: str, status: str) -> list[dict]:
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
