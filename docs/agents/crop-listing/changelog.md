# CropListingAgent — Changelog

## [1.0.0] — 2026-03-01 (Task 7)

### Added
- `CropListingAgent` — full implementation replacing corrupted stub
  - Natural language entity extraction (commodity, quantity, price) with EN/HI/KN aliases
  - Intent routing: `create`, `my_listings`, `cancel`, `update_price`
  - `process()` — NL interface for voice/chat queries
  - `execute()` — structured dict interface for orchestrators (actions: create/search/get/cancel/update_price)
  - Graceful fallbacks when `listing_service` is not configured
- `ListingService` — full lifecycle management
  - `create_listing(CreateListingRequest)` with 5-step auto-enrichment pipeline
  - `create_listing_from_dict()` — convenience wrapper for voice/agent use
  - `search_listings(filters)` → `PaginatedListings` with commodity/grade/price/district/adcl filters
  - `get_listing(id)` / `update_listing(id, updates)` / `cancel_listing(id)`
  - `get_farmer_listings(farmer_id)`
  - `attach_grade(listing_id, GradeAttachRequest)` with HITL flag logic
  - `expire_stale_listings()` — background job
  - `get_listing_service()` factory function for DI
- Pydantic models: `CreateListingRequest`, `UpdateListingRequest`, `GradeAttachRequest`,
  `ListingResponse`, `PaginatedListings`
- Commodity shelf-life calendar (10 crops + default)
- Grade ordering map for `min_grade` filter comparisons
- 7 REST endpoints in `src/api/routers/listings.py`
- `listings` router registered in `src/api/main.py` at `/api/v1`
- 50 unit tests in `tests/unit/test_listing_service.py`

### Fixed
- Corrupted class name `croplisting.Value.ToUpper()ropcroplisting.Value.ToUpper()istingAgent`
  → `CropListingAgent`
- Intent priority: `cancel` now checked before `my_listings` to prevent "cancel my listing"
  being routed to `_handle_my_listings`

## [Unreleased — pre-Task 7]
- Initial agent stub with corrupted class name and `NotImplementedError`
