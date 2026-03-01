# CropListingAgent — Specification

> **Version:** 1.0.0 | **Status:** ✅ Implemented (Task 7, 2026-03-01)  
> **Files:** `src/agents/crop_listing/agent.py`, `src/api/services/listing_service.py`

---

## Purpose

Natural language interface for creating and managing produce listings. Parses farmer
voice/text input (English, Hindi, Kannada) into structured listing operations and
delegates business logic to `ListingService`.

---

## Inputs

| Field | Type | Description |
|-------|------|-------------|
| `query` | `str` | Farmer's voice/text (NL) — e.g. "sell 200 kg tamatar at ₹25" |
| `context.farmer_id` | `str` | Authenticated farmer UUID |
| `context.listing_id` | `str` | Optional — used for cancel/update intents |
| `context.session_id` | `str` | Optional — for multi-turn tracking |

---

## Outputs

Returns `AgentResponse` with:

| Field | Description |
|-------|-------------|
| `content` | Voice-friendly confirmation or clarifying question |
| `confidence` | 0.5 (clarifying) → 0.95 (successful create) |
| `steps` | `listing_id`, `price`, `commodity` for downstream use |
| `error` | Non-null if service exception occurred |

---

## Supported Intents

| Intent | Triggers | Handler |
|--------|----------|---------|
| `create` | Default — commodity + qty detected | `_handle_create()` |
| `my_listings` | "my listing", "show listing", "mere listing" | `_handle_my_listings()` |
| `cancel` | "cancel", "withdraw", "remove", "delete" | `_handle_cancel()` |
| `update_price` | "update price", "change price", "new price" | `_handle_update_price()` |

> **Note:** `cancel` is matched before `my_listings` to prevent prefix collision ("cancel my listing").

---

## Entity Extraction

| Entity | Method | Example |
|--------|--------|---------|
| Commodity | `COMMODITY_ALIASES` dict lookup (EN/HI/KN) | tamatar → Tomato |
| Quantity | Regex `(\d+)\s*kg` or quintal conversion | "200 kg" → 200.0 |
| Price | Regex `₹/Rs/INR (\d+)` | "₹25 per kg" → 25.0 |

---

## Dependencies

| Dependency | Injection | Fallback |
|------------|-----------|----------|
| `ListingService` | Constructor | Returns preview response without persistence |
| `AuroraPostgresClient` | Via ListingService | UUID returned, no DB write |
| `PricingAgent` / `PricePredictionAgent` | Via ListingService | Hard fallback ₹25/kg |
| `QualityAssessmentAgent` | Via ListingService | `hitl_required=True` |
| `ADCLAgent` | Via ListingService | `adcl_tagged=False` |

---

## ListingService Auto-Enrichment Pipeline

```
create_listing(request)
  ├── 1. Auto-price  ← PricingAgent.predict() → quintal→kg → fallback ₹25
  ├── 2. Expiry      ← SHELF_LIFE_DAYS[commodity] (7–90 days from now)
  ├── 3. QR Code     ← CF-{3-letter crop}-{UUID8} e.g. CF-TOM-A3B4C5D6
  ├── 4. ADCL Tag    ← ADCLAgent.get_weekly_demand() commodity match
  ├── 5. QA Trigger  ← QualityAssessmentAgent.assess() if photos given
  └── 6. Persist     ← AuroraPostgresClient.create_listing()
```

---

## Commodity Shelf-Life Calendar

| Commodity | Days | Commodity | Days |
|-----------|------|-----------|------|
| Tomato | 7 | Okra | 4 |
| Onion | 60 | Carrot | 21 |
| Potato | 90 | Cauliflower | 7 |
| Beans | 5 | Chilli | 14 |
| Default | 14 | | |

---

## REST API (via `src/api/routers/listings.py`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/listings` | Create (auto-enriched), returns 201 |
| `GET` | `/api/v1/listings` | Paginated search (commodity/grade/price/district/adcl filters) |
| `GET` | `/api/v1/listings/farmer/{id}` | All listings for a farmer |
| `GET` | `/api/v1/listings/{id}` | Single listing by UUID |
| `PATCH` | `/api/v1/listings/{id}` | Update price/qty/status |
| `DELETE` | `/api/v1/listings/{id}` | Soft-cancel (status=cancelled) |
| `POST` | `/api/v1/listings/{id}/grade` | Attach quality grade + HITL flag |

---

## HITL Grade Attachment Policy

| Condition | hitl_required |
|-----------|--------------|
| `cv_confidence >= 0.70` and grade ≠ A+ | `False` |
| `cv_confidence < 0.70` | `True` |
| `grade == "A+"` (always) | `True` |
| Quality agent unavailable + photos given | `True` |

---

## Constraints

- Voice agent routes listing creation via `CropListingAgent.process()`; requires `farmer_id` in context
- `search_listings` with `min_grade` filter applied in Python (post-DB-fetch) for portability
- Circular FK (listings ↔ digital_twins) resolved via `ALTER TABLE` in migration 002

---

## Test Coverage

`tests/unit/test_listing_service.py` — **50 tests**

Covers: shelf life calendar, create enrichment pipeline, auto-price fallback, ADCL tagging,
quality trigger, pagination, min_grade filter, get/update/cancel/farmer-listings,
grade HITL logic, expiry background job, NL agent intents, execute() interface,
voice AC6 (DB.create_listing called from voice query).
