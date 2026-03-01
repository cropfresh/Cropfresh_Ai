# F002: Crop Listing Service

> **Priority:** P0 | **Status:** ✅ Implemented (Task 7, 2026-03-01)  
> **Sprint:** Sprint 05 — Core Agent Completion  
> **Files:** `src/agents/crop_listing/`, `src/api/services/listing_service.py`, `src/api/routers/listings.py`

---

## Overview

Full produce listing lifecycle management for the CropFresh marketplace. Farmers create
listings via voice commands or the REST API. The service auto-enriches every listing with
AI-suggested pricing, shelf-life expiry, batch QR codes, ADCL demand tags, and quality
assessment triggers.

---

## User Stories

| # | As a… | I want to… | So that… |
|---|--------|-----------|---------|
| 1 | Farmer | List my produce by saying "sell 200 kg tomatoes at ₹25" | I don't need to use an app form |
| 2 | Farmer | Get a fair price suggestion if I don't know the market rate | I don't undersell my crop |
| 3 | Buyer | Search listings by commodity, grade, and location | I find the best produce available |
| 4 | Platform | Automatically expire listings past their shelf life | The marketplace stays clean |
| 5 | Field Agent | Attach a quality grade to a listing | Buyers trust the grade accuracy |

---

## Acceptance Criteria

| # | Criterion | Status |
|---|-----------|--------|
| 1 | CRUD endpoints work with proper validation | ✅ 7 REST endpoints (POST/GET/PATCH/DELETE/grade) |
| 2 | Auto-price suggestion when no price given | ✅ PricingAgent.predict() → ₹/quintal → /kg, fallback ₹25 |
| 3 | Quality assessment triggered on photo upload | ✅ QualityAssessmentAgent.assess() called when photos list non-empty |
| 4 | Shelf-life-based auto-expiry | ✅ SHELF_LIFE_DAYS calendar: Tomato 7d, Onion 60d, Potato 90d, etc. |
| 5 | Search with commodity + location + grade filters | ✅ PaginatedListings with commodity/district/min_grade/max_price/adcl filters |
| 6 | Voice agent `create_listing` creates real DB record | ✅ CropListingAgent → ListingService → AuroraPostgresClient.create_listing() |

---

## API Reference

### Create Listing
```http
POST /api/v1/listings
Content-Type: application/json

{
  "farmer_id": "uuid",
  "commodity": "Tomato",
  "quantity_kg": 200.0,
  "asking_price_per_kg": 25.0,    // optional — auto-suggested if omitted
  "harvest_date": "2026-03-01",   // optional
  "photos": ["s3://bucket/p.jpg"] // optional — triggers quality assessment
}
```

**Response (201):**
```json
{
  "id": "uuid",
  "commodity": "Tomato",
  "asking_price_per_kg": 25.0,
  "suggested_price": 24.5,
  "grade": "Unverified",
  "hitl_required": false,
  "adcl_tagged": true,
  "batch_qr_code": "CF-TOM-A3B4C5D6",
  "expires_at": "2026-03-08T10:30:00",
  "status": "active"
}
```

### Search Listings
```http
GET /api/v1/listings?commodity=Tomato&min_grade=A&max_price=30&limit=20
```

### Attach Grade
```http
POST /api/v1/listings/{id}/grade
{ "grade": "A", "cv_confidence": 0.88, "defect_types": [] }
```

---

## Voice Interface

```
Farmer: "mujhe 200 kg tamatar bechna hai 25 rupaye mein"
Agent:  "Your listing has been created! 200kg Tomato at ₹25/kg (auto-suggested).
         Listing ID: CF-TOM-A3B4C5D6"
```

Supported commodity aliases: EN (tomato), HI (tamatar, pyaaz, aloo), KN (eerulli, alugedde, bendekai)

---

## Implementation Notes

- **Circular FK**: `listings.digital_twin_id` references `digital_twins`, while `digital_twins.listing_id`
  references `listings`. Resolved via `ALTER TABLE listings ADD CONSTRAINT` after both tables are created.
- **Min Grade Filter**: Applied in-Python after DB fetch (not SQL) for portability across DB backends.
- **HITL Policy**: `cv_confidence < 0.70` or `grade == "A+"` → always `hitl_required = True`.
