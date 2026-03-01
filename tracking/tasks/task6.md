# Task 6: Extend Database Schema — Full Business Tables

> **Priority:** 🔴 P0 | **Phase:** 2 | **Effort:** 2–3 days  
> **Files:** `src/db/schema.sql`, `src/db/postgres_client.py`, `src/db/migrations/` [NEW]  
> **Score Target:** 9/10 — All 10+ business tables with indexes, constraints, and migration support

---

## 📌 Problem Statement

Current `schema.sql` has 4 tables (`agri_knowledge`, `users`, `chat_history`, `produce`). Business model requires 10+ tables as defined in `docs/architecture/database-schema.md`.

---

## 🏗️ Implementation Spec

### Tables to Add (per database-schema.md)
1. **`farmers`** — detailed profile, quality_score, GPS, aadhaar_hash, language_pref
2. **`buyers`** — type (retailer/restaurant/...), delivery GPS, credit, subscription tier
3. **`listings`** — full columns: grade, cv_confidence, hitl_required, digital_twin_id, batch_qr, expiry
4. **`orders`** — complete lifecycle: escrow, AISP breakdown, hauler assignment
5. **`haulers`** — vehicle type, capacity, GPS, utilization, tiering
6. **`digital_twins`** — farmer photos, agent photos, AI annotations, defect types
7. **`disputes`** — departure/arrival twin comparison, liability, claim resolution
8. **`agents`** (field agents) — districts, quality_score, dispute_rate
9. **`price_history`** — mandi price time series with source tracking
10. **`adcl_reports`** — weekly demand crop lists

### Migration Strategy
```
src/db/migrations/
├── 001_initial_schema.sql          # Current 4 tables (already applied)
├── 002_business_tables.sql         # New 10 tables (this task)
├── 003_indexes_and_constraints.sql # Geospatial + performance indexes
└── migration_runner.py             # Version-tracked migration tool
```

### Key Design Decisions
- **PostGIS**: Use `GEOGRAPHY(POINT)` for GPS columns (Haversine built-in)
- **JSONB**: For flexible metadata (AI annotations, diff reports)
- **UUID primary keys**: `gen_random_uuid()` for all tables
- **Soft deletes**: `is_active` flags instead of DELETE
- **Audit columns**: `created_at`, `updated_at` on every table
- **Check constraints**: Status enums enforced at DB level

### postgres_client.py Additions
```python
# CRUD methods for each new table:
async def create_farmer(self, farmer_data: dict) -> str
async def get_farmer(self, farmer_id: str) -> dict
async def create_listing(self, listing_data: dict) -> str
async def search_listings(self, filters: dict) -> list[dict]
async def create_order(self, order_data: dict) -> str  
async def update_order_status(self, order_id: str, status: str) -> bool
async def create_digital_twin(self, twin_data: dict) -> str
async def insert_price_history(self, records: list[dict]) -> int
async def get_price_history(self, commodity: str, district: str, days: int) -> list[dict]
```

---

## ✅ Acceptance Criteria

| # | Criterion | Weight |
|---|-----------|--------|
| 1 | All 10 tables created with correct columns + constraints | 30% |
| 2 | Geospatial indexes (GIST) on GPS columns | 15% |
| 3 | Migration runner tracks applied versions | 15% |
| 4 | CRUD methods in postgres_client.py for all tables | 20% |
| 5 | `psql -f schema.sql` runs without errors | 10% |
| 6 | Foreign keys and check constraints enforced | 10% |
