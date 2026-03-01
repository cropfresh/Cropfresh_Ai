# Database Schema — CropFresh AI
> **Last Updated:** 2026-03-01
> **DB:** Amazon RDS PostgreSQL + pgvector (replaces Supabase — [ADR-012](../decisions/ADR-012-aurora-pgvector-consolidation.md))
> **Status:** Core tables implemented (`src/db/schema.sql`), business tables below are planned for Phase 4

> [!NOTE]
> **Current implementation** (`src/db/schema.sql`): `agri_knowledge` (pgvector), `users`, `chat_history`, `produce`
> **Business tables below** are the full target schema for production — will be migrated in Phase 4.

---

## Tables

### `farmers`
```sql
CREATE TABLE farmers (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    phone           TEXT UNIQUE NOT NULL,      -- OTP-verified
    aadhaar_hash    TEXT,                      -- Hashed, never plain
    location_gps    GEOGRAPHY(POINT),
    district        TEXT NOT NULL,
    village         TEXT,
    language_pref   TEXT DEFAULT 'kn',         -- kn/hi/en/te/ta/pa etc.
    quality_score   FLOAT DEFAULT 0.5,         -- 0.0–1.0 trust score
    grade_accuracy  FLOAT DEFAULT 0.5,         -- Ratio of self-grade vs agent grade
    onboarded_by    UUID REFERENCES agents(id),
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);
```

### `listings`
```sql
CREATE TABLE listings (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    farmer_id           UUID REFERENCES farmers(id) NOT NULL,
    commodity           TEXT NOT NULL,              -- "Tomato", "Beans"...
    variety             TEXT,                       -- "Hybrid", "Country"
    quantity_kg         FLOAT NOT NULL,
    asking_price_per_kg FLOAT NOT NULL,
    grade               TEXT CHECK (grade IN ('A','B','C','Unverified')),
    cv_confidence       FLOAT,                      -- AI grading confidence
    hitl_required       BOOLEAN DEFAULT FALSE,
    hitl_agent_id       UUID REFERENCES agents(id),
    digital_twin_id     UUID REFERENCES digital_twins(id),
    status              TEXT DEFAULT 'active'
        CHECK (status IN ('active','matched','fulfilled','expired','cancelled')),
    harvest_date        DATE,
    pickup_window_start TIMESTAMPTZ,
    pickup_window_end   TIMESTAMPTZ,
    pickup_gps          GEOGRAPHY(POINT),
    batch_qr_code       TEXT UNIQUE,
    adcl_tagged         BOOLEAN DEFAULT FALSE,      -- On weekly demand list
    created_at          TIMESTAMPTZ DEFAULT now(),
    expires_at          TIMESTAMPTZ
);
```

### `orders`
```sql
CREATE TABLE orders (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    listing_id      UUID REFERENCES listings(id) NOT NULL,
    buyer_id        UUID REFERENCES buyers(id) NOT NULL,
    hauler_id       UUID REFERENCES haulers(id),
    quantity_kg     FLOAT NOT NULL,
    farmer_payout   FLOAT NOT NULL,     -- What farmer gets (₹)
    logistics_cost  FLOAT NOT NULL,     -- Hauler fee (₹)
    platform_margin FLOAT NOT NULL,     -- CropFresh take (₹)
    risk_buffer     FLOAT NOT NULL,     -- 2% buffer (₹)
    aisp_total      FLOAT NOT NULL,     -- Total buyer pays (₹)
    aisp_per_kg     FLOAT NOT NULL,
    escrow_status   TEXT DEFAULT 'pending'
        CHECK (escrow_status IN ('pending','held','released','refunded')),
    order_status    TEXT DEFAULT 'confirmed'
        CHECK (order_status IN ('confirmed','pickup_scheduled','in_transit','delivered','disputed','settled')),
    pickup_time     TIMESTAMPTZ,
    delivery_time   TIMESTAMPTZ,
    settled_at      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now()
);
```

### `digital_twins`
```sql
CREATE TABLE digital_twins (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    listing_id      UUID REFERENCES listings(id) NOT NULL,
    farmer_photos   TEXT[],             -- S3 URLs — photos taken at farm
    agent_photos    TEXT[],             -- S3 URLs — agent verification photos
    ai_annotations  JSONB,             -- Grade overlay, defect bounding boxes
    gps_location    GEOGRAPHY(POINT),
    agent_id        UUID REFERENCES agents(id),
    grade           TEXT,
    confidence      FLOAT,
    defect_types    TEXT[],            -- ["bruise", "worm_hole", "colour_off"]
    shelf_life_days INT,               -- Predicted shelf life
    created_at      TIMESTAMPTZ DEFAULT now()
);
```

### `disputes`
```sql
CREATE TABLE disputes (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id            UUID REFERENCES orders(id) NOT NULL,
    raised_by           TEXT CHECK (raised_by IN ('buyer','farmer','platform')),
    reason              TEXT NOT NULL,
    departure_twin_id   UUID REFERENCES digital_twins(id),
    arrival_photos      TEXT[],             -- Photos submitted by buyer
    arrival_video_url   TEXT,
    diff_report         JSONB,              -- AI diff engine output
    liability           TEXT CHECK (liability IN ('farmer','hauler','buyer','platform','shared')),
    claim_percent       FLOAT,             -- 0–100%
    status              TEXT DEFAULT 'open'
        CHECK (status IN ('open','ai_analysed','resolved','escalated')),
    resolution_amount   FLOAT,
    resolved_at         TIMESTAMPTZ,
    created_at          TIMESTAMPTZ DEFAULT now()
);
```

### `haulers`
```sql
CREATE TABLE haulers (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    phone           TEXT UNIQUE NOT NULL,
    vehicle_type    TEXT CHECK (vehicle_type IN ('2w_ev','3w_auto','tempo','cold_chain')),
    vehicle_number  TEXT,
    capacity_kg     FLOAT NOT NULL,
    current_gps     GEOGRAPHY(POINT),
    rating          FLOAT DEFAULT 3.5,      -- 1.0–5.0
    tier            TEXT DEFAULT 'standard' CHECK (tier IN ('standard','silver','gold')),
    utilization_rate FLOAT DEFAULT 0.0,    -- % of vehicle capacity used per trip
    total_trips     INT DEFAULT 0,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT now()
);
```

### `buyers`
```sql
CREATE TABLE buyers (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name                TEXT NOT NULL,
    phone               TEXT UNIQUE NOT NULL,
    type                TEXT CHECK (type IN ('retailer','restaurant','hotel','processor','distributor','online_grocer')),
    district            TEXT NOT NULL,
    delivery_gps        GEOGRAPHY(POINT),
    credit_available    FLOAT DEFAULT 0,        -- BNPL credit in ₹
    credit_limit        FLOAT DEFAULT 0,
    subscription_tier   TEXT DEFAULT 'free' CHECK (tier IN ('free','pro','enterprise')),
    total_orders        INT DEFAULT 0,
    churn_risk          FLOAT DEFAULT 0.5,      -- ML-predicted churn probability
    created_at          TIMESTAMPTZ DEFAULT now()
);
```

### `agents`
```sql
CREATE TABLE agents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    phone           TEXT UNIQUE NOT NULL,
    district        TEXT NOT NULL,
    villages        TEXT[],             -- Assigned village list
    quality_score   FLOAT DEFAULT 0.8,  -- Grading accuracy
    dispute_rate    FLOAT DEFAULT 0.05, -- Dispute rate for their listings
    total_listings  INT DEFAULT 0,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT now()
);
```

### `price_history`
```sql
CREATE TABLE price_history (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    commodity   TEXT NOT NULL,
    district    TEXT NOT NULL,
    state       TEXT DEFAULT 'Karnataka',
    date        DATE NOT NULL,
    modal_price FLOAT NOT NULL,         -- ₹/quintal
    min_price   FLOAT,
    max_price   FLOAT,
    source      TEXT DEFAULT 'agmarknet',
    created_at  TIMESTAMPTZ DEFAULT now(),
    UNIQUE(commodity, district, date, source)
);
```

### `adcl_reports`
```sql
CREATE TABLE adcl_reports (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    week_start      DATE NOT NULL UNIQUE,
    crops           JSONB NOT NULL,     -- [{crop, predicted_price, demand_score, green_label}]
    generated_by    TEXT DEFAULT 'adcl_agent',
    created_at      TIMESTAMPTZ DEFAULT now()
);
```

---

## Key Indexes
```sql
-- Geospatial queries (GPS-based clustering)
CREATE INDEX idx_listings_pickup_gps ON listings USING GIST (pickup_gps);
CREATE INDEX idx_haulers_current_gps ON haulers USING GIST (current_gps);
CREATE INDEX idx_buyers_delivery_gps ON buyers USING GIST (delivery_gps);

-- Frequent lookups
CREATE INDEX idx_listings_status ON listings (status);
CREATE INDEX idx_orders_order_status ON orders (order_status);
CREATE INDEX idx_price_history_lookup ON price_history (commodity, district, date);
CREATE INDEX idx_disputes_status ON disputes (status);
```

---

## Relationships
```
farmers ──< listings ──< orders >── buyers
listings ──< digital_twins
orders >── haulers
orders ──< disputes >── digital_twins
agents >─ listings (onboarded, hitl_agent)
adcl_reports → (used by listings.adcl_tagged)
price_history (scraped daily by APMC scraper)
```
