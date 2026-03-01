-- ═══════════════════════════════════════════════════════════════
-- Migration 002 — Business Tables
-- Adds all 10 production business tables required by the
-- CropFresh AI business model (PDF: India's Agri-Intelligence
-- Marketplace Built on Trust, Transparency & Logistics Precision)
-- ═══════════════════════════════════════════════════════════════

-- ! IMPORTANT: Requires PostGIS extension for GEOGRAPHY columns
-- ! Run: CREATE EXTENSION IF NOT EXISTS postgis;  before applying
CREATE EXTENSION IF NOT EXISTS postgis;

-- ─────────────────────────────────────────────────────────────
-- Table 1: agents (field agents — independent of AI agents)
-- NOTE: Created first because farmers and listings reference it
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS field_agents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    phone           TEXT UNIQUE NOT NULL,
    district        TEXT NOT NULL,
    villages        TEXT[] DEFAULT '{}',        -- Assigned village list
    quality_score   FLOAT DEFAULT 0.8           -- Grading accuracy ratio
        CHECK (quality_score >= 0.0 AND quality_score <= 1.0),
    dispute_rate    FLOAT DEFAULT 0.05
        CHECK (dispute_rate >= 0.0 AND dispute_rate <= 1.0),
    total_listings  INT DEFAULT 0,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_field_agents_district  ON field_agents (district);
CREATE INDEX IF NOT EXISTS idx_field_agents_is_active ON field_agents (is_active);

-- ─────────────────────────────────────────────────────────────
-- Table 2: haulers (logistics providers)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS haulers (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name             TEXT NOT NULL,
    phone            TEXT UNIQUE NOT NULL,
    vehicle_type     TEXT NOT NULL
        CHECK (vehicle_type IN ('2w_ev', '3w_auto', 'tempo', 'cold_chain')),
    vehicle_number   TEXT,
    capacity_kg      FLOAT NOT NULL CHECK (capacity_kg > 0),
    current_gps      GEOGRAPHY(POINT, 4326),
    rating           FLOAT DEFAULT 3.5
        CHECK (rating >= 1.0 AND rating <= 5.0),
    tier             TEXT DEFAULT 'standard'
        CHECK (tier IN ('standard', 'silver', 'gold')),
    utilization_rate FLOAT DEFAULT 0.0
        CHECK (utilization_rate >= 0.0 AND utilization_rate <= 100.0),
    total_trips      INT DEFAULT 0,
    is_active        BOOLEAN DEFAULT TRUE,
    created_at       TIMESTAMPTZ DEFAULT now(),
    updated_at       TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_haulers_is_active    ON haulers (is_active);
CREATE INDEX IF NOT EXISTS idx_haulers_vehicle_type ON haulers (vehicle_type);
CREATE INDEX IF NOT EXISTS idx_haulers_tier         ON haulers (tier);

-- ─────────────────────────────────────────────────────────────
-- Table 3: buyers (retailer / restaurant / distributor etc.)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS buyers (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name              TEXT NOT NULL,
    phone             TEXT UNIQUE NOT NULL,
    type              TEXT NOT NULL
        CHECK (type IN ('retailer', 'restaurant', 'hotel', 'processor', 'distributor', 'online_grocer')),
    district          TEXT NOT NULL,
    delivery_gps      GEOGRAPHY(POINT, 4326),
    credit_available  FLOAT DEFAULT 0.0 CHECK (credit_available >= 0),
    credit_limit      FLOAT DEFAULT 0.0 CHECK (credit_limit >= 0),
    subscription_tier TEXT DEFAULT 'free'
        CHECK (subscription_tier IN ('free', 'pro', 'enterprise')),
    total_orders      INT DEFAULT 0,
    churn_risk        FLOAT DEFAULT 0.5
        CHECK (churn_risk >= 0.0 AND churn_risk <= 1.0),
    is_active         BOOLEAN DEFAULT TRUE,
    created_at        TIMESTAMPTZ DEFAULT now(),
    updated_at        TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_buyers_district ON buyers (district);
CREATE INDEX IF NOT EXISTS idx_buyers_type     ON buyers (type);
CREATE INDEX IF NOT EXISTS idx_buyers_active   ON buyers (is_active);

-- ─────────────────────────────────────────────────────────────
-- Table 4: farmers (detailed profile with trust score)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS farmers (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    phone           TEXT UNIQUE NOT NULL,
    aadhaar_hash    TEXT,                       -- SHA-256 hash only, never plain
    location_gps    GEOGRAPHY(POINT, 4326),
    district        TEXT NOT NULL,
    village         TEXT,
    language_pref   TEXT DEFAULT 'kn'
        CHECK (language_pref IN ('kn', 'hi', 'en', 'te', 'ta', 'pa', 'mr')),
    quality_score   FLOAT DEFAULT 0.5
        CHECK (quality_score >= 0.0 AND quality_score <= 1.0),
    grade_accuracy  FLOAT DEFAULT 0.5
        CHECK (grade_accuracy >= 0.0 AND grade_accuracy <= 1.0),
    onboarded_by    UUID REFERENCES field_agents(id) ON DELETE SET NULL,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_farmers_district  ON farmers (district);
CREATE INDEX IF NOT EXISTS idx_farmers_phone     ON farmers (phone);
CREATE INDEX IF NOT EXISTS idx_farmers_is_active ON farmers (is_active);

-- ─────────────────────────────────────────────────────────────
-- Table 5: listings (produce available on marketplace)
-- NOTE: digital_twin_id FK is added after digital_twins table
--       is created (see ALTER TABLE below)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS listings (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    farmer_id             UUID REFERENCES farmers(id) NOT NULL,
    commodity             TEXT NOT NULL,
    variety               TEXT,
    quantity_kg           FLOAT NOT NULL CHECK (quantity_kg > 0),
    asking_price_per_kg   FLOAT NOT NULL CHECK (asking_price_per_kg >= 0),
    grade                 TEXT DEFAULT 'Unverified'
        CHECK (grade IN ('A+', 'A', 'B', 'C', 'Unverified')),
    cv_confidence         FLOAT CHECK (cv_confidence >= 0 AND cv_confidence <= 1),
    hitl_required         BOOLEAN DEFAULT FALSE,
    hitl_agent_id         UUID REFERENCES field_agents(id) ON DELETE SET NULL,
    -- * digital_twin_id FK constraint added after digital_twins table creation
    digital_twin_id       UUID,
    status                TEXT DEFAULT 'active'
        CHECK (status IN ('active', 'matched', 'fulfilled', 'expired', 'cancelled')),
    harvest_date          DATE,
    pickup_window_start   TIMESTAMPTZ,
    pickup_window_end     TIMESTAMPTZ,
    pickup_gps            GEOGRAPHY(POINT, 4326),
    batch_qr_code         TEXT UNIQUE,
    adcl_tagged           BOOLEAN DEFAULT FALSE,
    created_at            TIMESTAMPTZ DEFAULT now(),
    expires_at            TIMESTAMPTZ,
    updated_at            TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_listings_farmer    ON listings (farmer_id);
CREATE INDEX IF NOT EXISTS idx_listings_commodity ON listings (commodity);
CREATE INDEX IF NOT EXISTS idx_listings_status    ON listings (status);
CREATE INDEX IF NOT EXISTS idx_listings_adcl      ON listings (adcl_tagged) WHERE adcl_tagged = TRUE;

-- ─────────────────────────────────────────────────────────────
-- Table 6: digital_twins (photo + AI annotation records)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS digital_twins (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    listing_id      UUID REFERENCES listings(id) ON DELETE CASCADE NOT NULL,
    farmer_photos   TEXT[] DEFAULT '{}',        -- S3 URLs taken at farm
    agent_photos    TEXT[] DEFAULT '{}',        -- S3 URLs by field agent
    ai_annotations  JSONB DEFAULT '{}',         -- Grade overlay + defect boxes
    gps_location    GEOGRAPHY(POINT, 4326),
    agent_id        UUID REFERENCES field_agents(id) ON DELETE SET NULL,
    grade           TEXT CHECK (grade IN ('A+', 'A', 'B', 'C')),
    confidence      FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    defect_types    TEXT[] DEFAULT '{}',        -- e.g. ["bruise","worm_hole"]
    shelf_life_days INT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_digital_twins_listing ON digital_twins (listing_id);
CREATE INDEX IF NOT EXISTS idx_digital_twins_grade   ON digital_twins (grade);

-- * Add circular FK from listings → digital_twins now that digital_twins exists
ALTER TABLE listings
    ADD CONSTRAINT IF NOT EXISTS fk_listings_digital_twin
    FOREIGN KEY (digital_twin_id) REFERENCES digital_twins(id) ON DELETE SET NULL;

-- ─────────────────────────────────────────────────────────────
-- Table 7: orders (full lifecycle with escrow + AISP breakdown)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS orders (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    listing_id       UUID REFERENCES listings(id) NOT NULL,
    buyer_id         UUID REFERENCES buyers(id) NOT NULL,
    hauler_id        UUID REFERENCES haulers(id) ON DELETE SET NULL,
    quantity_kg      FLOAT NOT NULL CHECK (quantity_kg > 0),
    farmer_payout    FLOAT NOT NULL CHECK (farmer_payout >= 0),
    logistics_cost   FLOAT NOT NULL CHECK (logistics_cost >= 0),
    platform_margin  FLOAT NOT NULL CHECK (platform_margin >= 0),
    risk_buffer      FLOAT NOT NULL CHECK (risk_buffer >= 0),
    aisp_total       FLOAT NOT NULL CHECK (aisp_total >= 0),
    aisp_per_kg      FLOAT NOT NULL CHECK (aisp_per_kg >= 0),
    escrow_status    TEXT DEFAULT 'pending'
        CHECK (escrow_status IN ('pending', 'held', 'released', 'refunded')),
    order_status     TEXT DEFAULT 'confirmed'
        CHECK (order_status IN (
            'confirmed', 'pickup_scheduled', 'in_transit',
            'delivered', 'disputed', 'settled'
        )),
    pickup_time      TIMESTAMPTZ,
    delivery_time    TIMESTAMPTZ,
    settled_at       TIMESTAMPTZ,
    created_at       TIMESTAMPTZ DEFAULT now(),
    updated_at       TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_orders_listing      ON orders (listing_id);
CREATE INDEX IF NOT EXISTS idx_orders_buyer        ON orders (buyer_id);
CREATE INDEX IF NOT EXISTS idx_orders_hauler       ON orders (hauler_id);
CREATE INDEX IF NOT EXISTS idx_orders_status       ON orders (order_status);
CREATE INDEX IF NOT EXISTS idx_orders_escrow       ON orders (escrow_status);

-- ─────────────────────────────────────────────────────────────
-- Table 8: disputes (AI-diff driven quality dispute resolution)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS disputes (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id            UUID REFERENCES orders(id) NOT NULL,
    raised_by           TEXT NOT NULL
        CHECK (raised_by IN ('buyer', 'farmer', 'platform')),
    reason              TEXT NOT NULL,
    departure_twin_id   UUID REFERENCES digital_twins(id) ON DELETE SET NULL,
    arrival_photos      TEXT[] DEFAULT '{}',
    arrival_video_url   TEXT,
    diff_report         JSONB DEFAULT '{}',     -- AI diff engine output
    liability           TEXT
        CHECK (liability IN ('farmer', 'hauler', 'buyer', 'platform', 'shared')),
    claim_percent       FLOAT CHECK (claim_percent >= 0 AND claim_percent <= 100),
    status              TEXT DEFAULT 'open'
        CHECK (status IN ('open', 'ai_analysed', 'resolved', 'escalated')),
    resolution_amount   FLOAT,
    notes               TEXT DEFAULT '',
    resolved_at         TIMESTAMPTZ,
    created_at          TIMESTAMPTZ DEFAULT now(),
    updated_at          TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_disputes_order  ON disputes (order_id);
CREATE INDEX IF NOT EXISTS idx_disputes_status ON disputes (status);

-- ─────────────────────────────────────────────────────────────
-- Table 9: price_history (mandi price time series)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS price_history (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    commodity   TEXT NOT NULL,
    district    TEXT NOT NULL,
    state       TEXT DEFAULT 'Karnataka',
    date        DATE NOT NULL,
    modal_price FLOAT NOT NULL CHECK (modal_price >= 0),
    min_price   FLOAT CHECK (min_price >= 0),
    max_price   FLOAT CHECK (max_price >= 0),
    source      TEXT DEFAULT 'agmarknet',
    created_at  TIMESTAMPTZ DEFAULT now(),
    UNIQUE (commodity, district, date, source)
);

CREATE INDEX IF NOT EXISTS idx_price_history_lookup
    ON price_history (commodity, district, date DESC);
CREATE INDEX IF NOT EXISTS idx_price_history_date
    ON price_history (date DESC);

-- ─────────────────────────────────────────────────────────────
-- Table 10: adcl_reports (weekly ADCL demand crop lists)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS adcl_reports (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    week_start    DATE NOT NULL UNIQUE,
    crops         JSONB NOT NULL,           -- [{crop, predicted_price, demand_score, green_label}]
    generated_by  TEXT DEFAULT 'adcl_agent',
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_adcl_reports_week ON adcl_reports (week_start DESC);
