-- ═══════════════════════════════════════════════════════════════
-- * CropFresh AI — Business-Aligned Schema Migration
-- * Aligns database with ARCHITECTURE.md 5-agent model.
-- *
-- * Tables: agents, farmers, buyers, haulers, listings (v2),
-- *         orders (v2), digital_twins, disputes, price_history,
-- *         adcl_reports
-- *
-- * Run after: 001-006 (backwards compatible — does not drop old tables)
-- ═══════════════════════════════════════════════════════════════

-- ! PostGIS extension required for GEOGRAPHY columns.
-- ! Run `CREATE EXTENSION IF NOT EXISTS postgis;` if not already enabled.
CREATE EXTENSION IF NOT EXISTS postgis;

-- ─── Mitra Agents (field agents who onboard farmers) ─────────
CREATE TABLE IF NOT EXISTS agents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    phone           TEXT UNIQUE NOT NULL,
    district        TEXT NOT NULL,
    villages        TEXT[],
    quality_score   FLOAT DEFAULT 0.8,
    dispute_rate    FLOAT DEFAULT 0.05,
    total_listings  INT DEFAULT 0,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- ─── Farmers ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS farmers (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    phone           TEXT UNIQUE NOT NULL,
    aadhaar_hash    TEXT,
    location_gps    GEOGRAPHY(POINT),
    district        TEXT NOT NULL,
    village         TEXT,
    language_pref   TEXT DEFAULT 'kn',
    quality_score   FLOAT DEFAULT 0.5,
    grade_accuracy  FLOAT DEFAULT 0.5,
    onboarded_by    UUID REFERENCES agents(id),
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

-- ─── Buyers ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS buyers (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name                TEXT NOT NULL,
    phone               TEXT UNIQUE NOT NULL,
    type                TEXT CHECK (type IN (
        'retailer','restaurant','hotel','processor','distributor','online_grocer'
    )),
    district            TEXT NOT NULL,
    delivery_gps        GEOGRAPHY(POINT),
    credit_available    FLOAT DEFAULT 0,
    credit_limit        FLOAT DEFAULT 0,
    subscription_tier   TEXT DEFAULT 'free'
        CHECK (subscription_tier IN ('free','pro','enterprise')),
    total_orders        INT DEFAULT 0,
    churn_risk          FLOAT DEFAULT 0.5,
    created_at          TIMESTAMPTZ DEFAULT now()
);

-- ─── Haulers (logistics partners) ────────────────────────────
CREATE TABLE IF NOT EXISTS haulers (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    phone           TEXT UNIQUE NOT NULL,
    vehicle_type    TEXT CHECK (vehicle_type IN (
        '2w_ev','3w_auto','tempo','cold_chain'
    )),
    vehicle_number  TEXT,
    capacity_kg     FLOAT NOT NULL,
    current_gps     GEOGRAPHY(POINT),
    rating          FLOAT DEFAULT 3.5,
    tier            TEXT DEFAULT 'standard'
        CHECK (tier IN ('standard','silver','gold')),
    utilization_rate FLOAT DEFAULT 0.0,
    total_trips     INT DEFAULT 0,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- ─── Digital Twins (photo-grade proof for dispute resolution) ─
CREATE TABLE IF NOT EXISTS digital_twins (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    listing_id      UUID NOT NULL,
    farmer_photos   TEXT[],
    agent_photos    TEXT[],
    ai_annotations  JSONB,
    gps_location    GEOGRAPHY(POINT),
    agent_id        UUID REFERENCES agents(id),
    grade           TEXT,
    confidence      FLOAT,
    defect_types    TEXT[],
    shelf_life_days INT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- ─── Listings v2 (business-aligned) ─────────────────────────
-- NOTE: Keeps old `listings` table intact. New code should use `listings_v2`.
CREATE TABLE IF NOT EXISTS listings_v2 (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    farmer_id           UUID REFERENCES farmers(id) NOT NULL,
    commodity           TEXT NOT NULL,
    variety             TEXT,
    quantity_kg         FLOAT NOT NULL,
    asking_price_per_kg FLOAT NOT NULL,
    grade               TEXT CHECK (grade IN ('A','B','C','Unverified')),
    cv_confidence       FLOAT,
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
    adcl_tagged         BOOLEAN DEFAULT FALSE,
    created_at          TIMESTAMPTZ DEFAULT now(),
    expires_at          TIMESTAMPTZ
);

-- ─── Orders v2 (AISP-aligned) ───────────────────────────────
CREATE TABLE IF NOT EXISTS orders_v2 (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    listing_id      UUID REFERENCES listings_v2(id) NOT NULL,
    buyer_id        UUID REFERENCES buyers(id) NOT NULL,
    hauler_id       UUID REFERENCES haulers(id),
    quantity_kg     FLOAT NOT NULL,
    farmer_payout   FLOAT NOT NULL,
    logistics_cost  FLOAT NOT NULL,
    platform_margin FLOAT NOT NULL,
    risk_buffer     FLOAT NOT NULL,
    aisp_total      FLOAT NOT NULL,
    aisp_per_kg     FLOAT NOT NULL,
    escrow_status   TEXT DEFAULT 'pending'
        CHECK (escrow_status IN ('pending','held','released','refunded')),
    order_status    TEXT DEFAULT 'confirmed'
        CHECK (order_status IN (
            'confirmed','pickup_scheduled','in_transit',
            'delivered','disputed','settled'
        )),
    pickup_time     TIMESTAMPTZ,
    delivery_time   TIMESTAMPTZ,
    settled_at      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- ─── Disputes ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS disputes (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id            UUID REFERENCES orders_v2(id) NOT NULL,
    raised_by           TEXT CHECK (raised_by IN ('buyer','farmer','platform')),
    reason              TEXT NOT NULL,
    departure_twin_id   UUID REFERENCES digital_twins(id),
    arrival_photos      TEXT[],
    arrival_video_url   TEXT,
    diff_report         JSONB,
    liability           TEXT CHECK (liability IN (
        'farmer','hauler','buyer','platform','shared'
    )),
    claim_percent       FLOAT,
    status              TEXT DEFAULT 'open'
        CHECK (status IN ('open','ai_analysed','resolved','escalated')),
    resolution_amount   FLOAT,
    resolved_at         TIMESTAMPTZ,
    created_at          TIMESTAMPTZ DEFAULT now()
);

-- ─── Price History (scraped daily from Agmarknet/eNAM) ───────
CREATE TABLE IF NOT EXISTS price_history (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    commodity   TEXT NOT NULL,
    district    TEXT NOT NULL,
    state       TEXT DEFAULT 'Karnataka',
    date        DATE NOT NULL,
    modal_price FLOAT NOT NULL,
    min_price   FLOAT,
    max_price   FLOAT,
    source      TEXT DEFAULT 'agmarknet',
    created_at  TIMESTAMPTZ DEFAULT now(),
    UNIQUE(commodity, district, date, source)
);

-- ─── ADCL Weekly Demand Reports ──────────────────────────────
CREATE TABLE IF NOT EXISTS adcl_reports (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    week_start      DATE NOT NULL UNIQUE,
    crops           JSONB NOT NULL,
    generated_by    TEXT DEFAULT 'adcl_agent',
    created_at      TIMESTAMPTZ DEFAULT now()
);


-- ═══════════════════════════════════════════════════════════════
-- * INDEXES
-- ═══════════════════════════════════════════════════════════════

CREATE INDEX IF NOT EXISTS idx_listings_v2_pickup_gps
    ON listings_v2 USING GIST (pickup_gps);
CREATE INDEX IF NOT EXISTS idx_listings_v2_status
    ON listings_v2 (status);
CREATE INDEX IF NOT EXISTS idx_listings_v2_farmer
    ON listings_v2 (farmer_id);

CREATE INDEX IF NOT EXISTS idx_haulers_current_gps
    ON haulers USING GIST (current_gps);

CREATE INDEX IF NOT EXISTS idx_buyers_delivery_gps
    ON buyers USING GIST (delivery_gps);

CREATE INDEX IF NOT EXISTS idx_orders_v2_status
    ON orders_v2 (order_status);
CREATE INDEX IF NOT EXISTS idx_orders_v2_buyer
    ON orders_v2 (buyer_id);

CREATE INDEX IF NOT EXISTS idx_disputes_status
    ON disputes (status);

CREATE INDEX IF NOT EXISTS idx_price_history_lookup
    ON price_history (commodity, district, date);

CREATE INDEX IF NOT EXISTS idx_farmers_district
    ON farmers (district);
CREATE INDEX IF NOT EXISTS idx_farmers_gps
    ON farmers USING GIST (location_gps);
