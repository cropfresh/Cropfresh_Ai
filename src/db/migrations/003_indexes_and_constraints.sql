-- ═══════════════════════════════════════════════════════════════
-- Migration 003 — Geospatial + Performance Indexes
-- Requires: PostGIS extension (applied in 002)
-- Requires: All tables from migrations 001 and 002
-- ═══════════════════════════════════════════════════════════════

-- ─────────────────────────────────────────────────────────────
-- * Geospatial GIST indexes for Haversine distance queries
-- These enable fast "find nearest hauler/buyer within X km"
-- ─────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_listings_pickup_gps
    ON listings USING GIST (pickup_gps);

CREATE INDEX IF NOT EXISTS idx_haulers_current_gps
    ON haulers USING GIST (current_gps);

CREATE INDEX IF NOT EXISTS idx_buyers_delivery_gps
    ON buyers USING GIST (delivery_gps);

CREATE INDEX IF NOT EXISTS idx_farmers_location_gps
    ON farmers USING GIST (location_gps);

CREATE INDEX IF NOT EXISTS idx_digital_twins_gps
    ON digital_twins USING GIST (gps_location);

-- ─────────────────────────────────────────────────────────────
-- * Composite indexes for frequent query patterns
-- ─────────────────────────────────────────────────────────────

-- Buyer matching: find active listings by commodity in district
CREATE INDEX IF NOT EXISTS idx_listings_commodity_status
    ON listings (commodity, status)
    WHERE status = 'active';

-- Order lifecycle lookups by buyer + status
CREATE INDEX IF NOT EXISTS idx_orders_buyer_status
    ON orders (buyer_id, order_status);

-- Dispute resolution queue: open disputes first
CREATE INDEX IF NOT EXISTS idx_disputes_status_created
    ON disputes (status, created_at)
    WHERE status IN ('open', 'ai_analysed');

-- Price prediction: latest prices for commodity + district
CREATE INDEX IF NOT EXISTS idx_price_history_recent
    ON price_history (commodity, district, date DESC)
    INCLUDE (modal_price, min_price, max_price);

-- Field agent performance lookup
CREATE INDEX IF NOT EXISTS idx_field_agents_quality
    ON field_agents (quality_score DESC)
    WHERE is_active = TRUE;

-- ─────────────────────────────────────────────────────────────
-- * GIN indexes for JSONB metadata queries
-- ─────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_digital_twins_annotations
    ON digital_twins USING gin (ai_annotations);

CREATE INDEX IF NOT EXISTS idx_disputes_diff_report
    ON disputes USING gin (diff_report);

CREATE INDEX IF NOT EXISTS idx_adcl_reports_crops
    ON adcl_reports USING gin (crops);

-- ─────────────────────────────────────────────────────────────
-- * Partial indexes for hot query paths
-- ─────────────────────────────────────────────────────────────

-- Active haulers available for dispatch
CREATE INDEX IF NOT EXISTS idx_haulers_active_tier
    ON haulers (tier, rating DESC)
    WHERE is_active = TRUE;

-- Buyers with BNPL credit available
CREATE INDEX IF NOT EXISTS idx_buyers_credit
    ON buyers (credit_available DESC)
    WHERE credit_available > 0 AND is_active = TRUE;

-- Listings expiring within window (near-expiry prioritisation)
CREATE INDEX IF NOT EXISTS idx_listings_expires
    ON listings (expires_at)
    WHERE status = 'active' AND expires_at IS NOT NULL;

-- ─────────────────────────────────────────────────────────────
-- * Utility: updated_at auto-refresh trigger function
-- NOTE: Attach via trigger on each table that has updated_at
-- ─────────────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION trigger_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Attach trigger to every table with updated_at column
DO $$
DECLARE
    tbl TEXT;
BEGIN
    FOREACH tbl IN ARRAY ARRAY[
        'farmers', 'buyers', 'haulers', 'field_agents',
        'listings', 'orders', 'disputes'
    ]
    LOOP
        EXECUTE format(
            'CREATE TRIGGER trg_%s_updated_at
             BEFORE UPDATE ON %I
             FOR EACH ROW EXECUTE FUNCTION trigger_set_updated_at()',
            tbl, tbl
        );
    END LOOP;
END;
$$;
