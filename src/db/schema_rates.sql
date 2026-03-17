-- Multi-source rate hub storage

CREATE TABLE IF NOT EXISTS normalized_rates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rate_kind TEXT NOT NULL,
    commodity TEXT,
    variety TEXT,
    state TEXT DEFAULT 'Karnataka',
    district TEXT,
    market TEXT,
    location_label TEXT NOT NULL,
    price_date DATE NOT NULL,
    unit TEXT NOT NULL,
    currency TEXT DEFAULT 'INR',
    price_value NUMERIC(12, 2),
    min_price NUMERIC(12, 2),
    max_price NUMERIC(12, 2),
    modal_price NUMERIC(12, 2),
    source TEXT NOT NULL,
    authority_tier TEXT NOT NULL,
    source_url TEXT NOT NULL,
    freshness TEXT DEFAULT 'live',
    fetched_at TIMESTAMPTZ DEFAULT now(),
    raw_record_id UUID REFERENCES source_data(id),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_normalized_rates_query
    ON normalized_rates (rate_kind, commodity, location_label, price_date DESC);
CREATE INDEX IF NOT EXISTS idx_normalized_rates_source
    ON normalized_rates (source, price_date DESC);
