-- ─── Price Intelligence Pipeline ────────────────────────────────

CREATE TABLE IF NOT EXISTS source_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source TEXT NOT NULL,
    raw_data JSONB NOT NULL,
    url TEXT,
    scraped_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_source_data_source ON source_data (source);

CREATE TABLE IF NOT EXISTS normalized_prices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    commodity TEXT NOT NULL,
    variety TEXT,
    state TEXT,
    market TEXT NOT NULL,
    price_date DATE NOT NULL,
    min_price NUMERIC(10, 2),
    max_price NUMERIC(10, 2),
    modal_price NUMERIC(10, 2),
    unit TEXT,
    source TEXT NOT NULL,
    raw_record_id UUID REFERENCES source_data(id),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_normalized_prices_query 
    ON normalized_prices (commodity, market, price_date);
CREATE INDEX IF NOT EXISTS idx_normalized_prices_date 
    ON normalized_prices (price_date);
CREATE INDEX IF NOT EXISTS idx_normalized_prices_source 
    ON normalized_prices (source);
