CREATE TABLE IF NOT EXISTS prices (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  crop_id UUID REFERENCES crops(id),
  mandi VARCHAR(100),
  price_min DECIMAL,
  price_max DECIMAL,
  price_modal DECIMAL,
  date DATE,
  created_at TIMESTAMPTZ DEFAULT now()
);
