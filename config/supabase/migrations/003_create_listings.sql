CREATE TABLE IF NOT EXISTS listings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  farmer_id UUID REFERENCES users(id),
  crop_id UUID REFERENCES crops(id),
  quantity_kg DECIMAL,
  price_per_kg DECIMAL,
  grade VARCHAR(5),
  image_urls TEXT[],
  status VARCHAR(20) DEFAULT 'active',
  created_at TIMESTAMPTZ DEFAULT now()
);
