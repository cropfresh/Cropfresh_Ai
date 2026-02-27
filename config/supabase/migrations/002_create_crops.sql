CREATE TABLE IF NOT EXISTS crops (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name_en VARCHAR(100) NOT NULL,
  name_kn VARCHAR(100),
  category VARCHAR(50),
  season VARCHAR(50)
);
