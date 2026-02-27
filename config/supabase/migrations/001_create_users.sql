CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  phone VARCHAR(15) UNIQUE NOT NULL,
  name VARCHAR(100),
  role VARCHAR(20) DEFAULT 'farmer',
  district VARCHAR(50),
  language VARCHAR(10) DEFAULT 'kn',
  created_at TIMESTAMPTZ DEFAULT now()
);
