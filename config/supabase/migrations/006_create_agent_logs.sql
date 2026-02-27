CREATE TABLE IF NOT EXISTS agent_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_type VARCHAR(50),
  input JSONB,
  output JSONB,
  latency_ms INTEGER,
  token_usage JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);
