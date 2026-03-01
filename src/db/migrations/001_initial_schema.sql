-- ═══════════════════════════════════════════════════════════════
-- Migration 001 — Initial Schema
-- Applied: First deployment (matches src/db/schema.sql)
-- Tables: agri_knowledge, users, chat_history, produce
-- ═══════════════════════════════════════════════════════════════

CREATE EXTENSION IF NOT EXISTS vector;

-- * Vector knowledge base for RAG (BGE-M3 1024-dim embeddings)
CREATE TABLE IF NOT EXISTS agri_knowledge (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text        TEXT NOT NULL,
    source      TEXT DEFAULT '',
    category    TEXT DEFAULT '',
    metadata    JSONB DEFAULT '{}',
    embedding   vector(1024),
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_agri_knowledge_embedding
    ON agri_knowledge USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_agri_knowledge_category ON agri_knowledge (category);
CREATE INDEX IF NOT EXISTS idx_agri_knowledge_source   ON agri_knowledge (source);
CREATE INDEX IF NOT EXISTS idx_agri_knowledge_metadata ON agri_knowledge USING gin (metadata);

-- * Core user accounts (farmer / buyer / admin)
CREATE TABLE IF NOT EXISTS users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    phone       TEXT UNIQUE NOT NULL,
    name        TEXT NOT NULL,
    user_type   TEXT DEFAULT 'farmer' CHECK (user_type IN ('farmer', 'buyer', 'admin')),
    location    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_users_phone ON users (phone);
CREATE INDEX IF NOT EXISTS idx_users_type  ON users (user_type);

-- * Conversation history per session
CREATE TABLE IF NOT EXISTS chat_history (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content     TEXT NOT NULL,
    agent_name  TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_history (session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_chat_user    ON chat_history (user_id);

-- * Legacy produce listings (superseded by listings table in migration 002)
CREATE TABLE IF NOT EXISTS produce (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    farmer_id     UUID REFERENCES users(id) ON DELETE CASCADE,
    crop_name     TEXT NOT NULL,
    quantity_kg   NUMERIC(10, 2) NOT NULL,
    price_per_kg  NUMERIC(10, 2) NOT NULL,
    quality_grade TEXT DEFAULT 'B' CHECK (quality_grade IN ('A+', 'A', 'B', 'C')),
    location      JSONB DEFAULT '{}',
    status        TEXT DEFAULT 'available'
        CHECK (status IN ('available', 'sold', 'expired', 'withdrawn')),
    created_at    TIMESTAMPTZ DEFAULT now(),
    updated_at    TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_produce_crop   ON produce (crop_name);
CREATE INDEX IF NOT EXISTS idx_produce_status ON produce (status);
CREATE INDEX IF NOT EXISTS idx_produce_farmer ON produce (farmer_id);
