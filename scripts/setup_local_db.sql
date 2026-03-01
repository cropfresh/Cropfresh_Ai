-- ═══════════════════════════════════════════════════════════════
-- CropFresh AI — Local PostgreSQL Setup (No pgvector)
-- ═══════════════════════════════════════════════════════════════
-- Run this against your local PostgreSQL as the 'postgres' superuser:
--
--   psql -U postgres -f scripts/setup_local_db.sql
--
-- This creates:
--   1. 'cropfresh' database
--   2. 'cropfresh_app' user
--   3. Relational tables (users, chat_history, produce)
--
-- NOTE: pgvector extension is skipped on local dev (not available
-- without Visual Studio Build Tools). Vector search uses Qdrant
-- fallback locally. pgvector activates on AWS RDS automatically.
-- ═══════════════════════════════════════════════════════════════

-- Create the application user (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'cropfresh_app') THEN
        CREATE ROLE cropfresh_app WITH LOGIN PASSWORD 'cropfresh_dev_2026';
    END IF;
END
$$;

-- Create the database (if not exists)
SELECT 'CREATE DATABASE cropfresh OWNER cropfresh_app'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'cropfresh')
\gexec

-- Connect to the cropfresh database
\c cropfresh

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE cropfresh TO cropfresh_app;
GRANT ALL ON SCHEMA public TO cropfresh_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO cropfresh_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO cropfresh_app;

-- Try pgvector (works on AWS RDS, may fail locally)
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS vector;
    RAISE NOTICE 'pgvector extension enabled';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pgvector not available locally - using Qdrant fallback for vectors';
END
$$;

-- ─── Users ────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    phone TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    user_type TEXT DEFAULT 'farmer' CHECK (user_type IN ('farmer', 'buyer', 'admin')),
    location JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_users_phone ON users (phone);
CREATE INDEX IF NOT EXISTS idx_users_type ON users (user_type);

-- ─── Chat History ─────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS chat_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    agent_name TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_history (session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_history (user_id);

-- ─── Produce Listings ─────────────────────────────────────────

CREATE TABLE IF NOT EXISTS produce (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    farmer_id UUID REFERENCES users(id) ON DELETE CASCADE,
    crop_name TEXT NOT NULL,
    quantity_kg NUMERIC(10, 2) NOT NULL,
    price_per_kg NUMERIC(10, 2) NOT NULL,
    quality_grade TEXT DEFAULT 'B' CHECK (quality_grade IN ('A+', 'A', 'B', 'C')),
    location JSONB DEFAULT '{}',
    status TEXT DEFAULT 'available' CHECK (status IN ('available', 'sold', 'expired', 'withdrawn')),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_produce_crop ON produce (crop_name);
CREATE INDEX IF NOT EXISTS idx_produce_status ON produce (status);
CREATE INDEX IF NOT EXISTS idx_produce_farmer ON produce (farmer_id);

-- ─── Done ─────────────────────────────────────────────────────
\echo ''
\echo '✅ CropFresh local database setup complete!'
\echo '   Database: cropfresh'
\echo '   User:     cropfresh_app'
\echo '   Password: cropfresh_dev_2026'
\echo '   Tables:   users, chat_history, produce'
\echo ''
\echo 'Vector search: Using Qdrant (set VECTOR_DB_PROVIDER=qdrant in .env)'
\echo 'On AWS: pgvector will be auto-enabled (set VECTOR_DB_PROVIDER=pgvector)'
