# ADR-002: Supabase as Primary Database

**Date**: 2026-02-01  |  **Status**: ⚠️ Superseded by [ADR-012](ADR-012-aurora-pgvector-consolidation.md) (2026-03-01)
> Supabase client retained for migration period. Production uses `AuroraPostgresClient` via `asyncpg`.

### Context
Need a primary relational database for users, listings, orders, and transactional data.

### Decision
Use Supabase (hosted PostgreSQL) as primary database.

### Consequences
- ✅ Free tier (500MB, 50K rows)
- ✅ Built-in auth, realtime subscriptions
- ✅ REST + Python SDK
- ✅ Edge functions for serverless logic
- ⚠️ Limited to PostgreSQL (fine for our use case)
