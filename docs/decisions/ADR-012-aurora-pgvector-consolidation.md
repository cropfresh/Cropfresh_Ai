# ADR-012: Aurora PostgreSQL + pgvector (Replacing Qdrant + Supabase)

**Date**: 2026-03-01  |  **Status**: Accepted  
**Supersedes**: ADR-001 (Qdrant), ADR-002 (Supabase)

### Context
CropFresh AI used three separate database services:
- **Qdrant Cloud** — vector search for RAG knowledge base
- **Supabase** — relational data (users, chat, produce) via REST API
- **Neo4j** — graph relationships (buyer-seller matching)

This created operational complexity (3 providers, 3 auth mechanisms, 3 billing accounts). Consolidating to AWS-managed services simplifies infrastructure and enables IAM-based auth.

### Decision
Replace Qdrant and Supabase with a **single Amazon RDS PostgreSQL** instance using the **pgvector** extension for vector search and standard SQL for relational data. Neo4j remains unchanged.

```
BEFORE                              AFTER
──────                              ─────
Qdrant Cloud  (vectors)     →   RDS PostgreSQL + pgvector  (one instance)
Supabase REST (relational)  →   Same RDS instance (asyncpg)
Neo4j Aura    (graph)       →   Neo4j (unchanged)
```

### Architecture
- **`src/db/postgres_client.py`** — unified `AuroraPostgresClient`
  - Vector: `vector_search()` using pgvector cosine similarity
  - Relational: CRUD for users, chat_history, produce via `asyncpg`
  - Auth: IAM token auth (production) or password (dev)
- **`src/db/schema.sql`** — full schema with `vector(1024)` column + IVFFlat index
- **Config**: `VECTOR_DB_PROVIDER` toggle (`pgvector` | `qdrant`)

### Consequences
- ✅ Single database = simpler ops, one connection pool
- ✅ Cost: ~$14/month (db.t3.micro) vs 3 free-tier services
- ✅ IAM auth — no API keys to rotate
- ✅ pgvector is native PostgreSQL — no vendor lock-in
- ✅ Consistent backup/restore across vectors + relational data
- ⚠️ pgvector less performant than dedicated Qdrant for >1M vectors
- ⚠️ Requires PostgreSQL knowledge vs Supabase REST simplicity

### Local Development
- PostgreSQL 17 locally for relational tables
- Qdrant via Docker for vector search (pgvector unavailable without VS Build Tools on Windows)
- Set `VECTOR_DB_PROVIDER=qdrant` in local `.env`

### Migration Path
1. Local PostgreSQL for relational tables (current) ✅
2. Upgrade AWS account → create RDS instance
3. Run `schema.sql` on RDS (pgvector pre-installed)
4. Set `VECTOR_DB_PROVIDER=pgvector` in production `.env`
5. Migrate Qdrant vectors → `agri_knowledge` table
