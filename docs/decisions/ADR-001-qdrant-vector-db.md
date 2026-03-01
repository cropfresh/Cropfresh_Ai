# ADR-001: Qdrant as Vector Database

**Date**: 2026-02-01
**Status**: ⚠️ Superseded by [ADR-012](ADR-012-aurora-pgvector-consolidation.md) (2026-03-01)
> Qdrant retained as dev fallback (`VECTOR_DB_PROVIDER=qdrant`). Production uses pgvector on RDS PostgreSQL.

### Context
CropFresh needs semantic search for crop listings, knowledge retrieval, and price history analysis. We need a vector database for embedding storage and similarity search.

### Decision
Use Qdrant Cloud as our vector database.

### Consequences
- ✅ Cloud-hosted (managed), no infra overhead
- ✅ Fast similarity search with filtering
- ✅ Python SDK with async support
- ✅ Free tier sufficient for MVP
- ⚠️ Vendor lock-in for vector storage
- ⚠️ Internet dependency for queries

### Alternatives Considered
- **Pinecone**: More expensive, less control
- **Weaviate**: Heavier, more complex setup
- **ChromaDB**: Good for local but limited cloud offering
- **pgvector**: Supabase addon but less performant for pure vector ops
