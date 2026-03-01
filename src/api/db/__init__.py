"""Database module for CropFresh AI."""
from src.db.supabase_client import get_supabase, SupabaseClient
from src.db.postgres_client import get_postgres, AuroraPostgresClient
from src.db.neo4j_client import get_neo4j, Neo4jClient

__all__ = [
    "get_supabase", "SupabaseClient",       # Legacy (Supabase REST)
    "get_postgres", "AuroraPostgresClient",  # New (Aurora pgvector)
    "get_neo4j", "Neo4jClient",              # Graph (kept as-is)
]
