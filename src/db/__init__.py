"""Database module for CropFresh AI."""
from src.db.neo4j_client import Neo4jClient, get_neo4j
from src.db.postgres_client import AuroraPostgresClient, get_postgres
from src.db.supabase_client import SupabaseClient, get_supabase

__all__ = [
    "get_supabase", "SupabaseClient",       # Legacy (Supabase REST)
    "get_postgres", "AuroraPostgresClient",  # New (Aurora pgvector)
    "get_neo4j", "Neo4jClient",              # Graph (kept as-is)
]
