"""
Aurora PostgreSQL Client Package.

Re-exports the main client and models for backward compatibility.
"""

from src.db.postgres.client import AuroraPostgresClient, get_postgres
from src.db.postgres.models import ChatMessage

__all__ = ["AuroraPostgresClient", "ChatMessage", "get_postgres"]
