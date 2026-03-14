"""
Aurora PostgreSQL Client — Backward compatibility redirect.

! This file is kept for backward compatibility. The actual implementation
! has been split into the `src.db.postgres` package.
! Import from `src.db.postgres` directly in new code.
"""

from src.db.postgres.client import AuroraPostgresClient, get_postgres
from src.db.postgres.models import ChatMessage

__all__ = ["AuroraPostgresClient", "ChatMessage", "get_postgres"]
