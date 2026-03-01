"""
Aurora PostgreSQL Client
========================
Unified PostgreSQL client using asyncpg for Amazon Aurora.

Replaces both:
- SupabaseClient (REST API) → direct SQL
- Qdrant KnowledgeBase (vector) → pgvector extension

Authentication:
- IAM token: boto3.rds.generate_db_auth_token() (production)
- Password: direct connection string (development)
"""

import json
from functools import lru_cache
from typing import Any, Optional
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str  # 'user' or 'assistant'
    content: str
    agent_name: Optional[str] = None


class AuroraPostgresClient:
    """
    Amazon Aurora PostgreSQL client with pgvector support.

    Handles both relational operations (users, chat, produce) and
    vector search (agri_knowledge with pgvector).

    Usage:
        client = AuroraPostgresClient(host, database, ...)
        await client.connect()
        results = await client.vector_search(embedding, top_k=5)
    """

    def __init__(
        self,
        host: str = "localhost",
        database: str = "cropfresh",
        port: int = 5432,
        user: str = "cropfresh_app",
        password: str = "",
        region: str = "ap-south-1",
        use_iam_auth: bool = False,
        pool_min: int = 2,
        pool_max: int = 10,
    ):
        """
        Initialize Aurora PostgreSQL client.

        Args:
            host: Aurora endpoint (e.g. cropfresh.cluster-xxx.ap-south-1.rds.amazonaws.com)
            database: Database name
            port: PostgreSQL port
            user: Database user
            password: Password (ignored if use_iam_auth=True)
            region: AWS region for IAM auth
            use_iam_auth: Use IAM token instead of password
            pool_min: Minimum pool connections
            pool_max: Maximum pool connections
        """
        self.host = host
        self.database = database
        self.port = port
        self.user = user
        self.password = password
        self.region = region
        self.use_iam_auth = use_iam_auth
        self.pool_min = pool_min
        self.pool_max = pool_max
        self._pool = None

        logger.info(f"Initializing Aurora PostgreSQL client for {host}/{database}")

    def _get_iam_token(self) -> str:
        """Generate IAM authentication token for RDS."""
        import boto3

        client = boto3.client("rds", region_name=self.region)
        token = client.generate_db_auth_token(
            DBHostname=self.host,
            Port=self.port,
            DBUsername=self.user,
            Region=self.region,
        )
        return token

    async def connect(self):
        """Establish connection pool."""
        import asyncpg

        password = self._get_iam_token() if self.use_iam_auth else self.password

        ssl_context = "require" if self.use_iam_auth else None

        try:
            self._pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=password,
                min_size=self.pool_min,
                max_size=self.pool_max,
                ssl=ssl_context,
                command_timeout=30,
            )
            logger.info(f"Connected to Aurora PostgreSQL at {self.host}")
        except Exception as e:
            logger.error(f"Failed to connect to Aurora PostgreSQL: {e}")
            raise

    async def close(self):
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Aurora PostgreSQL connection pool closed")

    @property
    def pool(self):
        """Get connection pool (must call connect() first)."""
        if self._pool is None:
            raise RuntimeError(
                "Connection pool not initialized. Call await client.connect() first."
            )
        return self._pool

    # ═══════════════════════════════════════════════════════════════
    # Schema Setup
    # ═══════════════════════════════════════════════════════════════

    async def initialize_schema(self):
        """
        Create tables and pgvector extension if they don't exist.

        Safe to call multiple times (uses IF NOT EXISTS).
        """
        from pathlib import Path

        schema_path = Path(__file__).parent / "schema.sql"

        if schema_path.exists():
            schema_sql = schema_path.read_text(encoding="utf-8")
            async with self.pool.acquire() as conn:
                await conn.execute(schema_sql)
            logger.info("Database schema initialized from schema.sql")
        else:
            # Inline minimal schema if file not found
            async with self.pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS agri_knowledge (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        text TEXT NOT NULL,
                        source TEXT DEFAULT '',
                        category TEXT DEFAULT '',
                        metadata JSONB DEFAULT '{}',
                        embedding vector(1024),
                        created_at TIMESTAMPTZ DEFAULT now()
                    );
                """)
            logger.info("Database schema initialized (inline)")

    # ═══════════════════════════════════════════════════════════════
    # Vector Operations (replaces Qdrant)
    # ═══════════════════════════════════════════════════════════════

    async def add_vectors(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        sources: list[str] | None = None,
        categories: list[str] | None = None,
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> int:
        """
        Add document vectors to the knowledge base.

        Args:
            texts: Document texts
            embeddings: Pre-computed embedding vectors (1024-dim)
            sources: Source identifiers
            categories: Category labels
            metadatas: Additional metadata dicts
            ids: Optional document IDs (auto-generated if not provided)

        Returns:
            Number of documents added
        """
        n = len(texts)
        sources = sources or [""] * n
        categories = categories or [""] * n
        metadatas = metadatas or [{}] * n
        ids = ids or [str(uuid4()) for _ in range(n)]

        rows = [
            (ids[i], texts[i], sources[i], categories[i],
             json.dumps(metadatas[i]), str(embeddings[i]))
            for i in range(n)
        ]

        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO agri_knowledge (id, text, source, category, metadata, embedding)
                VALUES ($1::uuid, $2, $3, $4, $5::jsonb, $6::vector)
                ON CONFLICT (id) DO UPDATE SET
                    text = EXCLUDED.text,
                    source = EXCLUDED.source,
                    category = EXCLUDED.category,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                rows,
            )

        logger.info(f"Added {n} vectors to agri_knowledge")
        return n

    async def vector_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        category: str | None = None,
        score_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Search for similar documents using cosine distance.

        Args:
            query_embedding: Query vector (1024-dim)
            top_k: Number of results
            category: Optional category filter
            score_threshold: Minimum similarity (0-1, higher = more similar)

        Returns:
            List of dicts with id, text, source, category, metadata, score
        """
        embedding_str = str(query_embedding)

        if category:
            query = """
                SELECT id, text, source, category, metadata,
                       1 - (embedding <=> $1::vector) AS score
                FROM agri_knowledge
                WHERE category = $2
                  AND 1 - (embedding <=> $1::vector) >= $3
                ORDER BY embedding <=> $1::vector
                LIMIT $4
            """
            params = [embedding_str, category, score_threshold, top_k]
        else:
            query = """
                SELECT id, text, source, category, metadata,
                       1 - (embedding <=> $1::vector) AS score
                FROM agri_knowledge
                WHERE 1 - (embedding <=> $1::vector) >= $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
            """
            params = [embedding_str, score_threshold, top_k]

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        results = []
        for row in rows:
            results.append({
                "id": str(row["id"]),
                "text": row["text"],
                "source": row["source"],
                "category": row["category"],
                "metadata": json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
                "score": float(row["score"]),
            })

        return results

    async def delete_vectors(
        self,
        document_ids: list[str] | None = None,
        category: str | None = None,
    ) -> int:
        """
        Delete documents from knowledge base.

        Args:
            document_ids: Specific IDs to delete
            category: Delete all in category

        Returns:
            Number of documents deleted
        """
        async with self.pool.acquire() as conn:
            if document_ids:
                result = await conn.execute(
                    "DELETE FROM agri_knowledge WHERE id = ANY($1::uuid[])",
                    document_ids,
                )
            elif category:
                result = await conn.execute(
                    "DELETE FROM agri_knowledge WHERE category = $1",
                    category,
                )
            else:
                raise ValueError("Provide document_ids or category to delete")

        count = int(result.split()[-1])
        logger.info(f"Deleted {count} vectors from agri_knowledge")
        return count

    async def get_vector_stats(self) -> dict[str, Any]:
        """Get vector collection statistics."""
        async with self.pool.acquire() as conn:
            total = await conn.fetchval(
                "SELECT COUNT(*) FROM agri_knowledge"
            )
            categories = await conn.fetch(
                "SELECT category, COUNT(*) as cnt FROM agri_knowledge GROUP BY category"
            )
            has_vectors = await conn.fetchval(
                "SELECT COUNT(*) FROM agri_knowledge WHERE embedding IS NOT NULL"
            )

        return {
            "total_documents": total or 0,
            "documents_with_vectors": has_vectors or 0,
            "categories": {row["category"]: row["cnt"] for row in categories},
        }

    # ═══════════════════════════════════════════════════════════════
    # Chat History (replaces SupabaseClient.save_chat_message, etc.)
    # ═══════════════════════════════════════════════════════════════

    async def save_chat_message(
        self,
        user_id: str,
        session_id: str,
        message: ChatMessage,
    ) -> dict[str, Any]:
        """Save a chat message to history."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO chat_history (user_id, session_id, role, content, agent_name)
                VALUES ($1::uuid, $2, $3, $4, $5)
                RETURNING id, user_id, session_id, role, content, agent_name, created_at
                """,
                user_id, session_id, message.role, message.content, message.agent_name,
            )

        logger.debug(f"Saved chat message for session {session_id}")
        return dict(row) if row else {}

    async def get_chat_history(
        self,
        session_id: str,
        limit: int = 20,
    ) -> list[ChatMessage]:
        """Get chat history for a session."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT role, content, agent_name
                FROM chat_history
                WHERE session_id = $1
                ORDER BY created_at ASC
                LIMIT $2
                """,
                session_id, limit,
            )

        return [
            ChatMessage(
                role=row["role"],
                content=row["content"],
                agent_name=row.get("agent_name"),
            )
            for row in rows
        ]

    # ═══════════════════════════════════════════════════════════════
    # Users (replaces SupabaseClient.get_user, create_user)
    # ═══════════════════════════════════════════════════════════════

    async def get_user(self, user_id: str) -> Optional[dict[str, Any]]:
        """Get user by ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM users WHERE id = $1::uuid",
                user_id,
            )
        return dict(row) if row else None

    async def create_user(
        self,
        phone: str,
        name: str,
        user_type: str = "farmer",
        location: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Create a new user."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO users (phone, name, user_type, location)
                VALUES ($1, $2, $3, $4::jsonb)
                RETURNING *
                """,
                phone, name, user_type, json.dumps(location or {}),
            )

        logger.info(f"Created user: {name} ({user_type})")
        return dict(row) if row else {}

    # ═══════════════════════════════════════════════════════════════
    # Produce Listings (replaces SupabaseClient.list_produce, etc.)
    # ═══════════════════════════════════════════════════════════════

    async def list_produce(
        self,
        crop_name: Optional[str] = None,
        quality_grade: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List available produce with optional filters."""
        conditions = ["status = 'available'"]
        params: list[Any] = []
        idx = 1

        if crop_name:
            conditions.append(f"crop_name ILIKE ${idx}")
            params.append(f"%{crop_name}%")
            idx += 1

        if quality_grade:
            conditions.append(f"quality_grade = ${idx}")
            params.append(quality_grade)
            idx += 1

        where = " AND ".join(conditions)
        params.append(limit)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM produce WHERE {where} LIMIT ${idx}",
                *params,
            )

        return [dict(row) for row in rows]

    async def create_produce_listing(
        self,
        farmer_id: str,
        crop_name: str,
        quantity_kg: float,
        price_per_kg: float,
        quality_grade: str = "B",
        location: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Create a new produce listing."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO produce
                    (farmer_id, crop_name, quantity_kg, price_per_kg, quality_grade, location, status)
                VALUES ($1::uuid, $2, $3, $4, $5, $6::jsonb, 'available')
                RETURNING *
                """,
                farmer_id, crop_name, quantity_kg, price_per_kg,
                quality_grade, json.dumps(location or {}),
            )

        logger.info(f"Created listing: {quantity_kg}kg {crop_name}")
        return dict(row) if row else {}

    # ═══════════════════════════════════════════════════════════════
    # Health Check
    # ═══════════════════════════════════════════════════════════════

    async def health_check(self) -> bool:
        """Check if PostgreSQL connection is healthy."""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
            return result == 1
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            return False


# ═══════════════════════════════════════════════════════════════
# Singleton Factory
# ═══════════════════════════════════════════════════════════════

_client: Optional[AuroraPostgresClient] = None


async def get_postgres(
    host: str | None = None,
    database: str | None = None,
    **kwargs,
) -> AuroraPostgresClient:
    """
    Get or create shared AuroraPostgresClient instance.

    On first call, initializes connection pool and schema.
    """
    global _client

    if _client is None:
        if host is None:
            from src.config import get_settings
            settings = get_settings()
            host = settings.pg_host
            database = settings.pg_database
            kwargs.setdefault("port", settings.pg_port)
            kwargs.setdefault("user", settings.pg_user)
            kwargs.setdefault("password", settings.pg_password)
            kwargs.setdefault("region", settings.aws_region)
            kwargs.setdefault("use_iam_auth", settings.pg_use_iam_auth)

        _client = AuroraPostgresClient(
            host=host,
            database=database or "cropfresh",
            **kwargs,
        )
        await _client.connect()

    return _client
