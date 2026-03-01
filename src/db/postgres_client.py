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
    # Farmers CRUD
    # ═══════════════════════════════════════════════════════════════

    async def create_farmer(self, farmer_data: dict[str, Any]) -> str:
        """
        Insert a new farmer record.

        Args:
            farmer_data: Dict with keys: name, phone, district, and optional
                         village, language_pref, aadhaar_hash, onboarded_by.

        Returns:
            New farmer UUID as string.
        """
        async with self.pool.acquire() as conn:
            farmer_id = await conn.fetchval(
                """
                INSERT INTO farmers
                    (name, phone, district, village, language_pref,
                     aadhaar_hash, onboarded_by)
                VALUES ($1, $2, $3, $4, $5, $6, $7::uuid)
                RETURNING id
                """,
                farmer_data["name"],
                farmer_data["phone"],
                farmer_data["district"],
                farmer_data.get("village"),
                farmer_data.get("language_pref", "kn"),
                farmer_data.get("aadhaar_hash"),
                farmer_data.get("onboarded_by"),
            )
        logger.info(f"Created farmer: {farmer_data['name']} ({farmer_data['phone']})")
        return str(farmer_id)

    async def get_farmer(self, farmer_id: str) -> Optional[dict[str, Any]]:
        """
        Fetch a farmer record by UUID.

        Args:
            farmer_id: Farmer UUID string.

        Returns:
            Dict of farmer fields, or None if not found.
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM farmers WHERE id = $1::uuid AND is_active = TRUE",
                farmer_id,
            )
        return dict(row) if row else None

    # ═══════════════════════════════════════════════════════════════
    # Listings CRUD
    # ═══════════════════════════════════════════════════════════════

    async def get_listing(self, listing_id: str) -> Optional[dict[str, Any]]:
        """
        Fetch a single listing by UUID, joined with farmer details.

        Args:
            listing_id: Listing UUID string.

        Returns:
            Dict of listing fields (with farmer_name, farmer_district),
            or None if not found.
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT l.*, f.name AS farmer_name, f.district AS farmer_district
                FROM listings l
                JOIN farmers f ON f.id = l.farmer_id
                WHERE l.id = $1::uuid
                """,
                listing_id,
            )
        return dict(row) if row else None

    async def create_listing(self, listing_data: dict[str, Any]) -> str:
        """
        Create a new produce listing.

        Args:
            listing_data: Dict with keys: farmer_id, commodity, quantity_kg,
                          asking_price_per_kg, and optional grade, variety,
                          harvest_date, pickup_window_start, pickup_window_end,
                          batch_qr_code, expires_at.

        Returns:
            New listing UUID as string.
        """
        async with self.pool.acquire() as conn:
            listing_id = await conn.fetchval(
                """
                INSERT INTO listings
                    (farmer_id, commodity, variety, quantity_kg,
                     asking_price_per_kg, grade, harvest_date,
                     pickup_window_start, pickup_window_end,
                     batch_qr_code, expires_at)
                VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING id
                """,
                listing_data["farmer_id"],
                listing_data["commodity"],
                listing_data.get("variety"),
                float(listing_data["quantity_kg"]),
                float(listing_data["asking_price_per_kg"]),
                listing_data.get("grade", "Unverified"),
                listing_data.get("harvest_date"),
                listing_data.get("pickup_window_start"),
                listing_data.get("pickup_window_end"),
                listing_data.get("batch_qr_code"),
                listing_data.get("expires_at"),
            )
        logger.info(
            f"Created listing: {listing_data['quantity_kg']}kg "
            f"{listing_data['commodity']} by farmer {listing_data['farmer_id']}"
        )
        return str(listing_id)

    async def search_listings(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Search active listings with optional filters.

        Args:
            filters: Dict with optional keys: commodity, grade, district,
                     min_qty_kg, max_price_per_kg, limit (default 50).

        Returns:
            List of matching listing dicts.
        """
        conditions = ["l.status = 'active'"]
        params: list[Any] = []
        idx = 1

        if commodity := filters.get("commodity"):
            conditions.append(f"l.commodity ILIKE ${idx}")
            params.append(f"%{commodity}%")
            idx += 1

        if grade := filters.get("grade"):
            conditions.append(f"l.grade = ${idx}")
            params.append(grade)
            idx += 1

        if min_qty := filters.get("min_qty_kg"):
            conditions.append(f"l.quantity_kg >= ${idx}")
            params.append(float(min_qty))
            idx += 1

        if max_price := filters.get("max_price_per_kg"):
            conditions.append(f"l.asking_price_per_kg <= ${idx}")
            params.append(float(max_price))
            idx += 1

        limit = filters.get("limit", 50)
        params.append(limit)

        where = " AND ".join(conditions)
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT l.*, f.name AS farmer_name, f.district AS farmer_district
                FROM listings l
                JOIN farmers f ON f.id = l.farmer_id
                WHERE {where}
                ORDER BY l.created_at DESC
                LIMIT ${idx}
                """,
                *params,
            )
        return [dict(row) for row in rows]

    # ═══════════════════════════════════════════════════════════════
    # Orders CRUD
    # ═══════════════════════════════════════════════════════════════

    async def create_order(self, order_data: dict[str, Any]) -> str:
        """
        Create a new order with escrow pending.

        Args:
            order_data: Dict with keys: listing_id, buyer_id, quantity_kg,
                        farmer_payout, logistics_cost, platform_margin,
                        risk_buffer, aisp_total, aisp_per_kg.
                        Optional: hauler_id.

        Returns:
            New order UUID as string.
        """
        async with self.pool.acquire() as conn:
            order_id = await conn.fetchval(
                """
                INSERT INTO orders
                    (listing_id, buyer_id, hauler_id, quantity_kg,
                     farmer_payout, logistics_cost, platform_margin,
                     risk_buffer, aisp_total, aisp_per_kg)
                VALUES ($1::uuid, $2::uuid, $3::uuid, $4, $5, $6, $7, $8, $9, $10)
                RETURNING id
                """,
                order_data["listing_id"],
                order_data["buyer_id"],
                order_data.get("hauler_id"),
                float(order_data["quantity_kg"]),
                float(order_data["farmer_payout"]),
                float(order_data["logistics_cost"]),
                float(order_data["platform_margin"]),
                float(order_data["risk_buffer"]),
                float(order_data["aisp_total"]),
                float(order_data["aisp_per_kg"]),
            )
        logger.info(
            f"Created order {order_id}: "
            f"{order_data['quantity_kg']}kg, ₹{order_data['aisp_total']}"
        )
        return str(order_id)

    async def update_order_status(
        self,
        order_id: str,
        status: str,
        escrow_status: Optional[str] = None,
    ) -> bool:
        """
        Update order lifecycle status.

        Args:
            order_id: Order UUID.
            status: New order_status value.
            escrow_status: Optional new escrow_status value.

        Returns:
            True if the row was updated, False if order not found.
        """
        if escrow_status:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    """
                    UPDATE orders
                    SET order_status = $1, escrow_status = $2
                    WHERE id = $3::uuid
                    """,
                    status, escrow_status, order_id,
                )
        else:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    "UPDATE orders SET order_status = $1 WHERE id = $2::uuid",
                    status, order_id,
                )

        updated = int(result.split()[-1]) > 0
        if updated:
            logger.info(f"Order {order_id} status → {status}")
        return updated

    # ═══════════════════════════════════════════════════════════════
    # Digital Twins CRUD
    # ═══════════════════════════════════════════════════════════════

    async def create_digital_twin(self, twin_data: dict[str, Any]) -> str:
        """
        Create a digital twin record for a listing.

        Args:
            twin_data: Dict with keys: listing_id, and optional farmer_photos,
                       agent_photos, ai_annotations, agent_id, grade,
                       confidence, defect_types, shelf_life_days.

        Returns:
            New digital_twin UUID as string.
        """
        async with self.pool.acquire() as conn:
            twin_id = await conn.fetchval(
                """
                INSERT INTO digital_twins
                    (listing_id, farmer_photos, agent_photos,
                     ai_annotations, agent_id, grade,
                     confidence, defect_types, shelf_life_days)
                VALUES ($1::uuid, $2, $3, $4::jsonb, $5::uuid, $6, $7, $8, $9)
                RETURNING id
                """,
                twin_data["listing_id"],
                twin_data.get("farmer_photos", []),
                twin_data.get("agent_photos", []),
                json.dumps(twin_data.get("ai_annotations", {})),
                twin_data.get("agent_id"),
                twin_data.get("grade"),
                twin_data.get("confidence"),
                twin_data.get("defect_types", []),
                twin_data.get("shelf_life_days"),
            )
        logger.info(f"Created digital twin {twin_id} for listing {twin_data['listing_id']}")
        return str(twin_id)

    async def get_digital_twin(self, twin_id: str) -> Optional[dict[str, Any]]:
        """
        Fetch a digital twin record by UUID.

        Args:
            twin_id: Digital twin UUID string.

        Returns:
            Row dict with all digital_twins columns, or None if not found.
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, listing_id, farmer_photos, agent_photos,
                       ai_annotations, agent_id, grade, confidence,
                       defect_types, shelf_life_days, created_at
                FROM digital_twins
                WHERE id = $1::uuid
                """,
                twin_id,
            )
        if row is None:
            return None
        return dict(row)

    async def update_dispute_diff_report(
        self,
        dispute_id: str,
        diff_report: dict[str, Any],
        liability: Optional[str] = None,
        claim_percent: Optional[float] = None,
    ) -> bool:
        """
        Update a dispute record with the AI diff report results.

        Args:
            dispute_id:   Dispute UUID string.
            diff_report:  DiffReport.to_dict() payload.
            liability:    Liable party ('farmer' | 'hauler' | 'buyer' | 'shared').
            claim_percent: Recommended claim percentage [0, 100].

        Returns:
            True if a row was updated, False if dispute not found.
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE disputes
                SET diff_report   = $2::jsonb,
                    liability     = COALESCE($3, liability),
                    claim_percent = COALESCE($4, claim_percent),
                    status        = 'ai_analysed',
                    updated_at    = NOW()
                WHERE id = $1::uuid
                """,
                dispute_id,
                json.dumps(diff_report),
                liability,
                claim_percent,
            )
        updated = result.split()[-1] != "0"
        if updated:
            logger.info(f"Dispute {dispute_id} diff report saved (liability={liability})")
        return updated

    # ═══════════════════════════════════════════════════════════════
    # Price History CRUD
    # ═══════════════════════════════════════════════════════════════

    async def insert_price_history(self, records: list[dict[str, Any]]) -> int:
        """
        Bulk insert market price records (upsert on conflict).

        Args:
            records: List of dicts with keys: commodity, district, date,
                     modal_price, and optional min_price, max_price, source, state.

        Returns:
            Number of rows inserted/updated.
        """
        rows = [
            (
                rec["commodity"],
                rec["district"],
                rec.get("state", "Karnataka"),
                rec["date"],
                float(rec["modal_price"]),
                float(rec["min_price"]) if rec.get("min_price") is not None else None,
                float(rec["max_price"]) if rec.get("max_price") is not None else None,
                rec.get("source", "agmarknet"),
            )
            for rec in records
        ]

        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO price_history
                    (commodity, district, state, date, modal_price,
                     min_price, max_price, source)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (commodity, district, date, source)
                DO UPDATE SET
                    modal_price = EXCLUDED.modal_price,
                    min_price   = EXCLUDED.min_price,
                    max_price   = EXCLUDED.max_price
                """,
                rows,
            )
        logger.info(f"Upserted {len(rows)} price history records")
        return len(rows)

    async def get_price_history(
        self,
        commodity: str,
        district: str,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """
        Fetch recent market prices for a commodity + district.

        Args:
            commodity: Crop name (case-insensitive match).
            district: Karnataka district name.
            days: Number of calendar days to look back (default 30).

        Returns:
            List of price records sorted newest-first.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT commodity, district, state, date,
                       modal_price, min_price, max_price, source
                FROM price_history
                WHERE commodity ILIKE $1
                  AND district ILIKE $2
                  AND date >= CURRENT_DATE - ($3 * INTERVAL '1 day')
                ORDER BY date DESC
                """,
                f"%{commodity}%",
                f"%{district}%",
                days,
            )
        return [dict(row) for row in rows]

    # ═══════════════════════════════════════════════════════════════
    # Buyers / Haulers / Disputes CRUD
    # ═══════════════════════════════════════════════════════════════

    async def create_buyer(self, buyer_data: dict[str, Any]) -> str:
        """
        Create a new buyer record.

        Args:
            buyer_data: Dict with keys: name, phone, type, district,
                        and optional credit_limit, subscription_tier.

        Returns:
            New buyer UUID as string.
        """
        async with self.pool.acquire() as conn:
            buyer_id = await conn.fetchval(
                """
                INSERT INTO buyers (name, phone, type, district, credit_limit, subscription_tier)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
                """,
                buyer_data["name"],
                buyer_data["phone"],
                buyer_data["type"],
                buyer_data["district"],
                float(buyer_data.get("credit_limit", 0.0)),
                buyer_data.get("subscription_tier", "free"),
            )
        logger.info(f"Created buyer: {buyer_data['name']} ({buyer_data['type']})")
        return str(buyer_id)

    async def get_buyer(self, buyer_id: str) -> Optional[dict[str, Any]]:
        """
        Fetch a single buyer record by UUID.

        Args:
            buyer_id: Buyer UUID string.

        Returns:
            Dict of buyer fields, or None if not found.
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM buyers WHERE id = $1::uuid AND is_active = TRUE",
                buyer_id,
            )
        return dict(row) if row else None

    async def get_farmer_by_phone(self, phone: str) -> Optional[dict[str, Any]]:
        """
        Fetch a farmer record by phone number.

        Args:
            phone: Normalised mobile number (e.g. +919876543210).

        Returns:
            Dict of farmer fields, or None if not found.
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM farmers WHERE phone = $1 AND is_active = TRUE LIMIT 1",
                phone,
            )
        return dict(row) if row else None

    async def get_buyer_by_phone(self, phone: str) -> Optional[dict[str, Any]]:
        """
        Fetch a buyer record by phone number.

        Args:
            phone: Normalised mobile number (e.g. +919876543210).

        Returns:
            Dict of buyer fields, or None if not found.
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM buyers WHERE phone = $1 AND is_active = TRUE LIMIT 1",
                phone,
            )
        return dict(row) if row else None

    async def update_farmer(
        self,
        farmer_id: str,
        updates: dict[str, Any],
    ) -> bool:
        """
        Update a farmer's profile fields.

        Args:
            farmer_id: Farmer UUID string.
            updates: Dict with any subset of: name, district, village,
                     language_pref, aadhaar_hash, quality_score.

        Returns:
            True if the row was updated, False if not found.
        """
        allowed = {
            "name", "district", "village", "language_pref",
            "aadhaar_hash", "quality_score",
        }
        fields = {k: v for k, v in updates.items() if k in allowed}
        if not fields:
            return False

        set_clauses = [f"{col} = ${i}" for i, col in enumerate(fields.keys(), start=1)]
        params: list[Any] = list(fields.values())
        params.append(farmer_id)

        async with self.pool.acquire() as conn:
            result = await conn.execute(
                f"UPDATE farmers SET {', '.join(set_clauses)} WHERE id = ${len(params)}::uuid",
                *params,
            )
        updated = int(result.split()[-1]) > 0
        if updated:
            logger.info(f"Farmer {farmer_id} updated: {list(fields.keys())}")
        return updated

    async def update_buyer(
        self,
        buyer_id: str,
        updates: dict[str, Any],
    ) -> bool:
        """
        Update a buyer's profile fields.

        Args:
            buyer_id: Buyer UUID string.
            updates: Dict with any subset of: name, district, type,
                     credit_limit, subscription_tier.

        Returns:
            True if the row was updated, False if not found.
        """
        allowed = {"name", "district", "type", "credit_limit", "subscription_tier"}
        fields = {k: v for k, v in updates.items() if k in allowed}
        if not fields:
            return False

        set_clauses = [f"{col} = ${i}" for i, col in enumerate(fields.keys(), start=1)]
        params: list[Any] = list(fields.values())
        params.append(buyer_id)

        async with self.pool.acquire() as conn:
            result = await conn.execute(
                f"UPDATE buyers SET {', '.join(set_clauses)} WHERE id = ${len(params)}::uuid",
                *params,
            )
        updated = int(result.split()[-1]) > 0
        if updated:
            logger.info(f"Buyer {buyer_id} updated: {list(fields.keys())}")
        return updated

    async def create_dispute(self, dispute_data: dict[str, Any]) -> str:
        """
        Open a new dispute for an order.

        Args:
            dispute_data: Dict with keys: order_id, raised_by, reason,
                          and optional departure_twin_id, arrival_photos.

        Returns:
            New dispute UUID as string.
        """
        async with self.pool.acquire() as conn:
            dispute_id = await conn.fetchval(
                """
                INSERT INTO disputes
                    (order_id, raised_by, reason,
                     departure_twin_id, arrival_photos)
                VALUES ($1::uuid, $2, $3, $4::uuid, $5)
                RETURNING id
                """,
                dispute_data["order_id"],
                dispute_data["raised_by"],
                dispute_data["reason"],
                dispute_data.get("departure_twin_id"),
                dispute_data.get("arrival_photos", []),
            )
        logger.info(f"Opened dispute {dispute_id} for order {dispute_data['order_id']}")
        return str(dispute_id)

    async def get_order(self, order_id: str) -> Optional[dict[str, Any]]:
        """
        Fetch a single order with listing and buyer details.

        Args:
            order_id: Order UUID string.

        Returns:
            Dict of order fields joined with listing commodity and buyer name,
            or None if not found.
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT o.*,
                       l.commodity, l.farmer_id,
                       b.name AS buyer_name
                FROM orders o
                JOIN listings l ON l.id = o.listing_id
                JOIN buyers   b ON b.id = o.buyer_id
                WHERE o.id = $1::uuid
                """,
                order_id,
            )
        return dict(row) if row else None

    async def get_orders_by_farmer(
        self,
        farmer_id: str,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Fetch all orders belonging to a farmer's listings.

        Args:
            farmer_id: Farmer UUID string.
            status: Optional order_status filter.
            limit: Maximum rows to return.

        Returns:
            List of order dicts ordered newest-first.
        """
        params: list[Any] = [farmer_id, limit]
        status_clause = ""
        if status:
            status_clause = "AND o.order_status = $3"
            params = [farmer_id, limit, status]

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT o.*, l.commodity
                FROM orders o
                JOIN listings l ON l.id = o.listing_id
                WHERE l.farmer_id = $1::uuid
                  {status_clause}
                ORDER BY o.created_at DESC
                LIMIT $2
                """,
                *params,
            )
        return [dict(row) for row in rows]

    async def get_orders_by_buyer(
        self,
        buyer_id: str,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Fetch all orders placed by a buyer.

        Args:
            buyer_id: Buyer UUID string.
            status: Optional order_status filter.
            limit: Maximum rows to return.

        Returns:
            List of order dicts ordered newest-first.
        """
        params: list[Any] = [buyer_id, limit]
        status_clause = ""
        if status:
            status_clause = "AND o.order_status = $3"
            params = [buyer_id, limit, status]

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT o.*, l.commodity
                FROM orders o
                JOIN listings l ON l.id = o.listing_id
                WHERE o.buyer_id = $1::uuid
                  {status_clause}
                ORDER BY o.created_at DESC
                LIMIT $2
                """,
                *params,
            )
        return [dict(row) for row in rows]

    async def update_dispute(
        self,
        dispute_id: str,
        updates: dict[str, Any],
    ) -> bool:
        """
        Update a dispute record (status, diff_report, liability, claim_percent).

        Args:
            dispute_id: Dispute UUID string.
            updates: Dict with any subset of: status, diff_report,
                     liability, claim_percent, resolution_notes.

        Returns:
            True if the row was updated, False if not found.
        """
        allowed = {"status", "diff_report", "liability", "claim_percent", "resolution_notes"}
        fields = {k: v for k, v in updates.items() if k in allowed}
        if not fields:
            return False

        set_clauses = []
        params: list[Any] = []
        for i, (col, val) in enumerate(fields.items(), start=1):
            if col == "diff_report":
                set_clauses.append(f"{col} = ${i}::jsonb")
                params.append(json.dumps(val) if isinstance(val, dict) else val)
            else:
                set_clauses.append(f"{col} = ${i}")
                params.append(val)

        params.append(dispute_id)
        set_sql = ", ".join(set_clauses)

        async with self.pool.acquire() as conn:
            result = await conn.execute(
                f"UPDATE disputes SET {set_sql} WHERE id = ${len(params)}::uuid",
                *params,
            )
        updated = int(result.split()[-1]) > 0
        if updated:
            logger.info(f"Dispute {dispute_id} updated: {list(fields.keys())}")
        return updated

    async def get_dispute_status(self, dispute_id: str) -> Optional[dict[str, Any]]:
        """
        Fetch a dispute record with order and listing info.

        Args:
            dispute_id: Dispute UUID string.

        Returns:
            Dict with dispute fields, or None if not found.
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT d.*, o.order_status, o.aisp_total, l.commodity
                FROM disputes d
                JOIN orders o ON o.id = d.order_id
                JOIN listings l ON l.id = o.listing_id
                WHERE d.id = $1::uuid
                """,
                dispute_id,
            )
        return dict(row) if row else None

    # ═══════════════════════════════════════════════════════════════
    # Migrations
    # ═══════════════════════════════════════════════════════════════

    async def run_migrations(self) -> list[str]:
        """
        Apply all pending SQL migrations using the MigrationRunner.

        Returns:
            List of migration filenames that were applied.
        """
        from src.db.migrations.migration_runner import MigrationRunner

        runner = MigrationRunner(self.pool)
        applied = await runner.run_pending()
        return applied

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
