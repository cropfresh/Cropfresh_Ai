"""
Vector operations mixin for Aurora PostgreSQL (pgvector).

Replaces Qdrant vector knowledge base operations with pgvector extension.
"""

import json
from typing import Any
from uuid import uuid4

from loguru import logger


class VectorOperationsMixin:
    """
    Mixin providing vector search operations via pgvector.

    Requires the parent class to expose a `pool` property
    returning an asyncpg connection pool.
    """

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
