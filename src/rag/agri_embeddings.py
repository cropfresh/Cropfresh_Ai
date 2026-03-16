"""
Agricultural Domain Embedding Wrapper
======================================
Layer 1 of the two-layer agri-embedding strategy (ADR-009).

Wraps the existing BGE-M3 EmbeddingManager with:
- Agricultural domain instruction prefix for queries
- Bilingual Hindi/Kannada â†’ normalized English term map (60+ entries)
- Expected improvement: +8â€“12% context precision on agri golden dataset

Layer 2 (fine-tuned cropfresh-agri-embed-v1) is planned for Phase 4 / 2027.

Architecture: docs/architecture/agri_embeddings.md
ADR: docs/decisions/ADR-009-agri-embeddings.md
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from loguru import logger

from src.rag.agri_terms import AGRI_TERM_MAP
from src.rag.embeddings import EmbeddingManager


class AgriEmbeddingWrapper(EmbeddingManager):
    """Domain-tuned BGE-M3 wrapper for Indian agricultural retrieval."""

    # Domain-specific instruction prefix for agricultural queries
    # More specific than generic BGE instruction for better domain retrieval
    AGRI_QUERY_INSTRUCTION = (
        "Represent this Indian agricultural query for searching knowledge "
        "about crop cultivation, mandi commodity prices, pest and disease "
        "management, government agricultural schemes, soil and irrigation "
        "science, and Karnataka/Maharashtra farming practices: "
    )

    # Domain-specific instruction prefix for agricultural documents
    AGRI_DOC_INSTRUCTION = (
        "Represent this Indian agricultural document about farming knowledge, "
        "crop science, market information, or government schemes: "
    )

    TERM_MAP = AGRI_TERM_MAP

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        enable_term_normalization: bool = True,
    ):
        """
        Initialize the AgriEmbeddingWrapper.

        Args:
            model_name: HuggingFace model name (default: BAAI/bge-m3)
            device: Compute device â€” "cpu" or "cuda"
            cache_dir: Optional model cache directory
            enable_term_normalization: Set False to disable bilingual normalization (testing)
        """
        super().__init__(model_name=model_name, device=device, cache_dir=cache_dir)
        self.enable_term_normalization = enable_term_normalization
        logger.info(
            f"AgriEmbeddingWrapper initialized | "
            f"model={model_name} | "
            f"term_map_entries={len(self.TERM_MAP)} | "
            f"normalization={'ON' if enable_term_normalization else 'OFF'}"
        )

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a search query with agricultural domain context.

        Args:
            query: Raw user query (may contain Hindi/Kannada terms)

        Returns:
            1024-dimensional normalized embedding vector
        """
        # Step 1: Normalize bilingual terms
        if self.enable_term_normalization:
            query = self._normalize_terms(query)

        # Step 2: Add domain instruction prefix (replaces parent's generic BGE prefix)
        prefixed = f"{self.AGRI_QUERY_INSTRUCTION}{query}"

        # Step 3: Encode with base BGE-M3 model (skip parent prefix logic)
        embedding = self.model.encode(
            prefixed,
            normalize_embeddings=True,
        )

        logger.debug(f"AgriEmbeddingWrapper.embed_query | original_len={len(query)} | normalized_query={query[:80]}...")
        return embedding.tolist()

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Embed documents with agricultural domain instruction prefix.

        Args:
            texts: List of document texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            List of 1024-dimensional embedding vectors
        """
        if not texts:
            return []

        # Add agricultural instruction prefix to all documents
        enriched = [f"{self.AGRI_DOC_INSTRUCTION}{t}" for t in texts]

        logger.debug(f"AgriEmbeddingWrapper.embed_documents | n_docs={len(texts)}")
        embeddings = self.model.encode(
            enriched,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def embed_queries(
        self,
        queries: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """
        Embed multiple queries with agricultural domain context.

        Args:
            queries: List of query texts
            batch_size: Batch size for encoding

        Returns:
            List of 1024-dimensional embedding vectors
        """
        if not queries:
            return []

        # Normalize and prefix each query
        processed = []
        for q in queries:
            normalized = self._normalize_terms(q) if self.enable_term_normalization else q
            processed.append(f"{self.AGRI_QUERY_INSTRUCTION}{normalized}")

        embeddings = self.model.encode(
            processed,
            batch_size=batch_size,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def _normalize_terms(self, text: str) -> str:
        """
        Normalize bilingual agricultural terms to expanded English equivalents.

        Uses a single-pass regex replacement to prevent double-substitution
        (e.g., 'dhan' inside an already-expanded 'Pradhan Mantri' string).

        Terms are sorted by length (longest first) so more specific multi-word
        terms like 'kali mitti' are matched before shorter sub-terms.

        Args:
            text: Input text (may contain Hindi/Kannada terms)

        Returns:
            Text with expanded English equivalents

        Example:
            "pm-kisan scheme tamatar apply" â†’
            "PM-KISAN Pradhan Mantri Kisan Samman Nidhi income support scheme tomato Solanum lycopersicum apply"
        """
        import re

        text_lower = text.lower()

        # Build a single regex that matches all terms at word boundaries
        # Sort longest first to ensure multi-word terms match before sub-terms
        sorted_terms = sorted(self.TERM_MAP.keys(), key=len, reverse=True)

        # Escape special regex characters in term keys
        escaped_terms = [re.escape(t) for t in sorted_terms]

        # Build alternation pattern
        pattern = re.compile(
            r'\b(' + '|'.join(escaped_terms) + r')\b',
            re.IGNORECASE,
        )

        def replace_match(m: re.Match) -> str:
            matched = m.group(0).lower()
            return self.TERM_MAP.get(matched, m.group(0))

        return pattern.sub(replace_match, text_lower)


    def get_domain_stats(self) -> dict:
        """Return stats about the domain wrapper for observability."""
        return {
            "model_name": self.model_name,
            "term_map_size": len(self.TERM_MAP),
            "normalization_enabled": self.enable_term_normalization,
            "query_instruction_len": len(self.AGRI_QUERY_INSTRUCTION),
            "doc_instruction_len": len(self.AGRI_DOC_INSTRUCTION),
            "vector_dimensions": self._dimensions,
        }


@lru_cache(maxsize=1)
def get_agri_embedding_manager(
    model_name: str = "BAAI/bge-m3",
    device: str = "cpu",
) -> AgriEmbeddingWrapper:
    """
    Get a cached AgriEmbeddingWrapper instance.

    Uses LRU cache to avoid reloading the embedding model on every call.
    Respects EMBEDDING_DEVICE env var for device selection.

    Args:
        model_name: HuggingFace model name
        device: Compute device ("cpu" or "cuda")

    Returns:
        Singleton AgriEmbeddingWrapper instance
    """
    # Allow env override
    device = os.getenv("EMBEDDING_DEVICE", device)

    logger.info(f"Creating AgriEmbeddingWrapper | model={model_name} | device={device}")
    return AgriEmbeddingWrapper(model_name=model_name, device=device)
