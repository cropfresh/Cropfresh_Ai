"""
Contextual Chunking (Proxy)
===========================
This file exists for backward compatibility.
The actual implementation has been modularized into `src.rag.contextual_chunker`.
"""

from src.rag.contextual_chunker import (
    EnrichedChunk,
    ChunkingConfig,
    ContextualChunker,
    create_contextual_chunker,
    enrich_documents,
)

__all__ = [
    "EnrichedChunk",
    "ChunkingConfig",
    "ContextualChunker",
    "create_contextual_chunker",
    "enrich_documents",
]
