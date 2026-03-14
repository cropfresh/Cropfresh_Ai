"""
Contextual Chunker Package
==========================
Enhanced document chunking with metadata and LLM-generated summaries.
"""

from typing import Optional

from .models import EnrichedChunk, ChunkingConfig
from .chunker import ContextualChunker


# Factory function
def create_contextual_chunker(
    llm=None,
    config: Optional[ChunkingConfig] = None,
) -> ContextualChunker:
    """
    Create a configured contextual chunker.
    
    Args:
        llm: LLM for context generation
        config: Chunking configuration
        
    Returns:
        ContextualChunker instance
    """
    return ContextualChunker(llm=llm, config=config)


# Utility function for batch processing
async def enrich_documents(
    documents: list,
    llm=None,
    config: Optional[ChunkingConfig] = None,
) -> list[EnrichedChunk]:
    """
    Convenience function to chunk multiple documents with context.
    
    Args:
        documents: List of documents
        llm: LLM for context generation
        config: Chunking configuration
        
    Returns:
        Flat list of all EnrichedChunk from all documents
    """
    chunker = create_contextual_chunker(llm=llm, config=config)
    
    all_chunks = []
    for doc in documents:
        chunks = await chunker.chunk_with_context(doc)
        all_chunks.extend(chunks)
    
    return all_chunks

__all__ = [
    "EnrichedChunk",
    "ChunkingConfig",
    "ContextualChunker",
    "create_contextual_chunker",
    "enrich_documents",
]
