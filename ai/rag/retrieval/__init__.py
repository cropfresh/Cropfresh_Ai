"""
Advanced Retrieval Package — Phase 4 modules (ADR-010).

Re-exports the public API for advanced retrieval features:
- ContextualChunkEnricher: Anthropic-style chunk context injection
- QueryDecomposer: Multi-part query splitting
- TimeAwareRetriever: Freshness-boosted reranking
- AdvancedRetriever: Coordinator tying everything together
"""

from ai.rag.retrieval.advanced_retriever import AdvancedRetriever, AdvancedRetrievalResult
from ai.rag.retrieval.contextual_enricher import ContextualChunk, ContextualChunkEnricher
from ai.rag.retrieval.query_decomposer import DecomposedQuery, QueryDecomposer
from ai.rag.retrieval.time_aware import FreshnessCategory, TimeAwareResult, TimeAwareRetriever

__all__ = [
    "AdvancedRetriever",
    "AdvancedRetrievalResult",
    "ContextualChunkEnricher",
    "ContextualChunk",
    "QueryDecomposer",
    "DecomposedQuery",
    "TimeAwareRetriever",
    "TimeAwareResult",
    "FreshnessCategory",
]
