"""Lazy export registry for the ``src.rag`` package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Avoid eager imports so submodules can load without booting the full graph stack.
MODULE_EXPORTS: dict[str, list[str]] = {
    "src.rag.advanced_reranker": [
        "AdvancedReranker",
        "RerankerConfig",
        "RerankerType",
        "create_advanced_reranker",
    ],
    "src.rag.contextual_chunker": [
        "ChunkingConfig",
        "ContextualChunker",
        "EnrichedChunk",
        "create_contextual_chunker",
        "enrich_documents",
    ],
    "src.rag.embeddings": ["EmbeddingManager", "get_embedding_manager"],
    "src.rag.enhanced_retriever": [
        "EnhancedRetriever",
        "MMRRetriever",
        "ParentDocumentRetriever",
        "RetrievalStrategy",
        "SentenceWindowRetriever",
        "create_enhanced_retriever",
    ],
    "src.rag.evaluation": [
        "EvalResult",
        "EvaluationSuite",
        "TestDataPoint",
        "create_evaluation_suite",
    ],
    "src.rag.grader": [
        "DOC_GRADER_PROMPT",
        "DocumentGrader",
        "GradeResult",
        "GradingResult",
        "HallucinationChecker",
        "MARKET_DOC_MAX_AGE_SECONDS",
        "MARKET_KEYWORDS",
    ],
    "src.rag.graph": ["RAGState", "create_rag_graph", "run_agentic_rag"],
    "src.rag.graph_constructor": [
        "ConstructedGraph",
        "EntityType",
        "GraphConstructor",
        "GraphEdge",
        "GraphNode",
        "RelationType",
        "create_graph_constructor",
    ],
    "src.rag.graph_retriever": [
        "EntityExtractor",
        "GraphAugmentedRetriever",
        "GraphContext",
        "GraphRetriever",
    ],
    "src.rag.hybrid_search": [
        "BM25Index",
        "HybridRetriever",
        "HybridSearchResult",
        "get_hybrid_retriever",
    ],
    "src.rag.knowledge_base": ["Document", "KnowledgeBase", "SearchResult"],
    "src.rag.knowledge_injection": [
        "AlertSeverity",
        "KnowledgeInjector",
        "MarketAlertSystem",
        "NewsStreamer",
        "RealTimeUpdate",
        "SchemeCrawler",
        "WeatherAdvisorySystem",
        "create_knowledge_injector",
    ],
    "src.rag.observability": ["RAGEvaluator", "configure_langsmith", "trace_rag"],
    "src.rag.production": ["ProductionGuard", "RAGCache", "RateLimiter", "production_guard"],
    "src.rag.query_analyzer": [
        "ROUTE_COST_MAP",
        "AdaptiveQueryRouter",
        "QueryAnalysis",
        "QueryAnalyzer",
        "QueryCategory",
        "QueryType",
        "RetrievalRoute",
        "RoutingDecision",
    ],
    "src.rag.query_processor": [
        "AdvancedQueryProcessor",
        "ExpandedQuery",
        "QueryExpansionType",
        "QueryProcessorConfig",
        "create_query_processor",
    ],
    "src.rag.raptor": ["RAPTORConfig", "RAPTORIndex", "RAPTORNode", "create_raptor_index"],
    "src.rag.reranker": [
        "CrossEncoderReranker",
        "LightweightReranker",
        "RerankedResult",
        "get_reranker",
    ],
    "src.rag.retriever": ["RAGRetriever", "RetrievalResult", "decompose_query"],
}

SRC_RAG_EXPORTS = {
    name: module_name
    for module_name, names in MODULE_EXPORTS.items()
    for name in names
}
SRC_RAG_ALL = [name for names in MODULE_EXPORTS.values() for name in names]


def resolve_src_rag_export(name: str) -> Any:
    """Load a ``src.rag`` export on demand."""
    module_name = SRC_RAG_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'src.rag' has no attribute {name!r}")
    return getattr(import_module(module_name), name)
