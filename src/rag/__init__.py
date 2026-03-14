"""RAG module - Advanced Agentic Retrieval Augmented Generation system."""

# Phase 5: Advanced Reranking
from src.rag.advanced_reranker import (
    AdvancedReranker,
    RerankerConfig,
    RerankerType,
    create_advanced_reranker,
)
from src.rag.contextual_chunker import (
    ChunkingConfig,
    ContextualChunker,
    EnrichedChunk,
    create_contextual_chunker,
    enrich_documents,
)
from src.rag.embeddings import EmbeddingManager, get_embedding_manager
from src.rag.enhanced_retriever import (
    EnhancedRetriever,
    MMRRetriever,
    ParentDocumentRetriever,
    RetrievalStrategy,
    SentenceWindowRetriever,
    create_enhanced_retriever,
)

# Phase 9: Evaluation & Testing
from src.rag.evaluation import (
    EvalResult,
    EvaluationSuite,
    TestDataPoint,
    create_evaluation_suite,
)
from src.rag.grader import DocumentGrader, GradeResult, HallucinationChecker
from src.rag.graph import RAGState, create_rag_graph, run_agentic_rag

# Phase 7: Enhanced Graph RAG
from src.rag.graph_constructor import (
    ConstructedGraph,
    EntityType,
    GraphConstructor,
    GraphEdge,
    GraphNode,
    RelationType,
    create_graph_constructor,
)
from src.rag.graph_retriever import (
    EntityExtractor,
    GraphAugmentedRetriever,
    GraphContext,
    GraphRetriever,
)

# Enhancement modules
from src.rag.hybrid_search import (
    BM25Index,
    HybridRetriever,
    HybridSearchResult,
    get_hybrid_retriever,
)
from src.rag.knowledge_base import Document, KnowledgeBase, SearchResult

# Phase 6: Real-Time Knowledge Injection
from src.rag.knowledge_injection import (
    AlertSeverity,
    KnowledgeInjector,
    MarketAlertSystem,
    NewsStreamer,
    RealTimeUpdate,
    SchemeCrawler,
    WeatherAdvisorySystem,
    create_knowledge_injector,
)
from src.rag.observability import RAGEvaluator, configure_langsmith, trace_rag

# Phase 8: Production Hardening
from src.rag.production import (
    ProductionGuard,
    RAGCache,
    RateLimiter,
    production_guard,
)
from src.rag.query_analyzer import QueryAnalysis, QueryAnalyzer, QueryCategory, QueryType

# Phase 3-4: Query Processing & Enhanced Retrieval
from src.rag.query_processor import (
    AdvancedQueryProcessor,
    ExpandedQuery,
    QueryExpansionType,
    QueryProcessorConfig,
    create_query_processor,
)

# Advanced RAG modules (Phase 2)
from src.rag.raptor import RAPTORConfig, RAPTORIndex, RAPTORNode, create_raptor_index
from src.rag.reranker import CrossEncoderReranker, LightweightReranker, RerankedResult, get_reranker
from src.rag.retriever import RAGRetriever, RetrievalResult, decompose_query

__all__ = [
    # Embeddings
    "EmbeddingManager",
    "get_embedding_manager",
    # Knowledge Base
    "Document",
    "KnowledgeBase",
    "SearchResult",
    # Query Analysis
    "QueryAnalyzer",
    "QueryAnalysis",
    "QueryType",
    "QueryCategory",
    # Grading
    "DocumentGrader",
    "GradeResult",
    "HallucinationChecker",
    # Retrieval
    "RAGRetriever",
    "RetrievalResult",
    "decompose_query",
    # Graph (LangGraph workflow)
    "create_rag_graph",
    "run_agentic_rag",
    "RAGState",
    # Hybrid Search
    "BM25Index",
    "HybridRetriever",
    "HybridSearchResult",
    "get_hybrid_retriever",
    # Reranking
    "CrossEncoderReranker",
    "LightweightReranker",
    "RerankedResult",
    "get_reranker",
    # Graph RAG
    "GraphRetriever",
    "GraphAugmentedRetriever",
    "GraphContext",
    "EntityExtractor",
    # Observability
    "configure_langsmith",
    "trace_rag",
    "RAGEvaluator",
    # RAPTOR (Phase 2)
    "RAPTORIndex",
    "RAPTORNode",
    "RAPTORConfig",
    "create_raptor_index",
    # Contextual Chunking (Phase 2)
    "ContextualChunker",
    "EnrichedChunk",
    "ChunkingConfig",
    "create_contextual_chunker",
    "enrich_documents",
    # Query Processing (Phase 3)
    "AdvancedQueryProcessor",
    "ExpandedQuery",
    "QueryProcessorConfig",
    "QueryExpansionType",
    "create_query_processor",
    # Enhanced Retrieval (Phase 4)
    "EnhancedRetriever",
    "ParentDocumentRetriever",
    "SentenceWindowRetriever",
    "MMRRetriever",
    "RetrievalStrategy",
    "create_enhanced_retriever",
    # Advanced Reranking (Phase 5)
    "AdvancedReranker",
    "RerankerConfig",
    "RerankerType",
    "create_advanced_reranker",
    # Knowledge Injection (Phase 6)
    "KnowledgeInjector",
    "NewsStreamer",
    "MarketAlertSystem",
    "WeatherAdvisorySystem",
    "SchemeCrawler",
    "RealTimeUpdate",
    "AlertSeverity",
    "create_knowledge_injector",
    # Graph Construction (Phase 7)
    "GraphConstructor",
    "ConstructedGraph",
    "GraphNode",
    "GraphEdge",
    "EntityType",
    "RelationType",
    "create_graph_constructor",
    # Production Hardening (Phase 8)
    "ProductionGuard",
    "RAGCache",
    "RateLimiter",
    "production_guard",
    # Evaluation (Phase 9)
    "EvaluationSuite",
    "TestDataPoint",
    "EvalResult",
    "create_evaluation_suite",
]


