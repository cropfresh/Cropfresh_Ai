"""RAG module - Advanced Agentic Retrieval Augmented Generation system."""

from src.rag.embeddings import EmbeddingManager, get_embedding_manager
from src.rag.knowledge_base import Document, KnowledgeBase, SearchResult
from src.rag.query_analyzer import QueryAnalyzer, QueryAnalysis, QueryType, QueryCategory
from src.rag.grader import DocumentGrader, GradeResult, HallucinationChecker
from src.rag.retriever import RAGRetriever, RetrievalResult, decompose_query
from src.rag.graph import create_rag_graph, run_agentic_rag, RAGState

# Enhancement modules
from src.rag.hybrid_search import BM25Index, HybridRetriever, HybridSearchResult, get_hybrid_retriever
from src.rag.reranker import CrossEncoderReranker, LightweightReranker, RerankedResult, get_reranker
from src.rag.graph_retriever import GraphRetriever, GraphAugmentedRetriever, GraphContext, EntityExtractor
from src.rag.observability import configure_langsmith, trace_rag, RAGEvaluator

# Advanced RAG modules (Phase 2)
from src.rag.raptor import RAPTORIndex, RAPTORNode, RAPTORConfig, create_raptor_index
from src.rag.contextual_chunker import (
    ContextualChunker,
    EnrichedChunk,
    ChunkingConfig,
    create_contextual_chunker,
    enrich_documents,
)

# Phase 3-4: Query Processing & Enhanced Retrieval
from src.rag.query_processor import (
    AdvancedQueryProcessor,
    ExpandedQuery,
    QueryProcessorConfig,
    QueryExpansionType,
    create_query_processor,
)
from src.rag.enhanced_retriever import (
    EnhancedRetriever,
    ParentDocumentRetriever,
    SentenceWindowRetriever,
    MMRRetriever,
    RetrievalStrategy,
    create_enhanced_retriever,
)

# Phase 5: Advanced Reranking
from src.rag.advanced_reranker import (
    AdvancedReranker,
    RerankerConfig,
    RerankerType,
    create_advanced_reranker,
)

# Phase 6: Real-Time Knowledge Injection
from src.rag.knowledge_injection import (
    KnowledgeInjector,
    NewsStreamer,
    MarketAlertSystem,
    WeatherAdvisorySystem,
    SchemeCrawler,
    RealTimeUpdate,
    AlertSeverity,
    create_knowledge_injector,
)

# Phase 7: Enhanced Graph RAG
from src.rag.graph_constructor import (
    GraphConstructor,
    ConstructedGraph,
    GraphNode,
    GraphEdge,
    EntityType,
    RelationType,
    create_graph_constructor,
)

# Phase 8: Production Hardening
from src.rag.production import (
    ProductionGuard,
    RAGCache,
    RateLimiter,
    production_guard,
)

# Phase 9: Evaluation & Testing
from src.rag.evaluation import (
    EvaluationSuite,
    TestDataPoint,
    EvalResult,
    create_evaluation_suite,
)

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


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 05: Agentic RAG & Adaptive Intelligence
# ─────────────────────────────────────────────────────────────────────────────

from ai.rag.agri_embeddings import AgriEmbeddingWrapper, get_agri_embedding_manager
from ai.rag.query_analyzer import (
    RetrievalRoute,
    RoutingDecision,
    AdaptiveQueryRouter,
    ROUTE_COST_MAP,
)
from ai.rag.agentic_orchestrator import (
    AgenticOrchestrator,
    SpeculativeDraftEngine,
    AgenticSelfEvaluator,
    RetrievalPlanner,
    RetrievalPlan,
    ToolCall,
    EvalGate,
    Draft,
    OrchestratorResult,
)

# Sprint 06: Browser-Augmented RAG
from ai.rag.browser_rag import (
    BrowserRAGIntegration,
    CitedAnswer,
    Citation,
    AgriSourceSelector,
    ContentExtractor,
    QualityFilter,
    ScrapeIntent,
    TargetSource,
)

__all__ += [
    # AgriEmbeddings (Sprint 05)
    "AgriEmbeddingWrapper",
    "get_agri_embedding_manager",
    # Adaptive Query Router (Sprint 05)
    "RetrievalRoute",
    "RoutingDecision",
    "AdaptiveQueryRouter",
    "ROUTE_COST_MAP",
    # Agentic Orchestrator (Sprint 05)
    "AgenticOrchestrator",
    "SpeculativeDraftEngine",
    "AgenticSelfEvaluator",
    "RetrievalPlanner",
    "RetrievalPlan",
    "ToolCall",
    "EvalGate",
    "Draft",
    "OrchestratorResult",
    # Browser RAG (Sprint 06)
    "BrowserRAGIntegration",
    "CitedAnswer",
    "Citation",
    "AgriSourceSelector",
    "ContentExtractor",
    "QualityFilter",
    "ScrapeIntent",
    "TargetSource",
]
