"""Lazy export registry for the ``ai.rag`` package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

MODULE_EXPORTS: dict[str, list[str]] = {
    "src.rag.agri_embeddings": ["AgriEmbeddingWrapper", "get_agri_embedding_manager"],
    "src.rag.agentic_orchestrator": [
        "AgenticOrchestrator",
        "AgenticSelfEvaluator",
        "Draft",
        "EvalGate",
        "OrchestratorResult",
        "RetrievalPlan",
        "RetrievalPlanner",
        "SpeculativeDraftEngine",
        "ToolCall",
    ],
    "src.rag.browser_rag": [
        "AgriSourceSelector",
        "BrowserRAGIntegration",
        "Citation",
        "CitedAnswer",
        "ContentExtractor",
        "QualityFilter",
        "ScrapeIntent",
        "TargetSource",
    ],
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
    "src.rag.grader": [
        "DOC_GRADER_PROMPT",
        "DocumentGrader",
        "GradeResult",
        "GradingResult",
        "HallucinationChecker",
        "MARKET_DOC_MAX_AGE_SECONDS",
        "MARKET_KEYWORDS",
    ],
}

AI_RAG_EXPORTS = {
    name: module_name
    for module_name, names in MODULE_EXPORTS.items()
    for name in names
}
AI_RAG_ALL = [name for names in MODULE_EXPORTS.values() for name in names]


def resolve_ai_rag_export(name: str) -> Any:
    """Load an ``ai.rag`` export on demand."""
    module_name = AI_RAG_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'ai.rag' has no attribute {name!r}")
    return getattr(import_module(module_name), name)
