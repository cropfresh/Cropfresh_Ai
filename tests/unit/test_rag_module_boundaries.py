from pathlib import Path

from src.rag.grader import DocumentGrader as AiDocumentGrader
from src.rag.grader import DocumentGrader as SrcDocumentGrader
from src.rag.query_analyzer import AdaptiveQueryRouter as AiAdaptiveQueryRouter
from src.rag.query_analyzer import AdaptiveQueryRouter as SrcAdaptiveQueryRouter
from src.rag.query_analyzer import QueryAnalyzer as AiQueryAnalyzer
from src.rag.query_analyzer import QueryAnalyzer as SrcQueryAnalyzer

ROOT = Path(__file__).resolve().parents[2]


def test_src_surface_exposes_current_router_and_grader():
    assert SrcAdaptiveQueryRouter is AiAdaptiveQueryRouter
    assert SrcQueryAnalyzer is AiQueryAnalyzer
    assert SrcDocumentGrader is AiDocumentGrader


def test_ai_rag_duplicate_top_level_modules_are_removed():
    removed_paths = [
        "ai/rag/graph.py",
        "ai/rag/evaluation.py",
        "ai/rag/knowledge_base.py",
        "ai/rag/embeddings.py",
        "ai/rag/retriever.py",
        "ai/rag/reranker.py",
    ]

    for relative_path in removed_paths:
        assert not (ROOT / relative_path).exists()
