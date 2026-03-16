"""
RAG Graph Nodes — LangGraph node functions (ADR-010 Phase 3).

Each function takes RAGGraphState and returns a partial state update.
Nodes delegate to existing Phase 1 modules: QueryRewriter, DocumentGrader,
CitationEngine, ConfidenceGate, and the AgenticSelfEvaluator.

All under ~200 LOC by importing existing implementations.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from ai.rag.graph.state import RAGGraphState
from ai.rag.routing import AdaptiveQueryRouter, RetrievalRoute


# ---------------------------------------------------------------------------
# Node 1: Query Rewriting
# ---------------------------------------------------------------------------

async def rewrite_node(state: RAGGraphState) -> dict[str, Any]:
    """Rewrite query using HyDE / step-back / multi-query."""
    query = state.get("query", "")
    if not query:
        return {"rewritten_queries": [], "rewrite_strategy": "none"}

    try:
        llm = state.get("llm")
        router = AdaptiveQueryRouter(llm=llm)
        decision = await router.route(query, has_image=state.get("has_image", False))
        from ai.rag.query_rewriter import QueryRewriter

        rewrite_strategy = "none" if decision.strategy is not RetrievalRoute.FULL_AGENTIC else "auto"
        rewriter = QueryRewriter(llm=llm)
        result = await rewriter.rewrite(query, strategy=rewrite_strategy)

        logger.info(
            "rewrite_node: route={} strategy={} | queries={}",
            decision.strategy.value,
            result.strategy_used,
            len(result.rewritten_queries),
        )
        return {
            "rewritten_queries": result.rewritten_queries,
            "rewrite_strategy": result.strategy_used,
            "hyde_document": result.hyde_document or "",
            "route": decision.strategy.value,
            "route_reason": decision.reason,
            "tool_calls": [decision.strategy.value],
            "current_node": "rewrite",
        }
    except Exception as e:
        logger.warning(f"rewrite_node failed: {e}")
        return {
            "rewritten_queries": [query],
            "rewrite_strategy": "fallback",
            "route": RetrievalRoute.VECTOR_ONLY.value,
            "route_reason": "Rewrite fallback",
            "tool_calls": [RetrievalRoute.VECTOR_ONLY.value],
            "current_node": "rewrite",
        }


# ---------------------------------------------------------------------------
# Node 2: Document Retrieval
# ---------------------------------------------------------------------------

async def retrieve_node(state: RAGGraphState) -> dict[str, Any]:
    """Retrieve documents using rewritten queries."""
    queries = state.get("rewritten_queries", [state.get("query", "")])
    if not queries:
        return {"documents": [], "retrieval_source": "none"}

    try:
        from ai.rag.graph.services import (
            retrieve_browser_documents,
            retrieve_live_price_documents,
            retrieve_vector_documents,
            retrieve_weather_documents,
        )

        route = state.get("route", RetrievalRoute.VECTOR_ONLY.value)
        if route == RetrievalRoute.LIVE_PRICE_API.value:
            all_docs = await retrieve_live_price_documents(state.get("query", ""))
            retrieval_source = "live_price"
        elif route == RetrievalRoute.WEATHER_API.value:
            all_docs = await retrieve_weather_documents(state.get("query", ""))
            retrieval_source = "weather"
        elif route == RetrievalRoute.BROWSER_SCRAPE.value:
            all_docs = await retrieve_browser_documents(
                state.get("query", ""),
                state.get("web_search_tool"),
            )
            retrieval_source = "browser"
        else:
            all_docs = await retrieve_vector_documents(
                state.get("knowledge_base"),
                queries,
                top_k=5,
            )
            retrieval_source = "vector"

        logger.info(
            f"retrieve_node: {len(all_docs)} unique docs from "
            f"{len(queries)} queries"
        )
        return {
            "documents": all_docs,
            "retrieval_source": retrieval_source,
            "current_node": "retrieve",
        }
    except Exception as e:
        logger.warning(f"retrieve_node failed: {e}")
        return {
            "documents": [],
            "retrieval_source": "error",
            "current_node": "retrieve",
        }


# ---------------------------------------------------------------------------
# Node 3: Document Grading
# ---------------------------------------------------------------------------

async def grade_node(state: RAGGraphState) -> dict[str, Any]:
    """Grade retrieved documents for relevance using enhanced grader."""
    documents = state.get("documents", [])
    query = state.get("query", "")

    if not documents:
        return {
            "relevant_documents": [],
            "needs_web_search": True,
            "grading_scores": [],
            "current_node": "grade",
        }

    try:
        from ai.rag.grader import DocumentGrader

        grader = DocumentGrader(llm=None)
        result = await grader.grade_documents(documents, query)

        scores = [
            getattr(doc, "score", 0.5)
            for doc in result.relevant_documents
        ]

        logger.info(
            f"grade_node: {len(result.relevant_documents)}/{len(documents)} "
            f"relevant | web_search={result.needs_web_search}"
        )
        return {
            "relevant_documents": result.relevant_documents,
            "needs_web_search": result.needs_web_search,
            "grading_scores": scores,
            "current_node": "grade",
        }
    except Exception as e:
        logger.warning(f"grade_node failed: {e}")
        return {
            "relevant_documents": documents,
            "needs_web_search": False,
            "current_node": "grade",
        }


# ---------------------------------------------------------------------------
# Node 4: Answer Generation (Speculative Drafting)
# ---------------------------------------------------------------------------

async def generate_node(state: RAGGraphState) -> dict[str, Any]:
    """Generate answer using speculative draft engine."""
    docs = state.get("relevant_documents", state.get("documents", []))
    query = state.get("query", "")

    try:
        from ai.rag.graph.services import generate_answer

        answer, model_name = await generate_answer(
            query=query,
            documents=docs,
            llm=state.get("llm"),
        )

        logger.info(f"generate_node: answer_len={len(answer)}")
        return {
            "answer": answer,
            "generation_model": model_name,
            "current_node": "generate",
        }
    except Exception as e:
        logger.warning(f"generate_node failed: {e}")
        return {
            "answer": "Unable to generate an answer at this time.",
            "generation_model": "error",
            "current_node": "generate",
        }


# ---------------------------------------------------------------------------
# Node 5: Citation
# ---------------------------------------------------------------------------

async def cite_node(state: RAGGraphState) -> dict[str, Any]:
    """Add inline citations to the answer."""
    answer = state.get("answer", "")
    docs = state.get("relevant_documents", [])

    if not answer or not docs:
        return {
            "cited_answer": answer,
            "sources": [],
            "citation_count": 0,
            "current_node": "cite",
        }

    try:
        from ai.rag.citation_engine import CitationEngine

        engine = CitationEngine(llm=None)
        result = await engine.add_citations(answer, docs)

        return {
            "cited_answer": result.answer,
            "sources": [s.model_dump() for s in result.sources],
            "citation_count": result.citation_count,
            "current_node": "cite",
        }
    except Exception as e:
        logger.warning(f"cite_node failed: {e}")
        return {"cited_answer": answer, "current_node": "cite"}
