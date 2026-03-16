"""Conditional edge routing for the LangGraph RAG pipeline."""

from __future__ import annotations

from loguru import logger

from ai.rag.graph.state import RAGGraphState


def after_grade(state: RAGGraphState) -> str:
    """Route to web search once, then continue toward generation/gating."""
    needs_web = state.get("needs_web_search", False)
    relevant_docs = state.get("relevant_documents", [])
    web_search_attempted = state.get("web_search_attempted", False)

    if needs_web and not relevant_docs and not web_search_attempted:
        logger.info("after_grade -> web_search (no relevant docs)")
        return "web_search"

    logger.info("after_grade -> generate (relevant_docs={})", len(relevant_docs))
    return "generate"


def after_evaluate(state: RAGGraphState) -> str:
    """Retry low-confidence answers until the configured retry limit is hit."""
    should_retry = state.get("should_retry", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if should_retry and retry_count < max_retries:
        logger.info("after_evaluate -> rewrite (retry {}/{})", retry_count + 1, max_retries)
        return "rewrite"

    logger.info(
        "after_evaluate -> gate (confidence={:.2f}, retries={})",
        state.get("confidence", 0.0),
        retry_count,
    )
    return "gate"


def after_gate(state: RAGGraphState) -> str:
    """Finish after the confidence gate, whether approved or declined."""
    if state.get("is_approved", True):
        logger.info("after_gate -> end (approved)")
    else:
        logger.warning(
            "after_gate -> end (declined, safety={}, reason={})",
            state.get("safety_level", "safe"),
            state.get("decline_reason", "unknown"),
        )
    return "end"
