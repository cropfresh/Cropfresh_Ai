"""Compatibility facade for the split agentic RAG graph runtime."""

from __future__ import annotations

from typing import Any

from ai.rag.graph import GraphRunResult, RAGGraphState, build_rag_graph, run_rag_graph

RAGState = RAGGraphState


def create_rag_graph(
    knowledge_base: Any,
    llm: Any,
    web_search_tool: Any = None,
):
    """Return the compiled graph while keeping the legacy signature stable."""
    del knowledge_base, llm, web_search_tool
    return build_rag_graph()


def _result_to_legacy_payload(question: str, result: GraphRunResult) -> dict[str, Any]:
    """Map the new graph result into the legacy dict contract."""
    documents = result.documents
    sources = []
    for document in documents:
        source_name = getattr(document, "source", "")
        if not source_name:
            metadata = getattr(document, "metadata", {}) or {}
            source_name = str(metadata.get("source", ""))
        if source_name:
            sources.append(source_name)

    return {
        "question": question,
        "query_type": result.route or "vector_only",
        "generation": result.cited_answer or result.answer,
        "final_answer": result.answer,
        "documents": documents,
        "steps": [
            step
            for step in (
                "rewrite",
                "retrieve",
                "grade",
                "generate",
                "cite",
                "evaluate",
                "gate",
            )
        ],
        "web_search": "Yes" if result.route in {"browser_scrape", "weather_api"} else "No",
        "retry_count": result.retry_count,
        "sources": list(dict.fromkeys(sources)),
        "confidence": result.confidence,
        "faithfulness": result.faithfulness,
        "relevance": result.relevance,
        "route_reason": result.route_reason,
        "tool_calls": result.tool_calls,
    }


async def run_agentic_rag(
    question: str,
    knowledge_base: Any,
    llm: Any,
    web_search_tool: Any = None,
) -> dict[str, Any]:
    """Run the canonical graph runtime and return the legacy payload shape."""
    result = await run_rag_graph(
        query=question,
        knowledge_base=knowledge_base,
        llm=llm,
        web_search_tool=web_search_tool,
    )
    return _result_to_legacy_payload(question, result)
