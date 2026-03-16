"""
RAG Graph Builder — Compiles the LangGraph state machine (ADR-010 Phase 3).

Assembles nodes, edges, and conditional routing into a compiled StateGraph.

Pipeline flow:
    rewrite → retrieve → grade ─┬─→ generate → cite → evaluate ─┬─→ gate → END
                                 │                                │
                                 └─→ web_search → grade ──────────└─→ rewrite (retry)
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph
from loguru import logger

from src.rag.graph_runtime.edges import after_evaluate, after_gate, after_grade
from src.rag.graph_runtime.nodes import (
    cite_node,
    generate_node,
    grade_node,
    retrieve_node,
    rewrite_node,
)
from src.rag.graph_runtime.nodes_safety import evaluate_node, gate_node, web_search_node
from src.rag.graph_runtime.state import GraphRunResult, RAGGraphState


def build_rag_graph() -> StateGraph:
    """Build and return the compiled RAG LangGraph state machine.

    Returns:
        Compiled StateGraph ready for `.ainvoke(state)`.
    """
    graph = StateGraph(RAGGraphState)

    # ── Register nodes ─────────────────────────────────────────────────
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generate", generate_node)
    graph.add_node("cite", cite_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("gate", gate_node)

    # ── Define edges ───────────────────────────────────────────────────
    graph.set_entry_point("rewrite")

    # Linear flow: rewrite → retrieve → grade
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade")

    # Conditional: grade → generate | web_search
    graph.add_conditional_edges(
        "grade",
        after_grade,
        {"generate": "generate", "web_search": "web_search"},
    )

    # Web search loops back to grade for re-evaluation
    graph.add_edge("web_search", "grade")

    # Linear: generate → cite → evaluate
    graph.add_edge("generate", "cite")
    graph.add_edge("cite", "evaluate")

    # Conditional: evaluate → gate (done) | rewrite (retry)
    graph.add_conditional_edges(
        "evaluate",
        after_evaluate,
        {"gate": "gate", "rewrite": "rewrite"},
    )

    # Conditional: gate → END
    graph.add_conditional_edges(
        "gate",
        after_gate,
        {"end": END},
    )

    logger.info("RAG LangGraph state machine built successfully")
    return graph.compile()


async def run_rag_graph(
    query: str,
    context: str = "",
    has_image: bool = False,
    max_retries: int = 2,
    knowledge_base: Any = None,
    llm: Any = None,
    web_search_tool: Any = None,
) -> GraphRunResult:
    """Run the full RAG pipeline through the LangGraph state machine.

    Args:
        query: User query text.
        context: Optional session context (previous turns).
        has_image: Whether the query includes an image.
        max_retries: Maximum self-correction retries.

    Returns:
        GraphRunResult with answer, citations, confidence scores.
    """
    compiled_graph = build_rag_graph()

    initial_state: RAGGraphState = {
        "query": query,
        "context": context,
        "has_image": has_image,
        "knowledge_base": knowledge_base,
        "llm": llm,
        "web_search_tool": web_search_tool,
        "retry_count": 0,
        "max_retries": max_retries,
        "web_search_attempted": False,
        "current_node": "start",
    }

    try:
        final_state = await compiled_graph.ainvoke(initial_state)
        final_answer = final_state.get("answer") or final_state.get("cited_answer", "")
        cited_answer = final_state.get("cited_answer", final_answer)

        return GraphRunResult(
            answer=final_answer,
            cited_answer=cited_answer,
            sources=final_state.get("sources", []),
            is_approved=final_state.get("is_approved", False),
            safety_level=final_state.get("safety_level", "safe"),
            confidence=final_state.get("confidence", 0.0),
            faithfulness=final_state.get("faithfulness", 0.0),
            relevance=final_state.get("relevance", 0.0),
            retry_count=final_state.get("retry_count", 0),
            grounding_score=final_state.get("grounding_score", 0.0),
            documents_used=len(final_state.get("relevant_documents", [])),
            rewrite_strategy=final_state.get("rewrite_strategy", "none"),
            route=final_state.get("route", ""),
            route_reason=final_state.get("route_reason", ""),
            tool_calls=final_state.get("tool_calls", []),
            documents=final_state.get("relevant_documents", final_state.get("documents", [])),
        )
    except Exception as e:
        logger.error(f"RAG graph execution failed: {e}")
        return GraphRunResult(
            answer="I'm sorry, I couldn't process your query right now.",
            error=str(e),
        )
