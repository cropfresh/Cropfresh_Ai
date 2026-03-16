"""
RAG Graph Safety Nodes — Gate and Evaluate (ADR-010 Phase 3).

Separated from nodes.py to stay under ~200 LOC per file.
Implements the confidence gate and self-evaluation nodes.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from ai.rag.graph.state import RAGGraphState

# ---------------------------------------------------------------------------
# Node 6: Confidence Gate
# ---------------------------------------------------------------------------

async def gate_node(state: RAGGraphState) -> dict[str, Any]:
    """Apply confidence gate to the cited answer."""
    answer = state.get("cited_answer", state.get("answer", ""))
    query = state.get("query", "")
    docs = state.get("relevant_documents", [])

    if not answer:
        return {
            "is_approved": False,
            "safety_level": "safe",
            "grounding_score": 0.0,
            "decline_reason": "No answer generated",
            "current_node": "gate",
        }

    try:
        from ai.rag.confidence_gate import ConfidenceGate

        gate = ConfidenceGate(llm=None)
        result = await gate.gate(
            query=query,
            answer=answer,
            documents=docs,
            faithfulness=state.get("faithfulness", 0.8),
            relevance=state.get("relevance", 0.8),
        )

        logger.info(
            f"gate_node: approved={result.is_approved} | "
            f"safety={result.safety_level.value} | "
            f"grounding={result.grounding_score:.2f}"
        )

        #! Override answer with decline response if not approved
        final_answer = answer if result.is_approved else result.answer

        return {
            "is_approved": result.is_approved,
            "safety_level": result.safety_level.value,
            "grounding_score": result.grounding_score,
            "decline_reason": result.decline_reason or "",
            "answer": final_answer,
            "current_node": "gate",
        }
    except Exception as e:
        logger.warning(f"gate_node failed: {e}")
        return {
            "is_approved": True,
            "safety_level": "safe",
            "current_node": "gate",
        }


# ---------------------------------------------------------------------------
# Node 7: Self-Evaluation (Retry decision)
# ---------------------------------------------------------------------------

async def evaluate_node(state: RAGGraphState) -> dict[str, Any]:
    """Evaluate answer quality; decide if retry is needed."""
    answer = state.get("answer", "")
    query = state.get("query", "")
    docs = state.get("relevant_documents", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if not answer or retry_count >= max_retries:
        return {
            "faithfulness": 0.8,
            "relevance": 0.8,
            "confidence": 0.8,
            "should_retry": False,
            "eval_reason": "Evaluation skipped",
            "current_node": "evaluate",
        }

    try:
        from ai.rag.agentic.evaluator import AgenticSelfEvaluator

        evaluator = AgenticSelfEvaluator(llm=state.get("llm"))
        gate = await evaluator.evaluate(
            query=query,
            answer=answer,
            retrieved_docs=docs,
        )

        should_retry = gate.should_retry and retry_count < max_retries

        logger.info(
            f"evaluate_node: faith={gate.faithfulness:.2f} | "
            f"rel={gate.relevance:.2f} | conf={gate.confidence:.2f} | "
            f"retry={should_retry} (attempt {retry_count + 1})"
        )

        return {
            "faithfulness": gate.faithfulness,
            "relevance": gate.relevance,
            "confidence": gate.confidence,
            "should_retry": should_retry,
            "eval_reason": gate.reason,
            "retry_count": retry_count + 1 if should_retry else retry_count,
            "current_node": "evaluate",
        }
    except Exception as e:
        logger.warning(f"evaluate_node failed: {e}")
        return {
            "faithfulness": 0.8,
            "relevance": 0.8,
            "confidence": 0.8,
            "should_retry": False,
            "eval_reason": f"Evaluation error: {e}",
            "current_node": "evaluate",
        }


# ---------------------------------------------------------------------------
# Node 8: Web Search Fallback
# ---------------------------------------------------------------------------

async def web_search_node(state: RAGGraphState) -> dict[str, Any]:
    """Fallback web search when grading finds insufficient docs."""
    query = state.get("query", "")

    logger.info(f"web_search_node: searching for '{query[:60]}...'")

    try:
        from ai.rag.graph.services import retrieve_browser_documents

        docs = await retrieve_browser_documents(
            query,
            state.get("web_search_tool"),
        )
        return {
            "documents": state.get("documents", []) + docs,
            "retrieval_source": "web_fallback" if docs else state.get("retrieval_source", "none"),
            "needs_web_search": False,
            "web_search_attempted": True,
            "current_node": "web_search",
        }
    except Exception as e:
        logger.warning(f"web_search_node failed: {e}")
        return {"web_search_attempted": True, "current_node": "web_search"}
