from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.knowledge_agent import KnowledgeAgent
from src.rag.knowledge_base import Document


@pytest.mark.asyncio
async def test_answer_with_debug_maps_graph_payload():
    agent = KnowledgeAgent(llm=None)
    agent._initialized = True
    agent._knowledge_base = AsyncMock()
    document = Document(
        id="doc-1",
        text="Tomato mandi price is Rs.2500 per quintal today.",
        source="agmarknet",
        metadata={"source": "agmarknet", "timestamp": 9999999999, "as_of": "2026-03-16T10:00:00"},
    )

    with patch(
        "src.rag.graph.run_agentic_rag",
        AsyncMock(
            return_value={
                "final_answer": "Tomato price is Rs.2500 today. [1]",
                "generation": "Tomato price is Rs.2500 today. [1]",
                "documents": [document],
                "query_type": "live_price_api",
                "tool_calls": ["live_price_api"],
                "confidence": 0.91,
                "retry_count": 1,
                "web_search": "No",
                "steps": ["rewrite", "retrieve", "cite"],
                "route_reason": "Price query detected",
            }
        ),
    ):
        result = await agent.answer_with_debug("What is tomato price in Kolar today?")

    assert result.answer == "Tomato price is Rs.2500 today. [1]"
    assert result.route == "live_price_api"
    assert result.tool_calls == ["live_price_api"]
    assert result.citations == ["[1]"]
    assert result.source_details[0].is_fresh is True


@pytest.mark.asyncio
async def test_answer_uses_debug_result_contract():
    agent = KnowledgeAgent(llm=None)
    debug_result = AsyncMock(
        return_value=type(
            "DebugResult",
            (),
            {
                "answer": "Ragi grows well in red soil.",
                "sources": ["kb"],
                "confidence": 0.82,
                "route": "vector_only",
                "metadata": {"steps": ["rewrite", "retrieve"]},
            },
        )()
    )
    agent.answer_with_debug = debug_result

    result = await agent.answer("How to grow ragi?")

    assert result.answer == "Ragi grows well in red soil."
    assert result.sources == ["kb"]
    assert result.query_type == "vector_only"
    assert result.steps == ["rewrite", "retrieve"]
