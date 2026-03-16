from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.rag.graph_runtime.state import GraphRunResult


@pytest.mark.asyncio
async def test_run_agentic_rag_maps_graph_result_to_legacy_payload():
    document = SimpleNamespace(source="agmarknet", metadata={"source": "agmarknet"})
    graph_result = GraphRunResult(
        answer="Tomato price is Rs.2500. [1]",
        cited_answer="Tomato price is Rs.2500. [1]",
        confidence=0.9,
        faithfulness=0.88,
        relevance=0.86,
        retry_count=1,
        route="live_price_api",
        route_reason="Price query detected",
        tool_calls=["live_price_api"],
        documents=[document],
    )

    with patch("src.rag.graph.run_rag_graph", AsyncMock(return_value=graph_result)):
        from src.rag.graph import run_agentic_rag

        result = await run_agentic_rag("What is tomato price today?", knowledge_base=None, llm=None)

    assert result["query_type"] == "live_price_api"
    assert result["final_answer"] == "Tomato price is Rs.2500. [1]"
    assert result["sources"] == ["agmarknet"]
    assert result["tool_calls"] == ["live_price_api"]
