from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from ai.rag.grader import GradingResult
from src.agents.knowledge_agent import KnowledgeAgent
from src.rag.knowledge_base import Document, SearchResult
from src.tools.agmarknet import AgmarknetPrice


class FakeKnowledgeBase:
    def __init__(self, documents: list[Document]):
        self.documents = documents

    async def search(self, query: str, top_k: int = 5):
        del query, top_k
        return SearchResult(
            documents=self.documents,
            query="test",
            total_found=len(self.documents),
            search_time_ms=1.0,
        )


def _build_agent(documents: list[Document]) -> KnowledgeAgent:
    agent = KnowledgeAgent(llm=None)
    agent._initialized = True
    agent._knowledge_base = FakeKnowledgeBase(documents)
    return agent


@pytest.mark.asyncio
async def test_live_market_query_returns_fresh_timestamped_sources(monkeypatch):
    async def fake_get_prices(self, commodity: str, state: str, district: str | None = None, market=None, limit: int = 20):
        del self, state, market, limit
        return [
            AgmarknetPrice(
                commodity=commodity,
                state="Karnataka",
                district=district or "Kolar",
                market="Kolar Main Market",
                date=datetime.now(),
                min_price=2200,
                max_price=2800,
                modal_price=2500,
            )
        ]

    monkeypatch.setattr("src.tools.agmarknet.AgmarknetTool.get_prices", fake_get_prices)
    result = await _build_agent([]).answer_with_debug("What is the price of tomato in Kolar mandi today?")

    assert result.route == "live_price_api"
    assert result.citations
    assert result.source_details[0].timestamp
    assert result.source_details[0].is_fresh is True


@pytest.mark.asyncio
async def test_agronomy_query_returns_citations_from_vector_path():
    documents = [
        Document(
            id="ragi-1",
            text="Ragi thrives in red laterite soil in Karnataka and should be sown in June-July.",
            source="kb",
            metadata={"source": "kb", "title": "Ragi Guide"},
        )
    ]

    result = await _build_agent(documents).answer_with_debug("How to grow ragi in red soil Karnataka?")

    assert result.route == "vector_only"
    assert result.citations
    assert "Ragi thrives in red laterite soil" in result.contexts[0]


@pytest.mark.asyncio
async def test_safety_critical_query_abstains_when_grounding_is_weak():
    documents = [
        Document(
            id="doc-1",
            text="Neem oil is a low-residue option for vegetables, but dosage guidance must come from verified recommendations.",
            source="kb",
            metadata={"source": "kb"},
        )
    ]

    with patch(
        "ai.rag.graph.services.generate_answer",
        AsyncMock(return_value=("Spray pesticide X at 50 ml per litre and sell tomorrow.", "mocked")),
    ), patch(
        "ai.rag.grader.DocumentGrader.grade_documents",
        AsyncMock(
            return_value=GradingResult(
                relevant_documents=documents,
                irrelevant_count=0,
                needs_web_search=False,
                total_graded=1,
            )
        ),
    ):
        result = await _build_agent(documents).answer_with_debug(
            "Safe pesticide for vegetables that I can spray and sell next day?"
        )

    assert "I don't have enough verified information" in result.answer
    assert result.route == "vector_only"


@pytest.mark.asyncio
async def test_kannada_market_query_reuses_live_price_route(monkeypatch):
    async def fake_get_prices(self, commodity: str, state: str, district: str | None = None, market=None, limit: int = 20):
        del self, state, market, limit
        return [
            AgmarknetPrice(
                commodity=commodity,
                state="Karnataka",
                district=district or "Kolar",
                market="Kolar Main Market",
                date=datetime.now(),
                min_price=2100,
                max_price=2900,
                modal_price=2550,
            )
        ]

    monkeypatch.setattr("src.tools.agmarknet.AgmarknetTool.get_prices", fake_get_prices)
    result = await _build_agent([]).answer_with_debug("Tomato bele yaavaga Kolar mandi alli?")

    assert result.route == "live_price_api"
    assert "live_price_api" in result.tool_calls
