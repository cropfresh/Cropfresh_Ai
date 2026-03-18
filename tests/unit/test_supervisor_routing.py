"""
Unit Tests: Supervisor Agent — Routing
=======================================
Tests the _route_rule_based() fallback method and routing logic:
  - Keyword matches route to correct agent
  - Zero keyword matches fall back to general_agent
  - Confidence scales with keyword density
  - _merge_responses() combines content and metadata correctly
"""

from unittest.mock import AsyncMock

import pytest

from src.agents.base_agent import AgentResponse
from src.agents.supervisor import SupervisorAgent

# ─────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────


@pytest.fixture
def supervisor():
    return SupervisorAgent(llm=None)  # No LLM → always uses rule-based routing


# ─────────────────────────────────────────────────
# Rule-based routing tests
# ─────────────────────────────────────────────────


class TestRuleBasedRouting:
    def test_agronomy_keywords(self, supervisor):
        decision = supervisor._route_rule_based("How to grow tomatoes in red soil?")
        assert decision.agent_name == "agronomy_agent"
        assert decision.confidence >= 0.3

    def test_commerce_keywords(self, supervisor):
        decision = supervisor._route_rule_based("What is the mandi price per quintal?")
        assert decision.agent_name == "commerce_agent"

    def test_kannada_price_query_routes_to_commerce(self, supervisor):
        decision = supervisor._route_rule_based("ಟೊಮೆಟೊ ಬೆಲೆ ಎಷ್ಟು?")
        assert decision.agent_name == "commerce_agent"

    def test_platform_keywords(self, supervisor):
        decision = supervisor._route_rule_based("How do I register on CropFresh app?")
        assert decision.agent_name == "platform_agent"

    def test_buyer_matching_keywords(self, supervisor):
        decision = supervisor._route_rule_based("Can you find buyer for my tomato listing?")
        assert decision.agent_name == "buyer_matching_agent"

    def test_quality_assessment_keywords(self, supervisor):
        decision = supervisor._route_rule_based(
            "Please do quality check for tomato defects and shelf life"
        )
        assert decision.agent_name == "quality_assessment_agent"

    def test_scraping_keywords(self, supervisor):
        decision = supervisor._route_rule_based("Get the current live price from Agmarknet today")
        assert decision.agent_name == "web_scraping_agent"

    def test_research_keywords(self, supervisor):
        decision = supervisor._route_rule_based(
            "Please give a comprehensive research report and analysis"
        )
        assert decision.agent_name == "research_agent"

    def test_general_keywords(self, supervisor):
        decision = supervisor._route_rule_based("Hello! Who are you?")
        assert decision.agent_name == "general_agent"

    def test_no_match_falls_back_to_general(self, supervisor):
        decision = supervisor._route_rule_based("xyzzy frobble snark")
        assert decision.agent_name == "general_agent"
        assert decision.confidence >= 0.3

    def test_confidence_capped_at_09(self, supervisor):
        # Flood with many agronomy keywords
        query = "grow plant cultivate harvest pest disease fertilizer soil seed irrigation organic variety crop farming agriculture"
        decision = supervisor._route_rule_based(query)
        assert decision.confidence <= 0.9


# ─────────────────────────────────────────────────
# _merge_responses tests
# ─────────────────────────────────────────────────


class TestMergeResponses:
    def _make_response(self, content: str, agent: str, confidence: float = 0.8) -> AgentResponse:
        return AgentResponse(
            content=content,
            agent_name=agent,
            confidence=confidence,
            sources=["src_a"],
            steps=["step1"],
            tools_used=["tool_a"],
        )

    def test_content_combined(self, supervisor):
        primary = self._make_response("Primary content", "agronomy_agent")
        secondary = self._make_response("Secondary content", "commerce_agent")
        merged = supervisor._merge_responses(primary, secondary)
        assert "Primary content" in merged.content
        assert "Secondary content" in merged.content

    def test_confidence_averaged(self, supervisor):
        primary = self._make_response("P", "agronomy_agent", confidence=0.8)
        secondary = self._make_response("S", "commerce_agent", confidence=0.6)
        merged = supervisor._merge_responses(primary, secondary)
        assert merged.confidence == pytest.approx(0.7)

    def test_sources_deduplicated(self, supervisor):
        r1 = self._make_response("P", "a")
        r2 = self._make_response("S", "b")
        r1.sources = ["doc_a", "doc_b"]
        r2.sources = ["doc_b", "doc_c"]
        merged = supervisor._merge_responses(r1, r2)
        assert len(merged.sources) == 3  # deduplicated

    def test_steps_concatenated(self, supervisor):
        primary = self._make_response("P", "agronomy_agent")
        secondary = self._make_response("S", "commerce_agent")
        merged = supervisor._merge_responses(primary, secondary)
        assert "merge:commerce_agent" in merged.steps


# ─────────────────────────────────────────────────
# JSON parse failure fallback
# ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_json_parse_failure_falls_back_to_rules():
    """When LLM returns invalid JSON, routing should fall back to rule-based."""
    mock_llm = AsyncMock()
    supervisor = SupervisorAgent(llm=mock_llm)
    supervisor._initialized = True

    # LLM returns garbage
    supervisor.generate_with_llm = AsyncMock(return_value="not valid json at all")

    decision = await supervisor.route_query("How to grow tomatoes?")
    # Should not crash; should fall back to rule-based and route to agronomy
    assert decision.agent_name == "agronomy_agent"
    assert decision.reasoning == "Rule-based routing"
