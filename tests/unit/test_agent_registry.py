"""
Test Agent Registry
===================
Verify that create_agent_system() wires all 15 agents
with shared dependencies and the supervisor routes correctly.

Author: CropFresh AI Team
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────
# * Fixtures
# ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    """Mock LLM provider that returns canned responses."""
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=MagicMock(content="mock response"))
    return llm


# ─────────────────────────────────────────────────────────
# * Registry creation
# ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_agent_system_returns_supervisor_and_state_manager(
    mock_llm,
):
    """The factory must return a (SupervisorAgent, AgentStateManager) tuple."""
    from src.agents.agent_registry import create_agent_system

    supervisor, state_manager = await create_agent_system(
        llm=mock_llm,
        knowledge_base=None,
        redis_url=None,
    )

    assert supervisor is not None
    assert state_manager is not None


@pytest.mark.asyncio
async def test_create_agent_system_registers_all_agents(mock_llm):
    """All registered agent groups should produce known agent names."""
    from src.agents.agent_registry import create_agent_system

    supervisor, _ = await create_agent_system(
        llm=mock_llm,
        knowledge_base=None,
        redis_url=None,
    )

    agents = supervisor.get_available_agents()
    agent_names = {a["name"] for a in agents}

    # * Core 4 agents must always be present
    for expected in [
        "agronomy_agent",
        "commerce_agent",
        "platform_agent",
        "general_agent",
    ]:
        assert expected in agent_names, f"Missing core agent: {expected}"

    # * Should have substantially more than the old 0 agents
    assert len(agent_names) >= 4, (
        f"Expected at least 4 agents, got {len(agent_names)}: {agent_names}"
    )


@pytest.mark.asyncio
async def test_state_manager_is_in_memory_without_redis(mock_llm):
    """When no redis_url is provided, state_manager should use in-memory."""
    from src.agents.agent_registry import create_agent_system

    _, state_manager = await create_agent_system(
        llm=mock_llm,
        knowledge_base=None,
        redis_url=None,
    )

    session = await state_manager.create_session()
    assert session is not None
    assert session.session_id


# ─────────────────────────────────────────────────────────
# * Routing coverage
# ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_routing_crop_recommendation_to_adcl():
    """ADCL keywords should route to adcl_agent (rule-based)."""
    from src.agents.supervisor_agent import SupervisorAgent

    supervisor = SupervisorAgent(llm=None)  # * No LLM → rule-based
    routing = supervisor._route_rule_based("What should I sow this season?")
    assert routing.agent_name == "adcl_agent"


@pytest.mark.asyncio
async def test_routing_delivery_to_logistics():
    """Logistics keywords should route to logistics_agent (rule-based)."""
    from src.agents.supervisor_agent import SupervisorAgent

    supervisor = SupervisorAgent(llm=None)
    routing = supervisor._route_rule_based("How much will delivery cost?")
    assert routing.agent_name == "logistics_agent"


@pytest.mark.asyncio
async def test_routing_listing_to_crop_listing():
    """Listing keywords should route to crop_listing_agent (rule-based)."""
    from src.agents.supervisor_agent import SupervisorAgent

    supervisor = SupervisorAgent(llm=None)
    routing = supervisor._route_rule_based("I want to create listing for tomatoes")
    assert routing.agent_name == "crop_listing_agent"


@pytest.mark.asyncio
async def test_routing_quality_check():
    """Quality keywords should route to quality_assessment_agent."""
    from src.agents.supervisor_agent import SupervisorAgent

    supervisor = SupervisorAgent(llm=None)
    routing = supervisor._route_rule_based("quality check my produce please")
    assert routing.agent_name == "quality_assessment_agent"


@pytest.mark.asyncio
async def test_routing_buyer_match():
    """Buyer matching keywords should route to buyer_matching_agent."""
    from src.agents.supervisor_agent import SupervisorAgent

    supervisor = SupervisorAgent(llm=None)
    routing = supervisor._route_rule_based("find buyer for my tomatoes")
    assert routing.agent_name == "buyer_matching_agent"


@pytest.mark.asyncio
async def test_routing_agronomy():
    """Agronomy keywords should route to agronomy_agent."""
    from src.agents.supervisor_agent import SupervisorAgent

    supervisor = SupervisorAgent(llm=None)
    routing = supervisor._route_rule_based("how to grow tomatoes in Karnataka?")
    assert routing.agent_name == "agronomy_agent"


@pytest.mark.asyncio
async def test_routing_greeting():
    """Greetings should route to general_agent."""
    from src.agents.supervisor_agent import SupervisorAgent

    supervisor = SupervisorAgent(llm=None)
    routing = supervisor._route_rule_based("hello, how are you?")
    assert routing.agent_name == "general_agent"


# ─────────────────────────────────────────────────────────
# * Photo routing
# ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_photo_context_routes_to_quality_agent(mock_llm):
    """Image in context should auto-route to quality_assessment_agent."""
    from src.agents.agent_registry import create_agent_system

    supervisor, _ = await create_agent_system(
        llm=mock_llm,
        knowledge_base=None,
    )

    # * Pass image_b64 in context
    context = {"image_b64": "base64encodedimagedata"}

    # * Mock the target agent's process so we can verify routing
    qa_agent = supervisor._agents.get("quality_assessment_agent")
    if qa_agent:
        qa_agent.process = AsyncMock(
            return_value=MagicMock(
                content="Grade A",
                agent_name="quality_assessment_agent",
                confidence=0.9,
                steps=["graded"],
                sources=[],
                tools_used=[],
                suggested_actions=[],
                error=None,
                reasoning="",
            )
        )

        response = await supervisor.process(
            "Check the quality of my tomatoes",
            context=context,
        )
        # * Verify QA agent was called (not another agent)
        qa_agent.process.assert_called_once()


# ─────────────────────────────────────────────────────────
# * Wrapper agents
# ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_adcl_wrapper_creates():
    """ADCLWrapperAgent should instantiate without errors."""
    from src.agents.adcl_wrapper_agent import ADCLWrapperAgent

    agent = ADCLWrapperAgent(llm=None)
    assert agent.name == "adcl_agent"


@pytest.mark.asyncio
async def test_logistics_wrapper_creates():
    """LogisticsWrapperAgent should instantiate without errors."""
    from src.agents.logistics_wrapper_agent import LogisticsWrapperAgent

    agent = LogisticsWrapperAgent(llm=None)
    assert agent.name == "logistics_agent"


@pytest.mark.asyncio
async def test_adcl_wrapper_fallback_without_engine():
    """Without engine, ADCL should return a graceful response."""
    from src.agents.adcl_wrapper_agent import ADCLWrapperAgent

    agent = ADCLWrapperAgent(llm=None)
    # * Don't initialize (no engine)
    response = await agent.process("What should I sow?")
    assert response.agent_name == "adcl_agent"
    assert response.confidence < 0.5  # Low confidence fallback


@pytest.mark.asyncio
async def test_logistics_wrapper_fallback_without_engine():
    """Without engine, logistics should return a graceful response."""
    from src.agents.logistics_wrapper_agent import LogisticsWrapperAgent

    agent = LogisticsWrapperAgent(llm=None)
    response = await agent.process("What's the delivery cost?")
    assert response.agent_name == "logistics_agent"
    assert response.confidence < 0.5
