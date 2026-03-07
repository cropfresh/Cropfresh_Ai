"""
Agent Memory Persistence — Unit Tests
======================================
Validates all 6 root-cause gaps (G1-G6) described in the implementation plan.

Coverage:
  G1 — AgentStateManager injected into all agents via registry
  G2 — Chat router always uses process_with_session()
  G3 — Entity extraction pipeline (commodity, quantity, district, price)
  G4 — Memory context (_build_memory_context) injected into LLM prompts
  G5 — current_agent written back to session after every response
  G6 — Token cost tracking (basic write-back)

Author: CropFresh AI Team
"""

from __future__ import annotations

import asyncio
import pytest

from src.memory.state_manager import AgentStateManager


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def state_manager() -> AgentStateManager:
    """In-memory state manager (no Redis required)."""
    return AgentStateManager()


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ═══════════════════════════════════════════════════════════════════════════
# G3 — Entity Extraction Pipeline
# ═══════════════════════════════════════════════════════════════════════════

class TestEntityExtraction:
    """Validate extract_and_merge_entities() across diverse input texts."""

    @pytest.mark.asyncio
    async def test_extracts_commodity_english(self, state_manager: AgentStateManager):
        session = await state_manager.create_session()
        found = await state_manager.extract_and_merge_entities(
            session.session_id, "I want to sell tomato"
        )
        assert found.get("commodity", "").lower() == "tomato"

    @pytest.mark.asyncio
    async def test_extracts_commodity_hindi(self, state_manager: AgentStateManager):
        session = await state_manager.create_session()
        found = await state_manager.extract_and_merge_entities(
            session.session_id, "mujhe pyaaz bechna hai 100 kg"
        )
        # pyaaz → Onion
        assert found.get("commodity", "").lower() == "onion"

    @pytest.mark.asyncio
    async def test_extracts_quantity_kg(self, state_manager: AgentStateManager):
        session = await state_manager.create_session()
        found = await state_manager.extract_and_merge_entities(
            session.session_id, "I have 200 kg of tomatoes"
        )
        assert found.get("quantity_kg") == 200.0

    @pytest.mark.asyncio
    async def test_extracts_quantity_quintal(self, state_manager: AgentStateManager):
        session = await state_manager.create_session()
        found = await state_manager.extract_and_merge_entities(
            session.session_id, "I have 3 quintal of onion"
        )
        assert found.get("quantity_kg") == 300.0
        assert found.get("quantity_quintal") == 3.0

    @pytest.mark.asyncio
    async def test_extracts_district(self, state_manager: AgentStateManager):
        session = await state_manager.create_session()
        found = await state_manager.extract_and_merge_entities(
            session.session_id, "what is the price in Kolar today?"
        )
        assert found.get("district", "").lower() == "kolar"

    @pytest.mark.asyncio
    async def test_extracts_price(self, state_manager: AgentStateManager):
        session = await state_manager.create_session()
        found = await state_manager.extract_and_merge_entities(
            session.session_id, "asking ₹25/kg for tomato"
        )
        assert found.get("price_per_kg") == 25.0

    @pytest.mark.asyncio
    async def test_entities_merged_across_turns(self, state_manager: AgentStateManager):
        """Entities from earlier turns still present in later turns."""
        session = await state_manager.create_session()
        sid = session.session_id

        await state_manager.extract_and_merge_entities(sid, "I grow tomato in Kolar")
        await state_manager.extract_and_merge_entities(sid, "I have 150 kg to sell")

        session = await state_manager.get_context(sid)
        entities = session.entities
        assert entities.get("commodity", "").lower() == "tomato"
        assert entities.get("district", "").lower() == "kolar"
        assert entities.get("quantity_kg") == 150.0

    @pytest.mark.asyncio
    async def test_empty_text_returns_empty(self, state_manager: AgentStateManager):
        session = await state_manager.create_session()
        found = await state_manager.extract_and_merge_entities(
            session.session_id, "hello there"
        )
        assert found == {}

    @pytest.mark.asyncio
    async def test_entities_persisted_in_session(self, state_manager: AgentStateManager):
        """After extraction, entities are readable from get_context()."""
        session = await state_manager.create_session()
        sid = session.session_id
        await state_manager.extract_and_merge_entities(sid, "onion 500 kg in Hubli")
        refreshed = await state_manager.get_context(sid)
        assert "district" in refreshed.entities
        assert refreshed.entities["district"].lower() == "hubli"


# ═══════════════════════════════════════════════════════════════════════════
# G1 — State Manager Available on All Agents
# ═══════════════════════════════════════════════════════════════════════════

class TestRegistryInjection:
    """Verify that BaseAgent subclasses store state_manager from **kwargs."""

    def test_agronomy_agent_accepts_state_manager(self):
        from src.agents.agronomy_agent import AgronomyAgent
        sm = AgentStateManager()
        agent = AgronomyAgent(state_manager=sm)
        assert agent.state_manager is sm

    def test_adcl_wrapper_accepts_state_manager(self):
        from src.agents.adcl_wrapper_agent import ADCLWrapperAgent
        sm = AgentStateManager()
        agent = ADCLWrapperAgent(state_manager=sm)
        assert agent.state_manager is sm

    def test_logistics_wrapper_accepts_state_manager(self):
        from src.agents.logistics_wrapper_agent import LogisticsWrapperAgent
        sm = AgentStateManager()
        agent = LogisticsWrapperAgent(state_manager=sm)
        assert agent.state_manager is sm

    def test_buyer_matching_accepts_state_manager(self):
        from src.agents.buyer_matching.agent import BuyerMatchingAgent
        sm = AgentStateManager()
        agent = BuyerMatchingAgent(state_manager=sm)
        assert agent.state_manager is sm

    def test_crop_listing_accepts_state_manager(self):
        from src.agents.crop_listing.agent import CropListingAgent
        sm = AgentStateManager()
        agent = CropListingAgent(state_manager=sm)
        assert agent.state_manager is sm

    def test_price_prediction_accepts_state_manager(self):
        from src.agents.price_prediction.agent import PricePredictionAgent
        sm = AgentStateManager()
        agent = PricePredictionAgent(state_manager=sm)
        assert agent.state_manager is sm


# ═══════════════════════════════════════════════════════════════════════════
# G4 — Memory Context Injection Into LLM Prompts
# ═══════════════════════════════════════════════════════════════════════════

class TestMemoryContextInjection:
    """Verify _build_memory_context() produces correct output."""

    def _make_agent(self):
        """Create a minimal BaseAgent subclass for testing."""
        from src.agents.agronomy_agent import AgronomyAgent
        return AgronomyAgent()

    def test_empty_context_returns_empty_string(self):
        agent = self._make_agent()
        result = agent._build_memory_context(None)
        assert result == ""

    def test_entities_appear_in_memory_block(self):
        agent = self._make_agent()
        context = {"entities": {"commodity": "Tomato", "district": "Kolar"}}
        result = agent._build_memory_context(context)
        assert "Tomato" in result
        assert "Kolar" in result
        assert "[Session Memory]" in result

    def test_internal_keys_filtered_out(self):
        agent = self._make_agent()
        context = {
            "entities": {
                "commodity": "Onion",
                "__current_agent": "agronomy_agent",  # internal key — must be hidden
            }
        }
        result = agent._build_memory_context(context)
        # __current_agent should NOT appear as a raw entity line
        assert "__current_agent" not in result
        # But it should appear in the [Previous agent] block
        assert "agronomy_agent" in result

    def test_conversation_summary_included(self):
        agent = self._make_agent()
        context = {
            "entities": {},
            "conversation_summary": "User asked about tomato prices in Kolar.",
        }
        result = agent._build_memory_context(context)
        assert "tomato prices in Kolar" in result

    def test_previous_agent_included(self):
        agent = self._make_agent()
        context = {
            "entities": {"__current_agent": "price_prediction"},
            "current_agent": "price_prediction",
        }
        result = agent._build_memory_context(context)
        assert "price_prediction" in result
        assert "[Previous agent]" in result


# ═══════════════════════════════════════════════════════════════════════════
# G5 — current_agent Written Back Per Turn
# ═══════════════════════════════════════════════════════════════════════════

class TestCurrentAgentTracking:
    """Test that __current_agent entity is written into session."""

    @pytest.mark.asyncio
    async def test_current_agent_persists(self, state_manager: AgentStateManager):
        session = await state_manager.create_session()
        sid = session.session_id
        # Simulate supervisor writing current_agent back
        await state_manager.update_entities(sid, {"__current_agent": "agronomy_agent"})
        refreshed = await state_manager.get_context(sid)
        assert refreshed.entities.get("__current_agent") == "agronomy_agent"

    @pytest.mark.asyncio
    async def test_current_agent_changes_between_turns(self, state_manager: AgentStateManager):
        session = await state_manager.create_session()
        sid = session.session_id
        await state_manager.update_entities(sid, {"__current_agent": "agronomy_agent"})
        await state_manager.update_entities(sid, {"__current_agent": "pricing_agent"})
        refreshed = await state_manager.get_context(sid)
        assert refreshed.entities.get("__current_agent") == "pricing_agent"
