"""Unit tests for the Sprint 10 voice orchestration service."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.memory.state_manager import AgentStateManager
from src.tools.registry import ToolResult
from src.voice.entity_extractor import ExtractionResult, VoiceIntent
from src.voice.orchestration import VoiceOrchestrator


class StubToolRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def execute(self, tool_name: str, **kwargs):
        self.calls.append((tool_name, kwargs))
        return ToolResult(
            tool_name=tool_name,
            success=True,
            result={
                "canonical_rates": [
                    {
                        "modal_price": 28,
                        "unit": "kg",
                        "location_label": "Kolar",
                        "source": "agmarknet",
                        "price_date": "2026-03-23",
                    }
                ]
            },
        )


class StubListingService:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []

    async def create_listing(self, **kwargs):
        self.created.append(kwargs)
        return {"id": "lst-123"}

    async def get_farmer_listings(self, farmer_id: str):
        del farmer_id
        return [{"commodity": "tomato", "quantity_kg": 100}]


class StubProcessingAgent:
    def __init__(self, content: str, tools_used: list[str], confidence: float) -> None:
        self._content = content
        self._tools_used = tools_used
        self._confidence = confidence
        self.calls: list[tuple[str, dict[str, object] | None]] = []

    async def process(self, text: str, context: dict | None = None):
        self.calls.append((text, context))
        return SimpleNamespace(
            content=self._content,
            tools_used=self._tools_used,
            confidence=self._confidence,
        )


class StubSupervisor:
    def __init__(self, routed_agent_name: str, reasoning: str, agent) -> None:
        self._routed_agent_name = routed_agent_name
        self._reasoning = reasoning
        self._agents = {routed_agent_name: agent}
        self._fallback_agent = None

    def _route_rule_based(self, text: str):
        del text
        return SimpleNamespace(
            agent_name=self._routed_agent_name,
            reasoning=self._reasoning,
        )


@pytest.mark.asyncio
async def test_price_turn_uses_price_api_and_updates_workflow() -> None:
    tool_registry = StubToolRegistry()
    orchestrator = VoiceOrchestrator(tool_registry=tool_registry)

    outcome = await orchestrator.handle_turn(
        text="What is tomato price in Kolar?",
        language="en",
        user_id="farmer-1",
        session_id="voice-price-1",
        workflow_context={},
        extraction=ExtractionResult(
            VoiceIntent.CHECK_PRICE,
            {"crop": "tomato", "location": "Kolar"},
            0.91,
            "What is tomato price in Kolar?",
            "en",
        ),
    )

    assert outcome is not None
    assert "28" in outcome.response_text
    assert outcome.tools_used == ["price_api"]
    assert outcome.workflow_updates["live_market_context"]["source"] == "agmarknet"
    assert tool_registry.calls == [
        ("price_api", {"commodity": "tomato", "district": "Kolar", "market": "Kolar"})
    ]


@pytest.mark.asyncio
async def test_listing_turn_reuses_pending_workflow_until_completion() -> None:
    listing_service = StubListingService()
    orchestrator = VoiceOrchestrator(listing_service=listing_service)

    first = await orchestrator.handle_turn(
        text="I want to list tomato",
        language="en",
        user_id="farmer-2",
        session_id="voice-listing-1",
        workflow_context={},
        extraction=ExtractionResult(
            VoiceIntent.CREATE_LISTING,
            {"crop": "tomato"},
            0.88,
            "I want to list tomato",
            "en",
        ),
    )
    second = await orchestrator.handle_turn(
        text="100 kg",
        language="en",
        user_id="farmer-2",
        session_id="voice-listing-1",
        workflow_context=first.workflow_updates,
        extraction=ExtractionResult(
            VoiceIntent.UNKNOWN,
            {"quantity": 100, "unit": "kg"},
            0.71,
            "100 kg",
            "en",
        ),
    )
    third = await orchestrator.handle_turn(
        text="25 rupees",
        language="en",
        user_id="farmer-2",
        session_id="voice-listing-1",
        workflow_context=second.workflow_updates,
        extraction=ExtractionResult(
            VoiceIntent.UNKNOWN,
            {"asking_price": 25},
            0.74,
            "25 rupees",
            "en",
        ),
    )

    assert first is not None
    assert second is not None
    assert third is not None
    assert "How many" in first.response_text
    assert "asking price" in second.response_text
    assert "Listing ID: lst-123" in third.response_text
    assert third.tools_used == ["listing_create"]
    assert "pending_intent" not in third.workflow_updates
    assert listing_service.created[0]["commodity"] == "tomato"
    assert listing_service.created[0]["quantity_kg"] == 100.0


@pytest.mark.asyncio
async def test_logistics_route_delegates_to_logistics_agent() -> None:
    logistics_agent = StubProcessingAgent(
        content="Pickup from Kolar and deliver to Bengaluru tomorrow morning.",
        tools_used=["route_optimizer"],
        confidence=0.92,
    )
    supervisor = StubSupervisor(
        routed_agent_name="logistics_agent",
        reasoning="Detected logistics intent",
        agent=logistics_agent,
    )
    orchestrator = VoiceOrchestrator(
        supervisor=supervisor,
        logistics_agent=logistics_agent,
    )

    outcome = await orchestrator.handle_turn(
        text="Need delivery from Kolar to Bengaluru",
        language="en",
        user_id="farmer-3",
        session_id="voice-logistics-1",
        workflow_context={},
        extraction=ExtractionResult(
            VoiceIntent.UNKNOWN,
            {"pickup": "Kolar", "destination": "Bengaluru"},
            0.62,
            "Need delivery from Kolar to Bengaluru",
            "en",
        ),
    )

    assert outcome is not None
    assert outcome.agent_name == "ravi_logistics_agent"
    assert outcome.tools_used == ["route_optimizer"]
    assert "Pickup from Kolar" in outcome.response_text
    assert logistics_agent.calls[0][1]["entities"]["pickup"] == "Kolar"


@pytest.mark.asyncio
async def test_logistics_route_includes_persisted_speaker_context() -> None:
    manager = AgentStateManager(redis_url=None)
    await manager.create_session(user_id="farmer-3", session_id="voice-logistics-speaker")
    await manager.update_active_speaker(
        "voice-logistics-speaker",
        speaker_label="Buyer Desk",
        speaker_role="buyer",
        speaker_metadata={"source": "test"},
    )

    logistics_agent = StubProcessingAgent(
        content="Pickup from Kolar and deliver to Bengaluru tomorrow morning.",
        tools_used=["route_optimizer"],
        confidence=0.92,
    )
    supervisor = StubSupervisor(
        routed_agent_name="logistics_agent",
        reasoning="Detected logistics intent",
        agent=logistics_agent,
    )
    orchestrator = VoiceOrchestrator(
        supervisor=supervisor,
        logistics_agent=logistics_agent,
        state_manager=manager,
    )

    outcome = await orchestrator.handle_turn(
        text="Need delivery from Kolar to Bengaluru",
        language="en",
        user_id="farmer-3",
        session_id="voice-logistics-speaker",
        workflow_context={},
        extraction=ExtractionResult(
            VoiceIntent.UNKNOWN,
            {"pickup": "Kolar", "destination": "Bengaluru"},
            0.62,
            "Need delivery from Kolar to Bengaluru",
            "en",
        ),
    )

    assert outcome is not None
    assert logistics_agent.calls[0][1]["speaker_context"]["active_speaker_id"] == "speaker:buyer-desk"
    assert logistics_agent.calls[0][1]["speaker_context"]["known_speakers"][0]["role"] == "buyer"


@pytest.mark.asyncio
async def test_fallback_route_delegates_to_supervisor_agent() -> None:
    fallback_agent = StubProcessingAgent(
        content="CropFresh can help with that after I gather a few more details.",
        tools_used=["faq_lookup"],
        confidence=0.61,
    )
    supervisor = StubSupervisor(
        routed_agent_name="general_agent",
        reasoning="General support request",
        agent=fallback_agent,
    )
    orchestrator = VoiceOrchestrator(supervisor=supervisor)

    outcome = await orchestrator.handle_turn(
        text="Explain how CropFresh works",
        language="en",
        user_id="farmer-4",
        session_id="voice-fallback-1",
        workflow_context={},
        extraction=ExtractionResult(
            VoiceIntent.UNKNOWN,
            {},
            0.55,
            "Explain how CropFresh works",
            "en",
        ),
    )

    assert outcome is not None
    assert outcome.agent_name == "admin_supervisor"
    assert outcome.tools_used == ["faq_lookup"]
    assert "CropFresh can help" in outcome.response_text
