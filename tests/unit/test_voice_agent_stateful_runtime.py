"""Unit tests for shared-state persistence in the Sprint 10 voice agent."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agents.voice.agent import VoiceAgent
from src.memory.state_manager import AgentStateManager, VoiceSessionState, VoiceTurn
from src.voice.entity_extractor import ExtractionResult, VoiceIntent
from src.voice.orchestration import VoiceOrchestrationResult


class StubTts:
    async def synthesize(self, text: str, language: str):
        del text, language
        return SimpleNamespace(audio=b"audio-bytes")


class StubExtractor:
    def __init__(self, extraction: ExtractionResult) -> None:
        self._extraction = extraction

    async def extract(self, text: str, language: str, context_intent: str = "") -> ExtractionResult:
        del text, language, context_intent
        return self._extraction


class StubOrchestrator:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def handle_turn(
        self,
        *,
        text: str,
        language: str,
        user_id: str,
        session_id: str,
        workflow_context: dict[str, object] | None = None,
        extraction=None,
    ):
        self.calls.append(
            {
                "text": text,
                "language": language,
                "user_id": user_id,
                "session_id": session_id,
                "workflow_context": dict(workflow_context or {}),
                "extraction": extraction,
            }
        )
        return VoiceOrchestrationResult(
            response_text="Listing captured. I saved 100 kg of tomato for you.",
            persona="Priya",
            agent_name="priya_farmer_assistant",
            routed_intent=VoiceIntent.CREATE_LISTING.value,
            tools_used=["listing_create"],
            workflow_updates={"last_listing_id": "lst-900"},
        )


@pytest.mark.asyncio
async def test_voice_agent_hydrates_and_persists_shared_session_state() -> None:
    manager = AgentStateManager(redis_url=None)
    session = await manager.create_session(user_id="farmer-1", session_id="voice-agent-1")
    await manager.update_active_workflow(
        session.session_id,
        {
            "pending_intent": VoiceIntent.CREATE_LISTING.value,
            "pending_listing": {"crop": "tomato"},
        },
    )
    await manager.append_recent_voice_turn(
        session.session_id,
        VoiceTurn(
            turn_id="turn-previous",
            user_text="I want to sell tomato",
            assistant_text="How many kg of tomato do you want to sell?",
            language="en",
        ),
    )

    orchestrator = StubOrchestrator()
    agent = VoiceAgent(
        stt=None,
        tts=StubTts(),
        entity_extractor=StubExtractor(
            ExtractionResult(
                VoiceIntent.UNKNOWN,
                {"quantity": 100, "unit": "kg"},
                0.84,
                "100 kg",
                "en",
            )
        ),
        state_manager=manager,
        orchestrator=orchestrator,
    )

    response = await agent.handle_text_input(
        "100 kg",
        user_id="farmer-1",
        session_id=session.session_id,
        language="en",
    )

    updated = await manager.get_context(session.session_id)
    local_session = agent.get_session(session.session_id)
    states = [event.state for event in manager.get_voice_state_events(session.session_id)]

    assert response.response_text.startswith("Listing captured.")
    assert orchestrator.calls[0]["workflow_context"]["pending_listing"]["crop"] == "tomato"
    assert updated is not None
    assert updated.current_agent == "priya_farmer_assistant"
    assert updated.active_workflow["last_listing_id"] == "lst-900"
    assert updated.active_workflow["voice_persona"] == "Priya"
    assert updated.recent_turns[-1].user_text == "100 kg"
    assert updated.recent_turns[-1].assistant_text == response.response_text
    assert local_session is not None
    assert local_session.history[0]["user"] == "I want to sell tomato"
    assert local_session.history[-1]["user"] == "100 kg"
    assert states == [
        VoiceSessionState.THINKING,
        VoiceSessionState.SPEAKING,
        VoiceSessionState.IDLE,
    ]


@pytest.mark.asyncio
async def test_voice_agent_persists_group_speaker_metadata() -> None:
    manager = AgentStateManager(redis_url=None)
    await manager.create_session(user_id="farmer-2", session_id="voice-agent-speaker-1")

    agent = VoiceAgent(
        stt=None,
        tts=StubTts(),
        entity_extractor=StubExtractor(
            ExtractionResult(
                VoiceIntent.CHECK_PRICE,
                {"crop": "tomato", "location": "Kolar"},
                0.93,
                "Tomato price in Kolar",
                "en",
            )
        ),
        state_manager=manager,
        orchestrator=StubOrchestrator(),
    )

    await agent.handle_text_input(
        "Tomato price in Kolar",
        user_id="farmer-2",
        session_id="voice-agent-speaker-1",
        language="en",
        speaker_label="Buyer Desk",
        speaker_role="buyer",
    )

    context = await manager.get_context("voice-agent-speaker-1")

    assert context is not None
    assert context.active_speaker_id == "speaker:buyer-desk"
    assert context.speaker_profiles["speaker:buyer-desk"].role == "buyer"
    assert context.recent_turns[-1].speaker_label == "Buyer Desk"
    assert context.recent_turns[-1].speaker_metadata["transport"] == "voice_agent"
