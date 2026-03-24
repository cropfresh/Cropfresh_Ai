"""Unit tests for the Sprint 10 shared voice state machine."""

from __future__ import annotations

import pytest

from src.memory.state_manager import (
    AgentStateManager,
    VoicePlaybackState,
    VoiceSessionState,
)


class StubRedis:
    """Minimal async Redis stub for state-manager pubsub tests."""

    def __init__(self) -> None:
        self.storage: dict[str, str] = {}
        self.published: list[tuple[str, str]] = []

    async def setex(self, key: str, ttl: int, value: str) -> None:
        del ttl
        self.storage[key] = value

    async def get(self, key: str) -> str | None:
        return self.storage.get(key)

    async def delete(self, key: str) -> None:
        self.storage.pop(key, None)

    async def publish(self, channel: str, payload: str) -> None:
        self.published.append((channel, payload))


@pytest.mark.asyncio
async def test_voice_state_transitions_record_events_and_publish_updates() -> None:
    manager = AgentStateManager(redis_url=None)
    manager._redis_client = StubRedis()
    session = await manager.create_session(user_id="farmer-1", session_id="voice-state-1")

    await manager.transition_voice_state(
        session.session_id,
        VoiceSessionState.LISTENING,
        source="test",
        reason="audio_chunk",
    )
    await manager.transition_voice_state(
        session.session_id,
        VoiceSessionState.VAD_TRIGGERED,
        source="test",
        reason="vad_end",
    )
    await manager.transition_voice_state(
        session.session_id,
        VoiceSessionState.TRANSCRIBING,
        source="test",
        reason="stt_start",
    )

    context = await manager.get_context(session.session_id)
    events = manager.get_voice_state_events(session.session_id)

    assert context is not None
    assert context.voice_state == VoiceSessionState.TRANSCRIBING
    assert context.playback_state == VoicePlaybackState.TRANSCRIBING
    assert [event.state for event in events] == [
        VoiceSessionState.LISTENING,
        VoiceSessionState.VAD_TRIGGERED,
        VoiceSessionState.TRANSCRIBING,
    ]
    assert len(manager._redis_client.published) == 6
    assert {
        channel for channel, _ in manager._redis_client.published
    } == {"voice:state", f"voice:state:{session.session_id}"}


@pytest.mark.asyncio
async def test_voice_state_rejects_invalid_transition() -> None:
    manager = AgentStateManager(redis_url=None)
    session = await manager.create_session(user_id="farmer-2")

    with pytest.raises(ValueError):
        await manager.transition_voice_state(
            session.session_id,
            VoiceSessionState.SPEAKING,
            source="test",
            reason="skip_pipeline",
        )


@pytest.mark.asyncio
async def test_update_active_workflow_replaces_voice_context() -> None:
    manager = AgentStateManager(redis_url=None)
    session = await manager.create_session(user_id="farmer-3")

    assert await manager.update_active_workflow(
        session.session_id,
        {"pending_intent": "create_listing", "pending_listing": {"crop": "tomato"}},
    )
    assert await manager.update_active_workflow(
        session.session_id,
        {"last_listing_id": "lst-10"},
    )

    context = await manager.get_context(session.session_id)
    assert context is not None
    assert context.active_workflow == {"last_listing_id": "lst-10"}
