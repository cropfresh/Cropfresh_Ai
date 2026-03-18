"""
Unit tests for AgentStateManager WebRTC session rehydration (NFR6).

NFR6 requirement:
    AgentStateManager must successfully rehydrate and resume 99% of dropped
    WebRTC voice sessions within < 1.0s upon client reconnection.

Test strategy:
    All tests use in-memory mode (no Redis) for deterministic, fast execution.
    A live Redis integration test is in tests/integration/ (not run in CI).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from src.memory.state_manager import (
    AgentStateManager,
    ConversationContext,
    Message,
    SessionExpiredError,
    VoiceTurn,
)


# * ═══════════════════════════════════════════════════════════════
# * Fixtures
# * ═══════════════════════════════════════════════════════════════


@pytest.fixture
def manager() -> AgentStateManager:
    """In-memory AgentStateManager for tests (no Redis)."""
    return AgentStateManager(redis_url=None)


async def _create_session(manager: AgentStateManager, user_id: str = "farmer_test_001") -> ConversationContext:
    """Helper: create a session and add a message."""
    session = await manager.create_session(user_id=user_id)
    await manager.add_message(session.session_id, Message(role="user", content="Hello"))
    return await manager.get_context(session.session_id)


# * ═══════════════════════════════════════════════════════════════
# * NFR6: Registration
# * ═══════════════════════════════════════════════════════════════


class TestVoiceSessionRegistration:

    @pytest.mark.asyncio
    async def test_register_sets_voice_session_id(self, manager: AgentStateManager):
        """After registering, the session should carry the voice_session_id."""
        session = await manager.create_session()
        voice_id = "rtc-abc123"

        await manager.register_voice_session(session.session_id, voice_id)

        ctx = await manager.get_context(session.session_id)
        assert ctx is not None
        assert ctx.voice_session_id == voice_id

    @pytest.mark.asyncio
    async def test_register_stores_reverse_mapping(self, manager: AgentStateManager):
        """Reverse mapping voice_id → session_id must be stored in memory."""
        session = await manager.create_session()
        voice_id = "rtc-xyz987"

        await manager.register_voice_session(session.session_id, voice_id)

        assert manager._voice_sessions[voice_id] == session.session_id

    @pytest.mark.asyncio
    async def test_register_for_nonexistent_session_does_not_crash(
        self, manager: AgentStateManager
    ):
        """Registering a voice session for an unknown session_id should not raise."""
        # Should complete without exception (silently skips body)
        await manager.register_voice_session("non-existent-session", "rtc-ghost")


# * ═══════════════════════════════════════════════════════════════
# * NFR6: Rehydration — happy path
# * ═══════════════════════════════════════════════════════════════


class TestVoiceSessionRehydration:

    @pytest.mark.asyncio
    async def test_rehydrate_returns_correct_context(self, manager: AgentStateManager):
        """
        Rehydrate a freshly-registered voice session → correct context returned.
        This is the core NFR6 scenario.
        """
        session = await _create_session(manager)
        voice_id = "rtc-main-001"
        await manager.register_voice_session(session.session_id, voice_id)

        ctx = await manager.rehydrate_voice_session(voice_id)

        assert ctx is not None
        assert ctx.session_id == session.session_id
        assert ctx.user_id == "farmer_test_001"
        assert len(ctx.messages) >= 1

    @pytest.mark.asyncio
    async def test_rehydrate_unknown_voice_id_returns_none(self, manager: AgentStateManager):
        """Unknown voice_session_id → returns None (no exception)."""
        result = await manager.rehydrate_voice_session("rtc-unknown-999")
        assert result is None

    @pytest.mark.asyncio
    async def test_rehydrate_multiple_times_is_idempotent(self, manager: AgentStateManager):
        """Two rehydrations for the same voice session should both succeed."""
        session = await _create_session(manager)
        voice_id = "rtc-idempotent"
        await manager.register_voice_session(session.session_id, voice_id)

        ctx1 = await manager.rehydrate_voice_session(voice_id)
        ctx2 = await manager.rehydrate_voice_session(voice_id)

        assert ctx1 is not None
        assert ctx2 is not None
        assert ctx1.session_id == ctx2.session_id

    @pytest.mark.asyncio
    async def test_rehydrate_preserves_conversation_history(self, manager: AgentStateManager):
        """Rehydrated context must include all messages added before the drop."""
        session = await manager.create_session()
        for content in ["Tomato 50kg", "Price query", "Grade assessment"]:
            await manager.add_message(session.session_id, Message(role="user", content=content))
        voice_id = "rtc-history"
        await manager.register_voice_session(session.session_id, voice_id)

        ctx = await manager.rehydrate_voice_session(voice_id)

        assert ctx is not None
        contents = [m.content for m in ctx.messages]
        assert "Tomato 50kg" in contents
        assert "Grade assessment" in contents


# * ═══════════════════════════════════════════════════════════════
# * NFR6: Staleness / Expiry
# * ═══════════════════════════════════════════════════════════════


class TestVoiceSessionExpiry:

    @pytest.mark.asyncio
    async def test_rehydrate_fresh_session_succeeds(self, manager: AgentStateManager):
        """A session active within the last 5 minutes must rehydrate successfully."""
        session = await manager.create_session()
        voice_id = "rtc-fresh"
        await manager.register_voice_session(session.session_id, voice_id)

        ctx = await manager.rehydrate_voice_session(voice_id)
        assert ctx is not None

    @pytest.mark.asyncio
    async def test_rehydrate_expired_session_raises(self, manager: AgentStateManager):
        """A session idle for > 5 minutes should raise SessionExpiredError."""
        session = await manager.create_session()
        voice_id = "rtc-expired"
        await manager.register_voice_session(session.session_id, voice_id)

        # * Force-age the last_active_at timestamp
        ctx = manager._sessions[session.session_id]
        ctx.last_active_at = datetime.now() - timedelta(seconds=310)
        manager._sessions[session.session_id] = ctx

        with pytest.raises(SessionExpiredError) as exc_info:
            await manager.rehydrate_voice_session(voice_id)

        err = exc_info.value
        assert err.voice_session_id == voice_id
        assert err.stale_seconds >= 300.0

    @pytest.mark.asyncio
    async def test_session_expired_error_message_is_descriptive(
        self, manager: AgentStateManager
    ):
        """SessionExpiredError.__str__ should include the voice_session_id."""
        err = SessionExpiredError("rtc-test", 350.0)
        assert "rtc-test" in str(err)
        assert "350" in str(err)


# * ═══════════════════════════════════════════════════════════════
# * NFR6: Deregistration
# * ═══════════════════════════════════════════════════════════════


class TestVoiceSessionDeregistration:

    @pytest.mark.asyncio
    async def test_deregister_removes_reverse_mapping(self, manager: AgentStateManager):
        """Deregistering should remove the in-memory reverse mapping."""
        session = await manager.create_session()
        voice_id = "rtc-dereg"
        await manager.register_voice_session(session.session_id, voice_id)
        assert voice_id in manager._voice_sessions

        await manager.deregister_voice_session(voice_id)

        assert voice_id not in manager._voice_sessions

    @pytest.mark.asyncio
    async def test_rehydrate_after_deregister_returns_none(
        self, manager: AgentStateManager
    ):
        """After clean disconnect (deregister), rehydrate should return None."""
        session = await manager.create_session()
        voice_id = "rtc-clean-close"
        await manager.register_voice_session(session.session_id, voice_id)
        await manager.deregister_voice_session(voice_id)

        result = await manager.rehydrate_voice_session(voice_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_deregister_unknown_does_not_crash(self, manager: AgentStateManager):
        """Deregistering an unknown voice session should not raise."""
        await manager.deregister_voice_session("rtc-ghost-not-registered")


# * ═══════════════════════════════════════════════════════════════
# * NFR6: Touch (keepalive)
# * ═══════════════════════════════════════════════════════════════


class TestVoiceSessionTouch:

    @pytest.mark.asyncio
    async def test_touch_refreshes_last_active_at(self, manager: AgentStateManager):
        """touch_voice_session() should extend last_active_at, keeping session fresh."""
        session = await manager.create_session()
        # Age the session first
        ctx = manager._sessions[session.session_id]
        ctx.last_active_at = datetime.now() - timedelta(seconds=200)
        manager._sessions[session.session_id] = ctx

        before = manager._sessions[session.session_id].last_active_at
        await manager.touch_voice_session(session.session_id)
        after = manager._sessions[session.session_id].last_active_at

        assert after > before

    @pytest.mark.asyncio
    async def test_touch_with_heartbeat_refreshes_last_heartbeat_at(
        self, manager: AgentStateManager
    ):
        """Heartbeat touches should refresh both active and heartbeat timestamps."""
        session = await manager.create_session()
        before = manager._sessions[session.session_id].last_heartbeat_at

        await manager.touch_voice_session(session.session_id, heartbeat=True)

        after = manager._sessions[session.session_id].last_heartbeat_at
        assert after > before


class TestReconnectTokensAndRecentTurns:

    @pytest.mark.asyncio
    async def test_ensure_session_preserves_requested_session_id(self, manager: AgentStateManager):
        """ensure_session() should create a session with the supplied id when absent."""
        context = await manager.ensure_session("voice-session-fixed", user_id="farmer-fixed")

        assert context.session_id == "voice-session-fixed"
        assert context.user_id == "farmer-fixed"

    @pytest.mark.asyncio
    async def test_validate_reconnect_token_round_trip(self, manager: AgentStateManager):
        """A registered reconnect token should validate for the same session."""
        session = await manager.create_session()

        await manager.register_voice_session(
            session.session_id,
            session.session_id,
            reconnect_token="secret-token",
            transport_mode="duplex_ws",
            language="kn",
        )

        assert await manager.validate_reconnect_token(session.session_id, "secret-token") is True
        assert await manager.validate_reconnect_token(session.session_id, "wrong-token") is False

    @pytest.mark.asyncio
    async def test_recent_voice_turns_are_capped_at_last_ten(self, manager: AgentStateManager):
        """Recent turn history should trim to the last 10 reconnect-safe turns."""
        session = await manager.create_session()
        for index in range(12):
            await manager.append_recent_voice_turn(
                session.session_id,
                VoiceTurn(
                    turn_id=f"turn-{index}",
                    user_text=f"user {index}",
                    assistant_text=f"assistant {index}",
                    language="en",
                ),
            )

        context = await manager.get_context(session.session_id)
        assert context is not None
        assert len(context.recent_turns) == 10
        assert context.recent_turns[0].turn_id == "turn-2"
        assert context.recent_turns[-1].turn_id == "turn-11"


# * ═══════════════════════════════════════════════════════════════
# * NFR6: SLA enforcement (timeout)
# * ═══════════════════════════════════════════════════════════════


class TestNFR6SLATimeout:

    @pytest.mark.asyncio
    async def test_rehydrate_completes_well_within_sla(self, manager: AgentStateManager):
        """
        In-memory rehydration must complete in <100ms (way inside the 1.0s SLA).
        If Redis is available the target is <1.0s; this proves the in-memory baseline.
        """
        import time

        session = await manager.create_session(user_id="sla-farmer")
        voice_id = "rtc-sla-check"
        await manager.register_voice_session(session.session_id, voice_id)

        start = time.perf_counter()
        ctx = await manager.rehydrate_voice_session(voice_id, timeout_sec=1.0)
        elapsed = time.perf_counter() - start

        assert ctx is not None
        assert elapsed < 0.1, f"Rehydration took {elapsed:.3f}s — expected < 0.1s in memory"

    @pytest.mark.asyncio
    async def test_rehydrate_returns_none_on_timeout(self, manager: AgentStateManager):
        """
        If rehydration exceeds timeout_sec (simulated with mocked slow coroutine),
        rehydrate_voice_session() must return None rather than raising TimeoutError.
        """
        from unittest.mock import AsyncMock, patch

        session = await manager.create_session()
        voice_id = "rtc-timeout-sim"
        await manager.register_voice_session(session.session_id, voice_id)

        # * Simulate slow lookup exceeding timeout
        async def _slow_lookup(_vid):
            await asyncio.sleep(2.0)   # 2s > timeout of 0.01s
            return None

        with patch.object(manager, "_lookup_voice_session", side_effect=_slow_lookup):
            result = await manager.rehydrate_voice_session(voice_id, timeout_sec=0.01)

        assert result is None   # TimeoutError caught; None returned (not raised)
