"""Unit tests for shared chat session preparation and execution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.api.chat_pkg.session import execute_chat_request
from src.memory.state_manager import AgentStateManager


def _make_response() -> SimpleNamespace:
    return SimpleNamespace(
        content="ok",
        agent_name="general_agent",
        confidence=0.8,
        sources=[],
        tools_used=[],
        steps=[],
        suggested_actions=[],
    )


@pytest.mark.asyncio
async def test_execute_chat_request_persists_language_profile_and_entities():
    state_manager = AgentStateManager()
    supervisor = SimpleNamespace(
        state_manager=state_manager,
        process_with_session=AsyncMock(return_value=_make_response()),
    )

    session_id, _ = await execute_chat_request(
        supervisor,
        "Tomato bele yaavaga?",
        context={
            "language_pref": "kannada",
            "district": "Kolar",
            "farmer_id": "farmer-123",
        },
    )

    context = await state_manager.get_context(session_id)
    assert context.user_profile["language"] == "kn"
    assert context.user_profile["language_pref"] == "kn"
    assert context.user_profile["district"] == "Kolar"
    assert context.entities["farmer_id"] == "farmer-123"
    supervisor.process_with_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_chat_request_normalizes_stateless_context():
    supervisor = SimpleNamespace(
        state_manager=None,
        process=AsyncMock(return_value=_make_response()),
    )

    session_id, _ = await execute_chat_request(
        supervisor,
        "ಟೊಮೆಟೊ ಬೆಲೆ ಎಷ್ಟು?",
        context={"farmer_id": "farmer-123"},
    )

    kwargs = supervisor.process.await_args.kwargs
    assert session_id
    assert kwargs["context"]["language"] == "kn"
    assert kwargs["context"]["response_language"] == "kn"
    assert kwargs["context"]["user_profile"]["language"] == "kn"
    assert kwargs["context"]["entities"]["farmer_id"] == "farmer-123"
