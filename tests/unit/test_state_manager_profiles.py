"""Focused tests for state-manager user profile persistence."""

import pytest

from src.memory.state_manager import AgentStateManager


@pytest.mark.asyncio
async def test_update_user_profile_merges_new_language_fields():
    state_manager = AgentStateManager()
    session = await state_manager.create_session(user_profile={"district": "Kolar"})

    updated = await state_manager.update_user_profile(
        session.session_id,
        {"language": "kn", "language_pref": "kn"},
    )

    assert updated is True
    refreshed = await state_manager.get_context(session.session_id)
    assert refreshed.user_profile["district"] == "Kolar"
    assert refreshed.user_profile["language"] == "kn"
    assert refreshed.user_profile["language_pref"] == "kn"


@pytest.mark.asyncio
async def test_update_user_profile_returns_false_for_unknown_session():
    state_manager = AgentStateManager()
    updated = await state_manager.update_user_profile("missing-session", {"language": "kn"})
    assert updated is False
