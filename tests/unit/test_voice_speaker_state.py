"""Unit tests for speaker-aware voice session state."""

from __future__ import annotations

import pytest

from src.memory.state_manager import AgentStateManager, VoiceTurn


@pytest.mark.asyncio
async def test_group_speaker_profiles_are_tracked_across_turns() -> None:
    manager = AgentStateManager(redis_url=None)
    session = await manager.create_session(user_id="farmer-group", session_id="voice-speakers-1")

    first_profile = await manager.update_active_speaker(
        session.session_id,
        speaker_label="Farmer A",
        speaker_role="seller",
        speaker_metadata={"source": "test"},
    )
    await manager.append_recent_voice_turn(
        session.session_id,
        VoiceTurn(
            turn_id="turn-1",
            user_text="I have 100 kg tomato.",
            assistant_text="What price do you want to set?",
            language="en",
            speaker_id=first_profile.speaker_id if first_profile else None,
            speaker_label="Farmer A",
            speaker_role="seller",
        ),
    )

    second_profile = await manager.update_active_speaker(
        session.session_id,
        speaker_id="speaker:buyer-b",
        speaker_label="Buyer B",
        speaker_role="buyer",
        speaker_metadata={"source": "test"},
    )
    await manager.append_recent_voice_turn(
        session.session_id,
        VoiceTurn(
            turn_id="turn-2",
            user_text="What is the latest mandi price?",
            assistant_text="Kolar tomato is around 30 rupees per kg.",
            language="en",
            speaker_id=second_profile.speaker_id if second_profile else None,
            speaker_label="Buyer B",
            speaker_role="buyer",
        ),
    )

    context = await manager.get_context(session.session_id)

    assert context is not None
    assert context.active_speaker_id == "speaker:buyer-b"
    assert sorted(context.speaker_profiles.keys()) == [
        "speaker:buyer-b",
        "speaker:farmer-a",
    ]
    assert context.speaker_profiles["speaker:farmer-a"].turn_count == 1
    assert context.speaker_profiles["speaker:buyer-b"].turn_count == 1
    assert context.recent_turns[-1].speaker_role == "buyer"
    assert context.messages[-2].metadata["speaker_id"] == "speaker:buyer-b"
    assert context.messages[-1].metadata["speaker_label"] == "Buyer B"
