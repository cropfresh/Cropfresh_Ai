"""API tests for speaker-aware voice REST processing."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.rest import voice as voice_rest
from src.memory.state_manager import AgentStateManager


class StubVoiceAgent:
    def __init__(self) -> None:
        self.state_manager = AgentStateManager(redis_url=None)
        self.calls: list[dict[str, object]] = []

    async def process_voice(
        self,
        *,
        audio: bytes,
        user_id: str,
        session_id: str | None = None,
        language: str = "auto",
        speaker_id: str | None = None,
        speaker_label: str | None = None,
        speaker_role: str | None = None,
    ):
        del audio
        session_id = session_id or "voice-rest-speaker-1"
        await self.state_manager.ensure_session(session_id, user_id=user_id)
        await self.state_manager.update_active_speaker(
            session_id,
            speaker_id=speaker_id,
            speaker_label=speaker_label,
            speaker_role=speaker_role,
            speaker_metadata={"source": "rest_test"},
        )
        await self.state_manager.update_active_workflow(
            session_id,
            {"pending_intent": "check_price"},
        )
        self.calls.append(
            {
                "user_id": user_id,
                "session_id": session_id,
                "language": language,
                "speaker_id": speaker_id,
                "speaker_label": speaker_label,
                "speaker_role": speaker_role,
            }
        )
        return SimpleNamespace(
            transcription="Tomato price in Kolar",
            detected_language="en",
            intent="check_price",
            entities={"crop": "tomato", "location": "Kolar"},
            response_text="Tomato in Kolar is around 30 rupees per kg.",
            response_audio=b"audio",
            session_id=session_id,
            confidence=0.94,
        )

    def get_session(self, session_id: str):
        del session_id
        return None


def test_voice_process_returns_speaker_context() -> None:
    app = FastAPI()
    app.state.voice_agent = StubVoiceAgent()
    app.include_router(voice_rest.router)
    client = TestClient(app)

    response = client.post(
        "/api/v1/voice/process",
        data={
            "user_id": "farmer-1",
            "session_id": "voice-rest-speaker-1",
            "language": "en",
            "speaker_label": "Buyer Desk",
            "speaker_role": "buyer",
        },
        files={"audio": ("sample.wav", b"fake-audio", "audio/wav")},
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["workflow_context"]["active_speaker_id"] == "speaker:buyer-desk"
    assert payload["workflow_context"]["known_speakers"] == ["speaker:buyer-desk"]
    assert app.state.voice_agent.calls[0]["speaker_label"] == "Buyer Desk"
    assert app.state.voice_agent.calls[0]["speaker_role"] == "buyer"
