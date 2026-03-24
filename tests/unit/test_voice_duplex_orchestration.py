"""Focused websocket tests for the Sprint 10 duplex orchestration path."""

from __future__ import annotations

import base64
import importlib
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.memory.state_manager import AgentStateManager, VoiceSessionState
from src.voice.duplex import PipelineState

voice_router_module = importlib.import_module("src.api.websocket.voice_pkg.router")


class StubDuplexPipeline:
    def __init__(self, **kwargs):  # noqa: ANN003
        del kwargs
        self.state = PipelineState.IDLE
        self._last_turn_timing: dict[str, float | None] = {}
        self._last_user_text = ""
        self._last_response_text = ""
        self._last_turn_id = "turn-orchestrated"
        self._callback = None
        self._cancelled = False
        self._conversation_history: list[dict[str, str]] = []

    @property
    def last_turn_timing(self):
        return dict(self._last_turn_timing)

    @property
    def last_user_text(self):
        return self._last_user_text

    @property
    def last_response_text(self):
        return self._last_response_text

    @property
    def last_turn_id(self):
        return self._last_turn_id

    def on_event(self, callback):
        self._callback = callback

    async def initialize(self) -> None:
        return None

    def load_conversation_history(self, history) -> None:
        self._conversation_history = list(history)

    def interrupt(self) -> None:
        self._cancelled = True
        self.state = PipelineState.INTERRUPTED

    def reset(self) -> None:
        self._cancelled = False
        self._last_turn_timing = {}
        self._last_user_text = ""
        self._last_response_text = ""
        self._last_turn_id = "turn-orchestrated"
        self.state = PipelineState.IDLE

    async def _emit(self, state, **data) -> None:
        self.state = state
        if self._callback is not None:
            await self._callback(SimpleNamespace(state=state, data=data))

    async def _transcribe(self, audio_bytes: bytes, language: str):
        del audio_bytes
        return "What is tomato price?", language

    async def _synthesize_sentence(
        self,
        sentence: str,
        language: str,
        start_chunk_index: int,
        is_final_sentence: bool,
    ):
        del language
        yield SimpleNamespace(
            audio_base64=base64.b64encode(b"audio").decode("utf-8"),
            format="mp3",
            sample_rate=24000,
            chunk_index=start_chunk_index,
            is_last=is_final_sentence,
            text=sentence,
            timing={},
        )

    async def close(self) -> None:
        return None


class StubVoiceOrchestrator:
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
        del extraction
        self.calls.append(
            {
                "text": text,
                "language": language,
                "user_id": user_id,
                "session_id": session_id,
                "workflow_context": dict(workflow_context or {}),
            }
        )
        return SimpleNamespace(
            response_text="Tomato in Kolar is trading near 30 rupees per kg.",
            persona="Arjun",
            agent_name="arjun_market_agent",
            routed_intent="check_price",
            tools_used=["price_api"],
            workflow_updates={
                "live_market_context": {
                    "commodity": "tomato",
                    "location": "Kolar",
                    "price_per_unit": "30.00",
                    "unit": "kg",
                }
            },
            metadata={"source": "test"},
        )


def _build_test_client(monkeypatch) -> TestClient:
    monkeypatch.setattr(voice_router_module, "DuplexPipeline", StubDuplexPipeline)
    monkeypatch.setattr(voice_router_module, "DUPLEX_AVAILABLE", True)
    monkeypatch.setattr(voice_router_module, "VAD_AVAILABLE", False)
    monkeypatch.setattr(
        voice_router_module,
        "get_settings",
        lambda: SimpleNamespace(
            voice_semantic_endpointing_enabled=False,
            voice_semantic_timeout_ms=150,
            voice_semantic_hold_max_ms=800,
            voice_heartbeat_interval_ms=10000,
            voice_dead_peer_timeout_ms=30000,
        ),
    )

    app = FastAPI()
    app.state.state_manager = AgentStateManager(redis_url=None)
    app.state.llm = None
    app.state.voice_orchestrator = StubVoiceOrchestrator()
    app.include_router(voice_router_module.router)
    return TestClient(app)


def test_duplex_websocket_uses_orchestrator_and_persists_state(monkeypatch) -> None:
    client = _build_test_client(monkeypatch)

    with client.websocket_connect("/api/v1/voice/ws/duplex?session_id=s-10&reconnect_token=token-10") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"

        ws.send_json(
            {
                "type": "audio_chunk",
                "audio_base64": base64.b64encode(b"hello").decode("utf-8"),
            }
        )
        ws.send_json({"type": "audio_end"})

        messages = []
        while True:
            payload = ws.receive_json()
            messages.append(payload)
            if payload["type"] == "response_end":
                break

        context = client.app.state.state_manager._sessions["s-10"]
        states = [
            event.state
            for event in client.app.state.state_manager.get_voice_state_events("s-10")
        ]

        assert any(msg["type"] == "response_sentence" for msg in messages)
        assert any(msg["type"] == "response_audio" for msg in messages)
        assert messages[-1]["type"] == "response_end"
        assert messages[-1]["full_text"] == "Tomato in Kolar is trading near 30 rupees per kg."
        assert messages[-1]["orchestrated"] is True
        assert context.current_agent == "arjun_market_agent"
        assert context.active_workflow["voice_persona"] == "Arjun"
        assert context.active_workflow["last_tools_used"] == "price_api"
        assert context.recent_turns[-1].assistant_text == messages[-1]["full_text"]
        assert states == [
            VoiceSessionState.IDLE,
            VoiceSessionState.LISTENING,
            VoiceSessionState.VAD_TRIGGERED,
            VoiceSessionState.TRANSCRIBING,
            VoiceSessionState.THINKING,
            VoiceSessionState.SPEAKING,
            VoiceSessionState.IDLE,
        ]


def test_duplex_websocket_persists_speaker_hints_for_group_turns(monkeypatch) -> None:
    client = _build_test_client(monkeypatch)

    with client.websocket_connect("/api/v1/voice/ws/duplex?session_id=s-11&reconnect_token=token-11") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["features"]["speaker_hints"] is True

        ws.send_json(
            {
                "type": "speaker_hint",
                "speaker_label": "Buyer Desk",
                "speaker_role": "buyer",
                "speaker_metadata": {"source": "test"},
            }
        )
        speaker_ack = ws.receive_json()
        assert speaker_ack["type"] == "speaker_ack"
        assert speaker_ack["speaker_id"] == "speaker:buyer-desk"

        ws.send_json(
            {
                "type": "audio_chunk",
                "audio_base64": base64.b64encode(b"hello").decode("utf-8"),
            }
        )
        ws.send_json({"type": "audio_end"})

        while True:
            payload = ws.receive_json()
            if payload["type"] == "response_end":
                break

        context = client.app.state.state_manager._sessions["s-11"]
        orchestrator = client.app.state.voice_orchestrator

        assert context.active_speaker_id == "speaker:buyer-desk"
        assert context.speaker_profiles["speaker:buyer-desk"].role == "buyer"
        assert context.recent_turns[-1].speaker_label == "Buyer Desk"
        assert context.recent_turns[-1].speaker_metadata["source"] == "test"
        assert orchestrator.calls[0]["workflow_context"]["active_speaker_id"] == "speaker:buyer-desk"
        assert orchestrator.calls[0]["workflow_context"]["known_speakers"] == ["speaker:buyer-desk"]
