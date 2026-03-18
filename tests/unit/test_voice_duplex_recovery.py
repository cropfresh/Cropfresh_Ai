"""Focused websocket tests for duplex reconnect and heartbeat behavior."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.websocket.voice_pkg.router as voice_router_module
from src.memory.state_manager import AgentStateManager, VoiceTurn
from src.voice.duplex import PipelineState


class StubDuplexPipeline:
    def __init__(self, **kwargs):  # noqa: ANN003
        del kwargs
        self.state = PipelineState.IDLE
        self.last_turn_timing = {}
        self.last_user_text = ""
        self.last_response_text = ""
        self.last_turn_id = "turn-stub"
        self._callback = None
        self.history = []

    def on_event(self, callback):
        self._callback = callback

    async def initialize(self) -> None:
        return None

    def load_conversation_history(self, history) -> None:
        self.history = list(history)

    async def close(self) -> None:
        return None

    def interrupt(self) -> None:
        self.state = PipelineState.INTERRUPTED


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
    app.include_router(voice_router_module.router)
    return TestClient(app)


def test_duplex_ready_payload_recovers_same_session_with_valid_token(monkeypatch) -> None:
    client = _build_test_client(monkeypatch)

    with client.websocket_connect("/api/v1/voice/ws/duplex?session_id=s-1&reconnect_token=token-1") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["session_id"] == "s-1"
        assert ready["recovered"] is False

    asyncio.run(
        client.app.state.state_manager.append_recent_voice_turn(
            "s-1",
            VoiceTurn(
                turn_id="turn-1",
                user_text="Tomato price?",
                assistant_text="Tomato is Rs 24 per kg.",
                language="en",
            ),
        )
    )

    with client.websocket_connect("/api/v1/voice/ws/duplex?session_id=s-1&reconnect_token=token-1") as ws:
        ready = ws.receive_json()
        assert ready["recovered"] is True
        assert ready["recovered_turn_count"] == 1
        assert ready["session_id"] == "s-1"


def test_duplex_reconnect_with_invalid_token_falls_back_to_fresh_session(monkeypatch) -> None:
    client = _build_test_client(monkeypatch)

    with client.websocket_connect("/api/v1/voice/ws/duplex?session_id=s-2&reconnect_token=token-2") as ws:
        ready = ws.receive_json()
        assert ready["session_id"] == "s-2"

    with client.websocket_connect("/api/v1/voice/ws/duplex?session_id=s-2&reconnect_token=wrong-token") as ws:
        ready = ws.receive_json()
        assert ready["recovered"] is False
        assert ready["session_id"] != "s-2"
        assert ready["recovery_outcome"] == "invalid_reconnect_token"


def test_duplex_heartbeat_acknowledges_keepalive(monkeypatch) -> None:
    client = _build_test_client(monkeypatch)

    with client.websocket_connect("/api/v1/voice/ws/duplex?session_id=s-3&reconnect_token=token-3") as ws:
        _ = ws.receive_json()
        ws.send_json({"type": "heartbeat"})
        ack = ws.receive_json()

        assert ack["type"] == "heartbeat_ack"
        assert ack["session_id"] == "s-3"
        assert ack["heartbeat_interval_ms"] == 10000
