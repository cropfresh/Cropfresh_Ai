"""Shared models for Sprint 10 voice orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.voice.entity_extractor import VoiceIntent


@dataclass
class VoiceWorkflowSession:
    """Minimal session shape shared by REST and duplex voice paths."""

    user_id: str
    language: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VoiceRoute:
    """Voice-facing route decision."""

    persona: str
    voice_agent_name: str
    downstream_target: str
    intent: str
    reasoning: str


@dataclass
class VoiceOrchestrationResult:
    """Structured outcome from the voice orchestration service."""

    response_text: str
    persona: str
    agent_name: str
    routed_intent: str
    tools_used: list[str] = field(default_factory=list)
    workflow_updates: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pending_intent(self) -> str | None:
        value = self.workflow_updates.get("pending_intent")
        return str(value) if value else None


VOICE_PERSONAS = {
    VoiceIntent.CHECK_PRICE.value: "Arjun",
    VoiceIntent.CREATE_LISTING.value: "Priya",
    VoiceIntent.MY_LISTINGS.value: "Priya",
    "logistics": "Ravi",
    "fallback": "Admin",
}
