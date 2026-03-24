"""Shared runtime builder for the Sprint 10 voice orchestrator."""

from __future__ import annotations

from typing import Any

from src.voice.orchestration import VoiceOrchestrator


def build_voice_orchestrator(
    *,
    supervisor: Any = None,
    state_manager: Any = None,
    tool_registry: Any = None,
    listing_service: Any = None,
    logistics_agent: Any = None,
    llm_provider: Any = None,
) -> VoiceOrchestrator:
    """Build the shared voice orchestrator used by REST and duplex voice paths."""
    return VoiceOrchestrator(
        supervisor=supervisor,
        state_manager=state_manager,
        tool_registry=tool_registry,
        listing_service=listing_service,
        logistics_agent=logistics_agent,
        llm_provider=llm_provider,
    )
