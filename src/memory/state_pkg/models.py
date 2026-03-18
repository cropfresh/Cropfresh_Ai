"""
Agent State Manager data models and exceptions.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class SessionExpiredError(Exception):
    """
    Raised when a voice session has been stale longer than the
    reconnection tolerance window (default: 5 minutes).
    """
    def __init__(self, voice_session_id: str, stale_seconds: float) -> None:
        self.voice_session_id = voice_session_id
        self.stale_seconds = stale_seconds
        super().__init__(
            f"Voice session '{voice_session_id}' expired after "
            f"{stale_seconds:.1f}s of inactivity"
        )


class Message(BaseModel):
    """Single message in conversation."""

    role: str  # user, assistant, system, tool
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # For tool messages
    tool_name: Optional[str] = None
    tool_result: Optional[dict] = None


class VoicePlaybackState(str, Enum):
    """Realtime playback states tracked for reconnect-aware voice sessions."""

    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    RECOVERING = "recovering"


class VoiceTurn(BaseModel):
    """Compact voice turn record used for reconnect recovery."""

    turn_id: str
    user_text: str
    assistant_text: str
    language: str = "en"
    interrupted: bool = False
    timing: dict[str, float | None] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class ConversationContext(BaseModel):
    """Full conversation context for an agent session."""

    session_id: str
    user_id: Optional[str] = None

    messages: list[Message] = Field(default_factory=list)
    entities: dict[str, Any] = Field(default_factory=dict)
    user_profile: dict[str, Any] = Field(default_factory=dict)

    current_agent: Optional[str] = None
    agent_stack: list[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    total_input_tokens: int = 0
    total_output_tokens: int = 0

    voice_session_id: Optional[str] = None
    last_active_at: datetime = Field(default_factory=datetime.now)
    transport_mode: Optional[str] = None
    language: str = "hi"
    playback_state: VoicePlaybackState = VoicePlaybackState.IDLE
    recent_turns: list[VoiceTurn] = Field(default_factory=list)
    pending_transcript: Optional[str] = None
    pending_segment_id: Optional[str] = None
    last_turn_id: Optional[str] = None
    reconnect_token_hash: Optional[str] = None
    last_heartbeat_at: datetime = Field(default_factory=datetime.now)


class AgentExecutionState(BaseModel):
    """State during single agent execution."""

    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str

    original_query: str
    rewritten_query: Optional[str] = None

    selected_agent: str = ""
    routing_confidence: float = 0.0
    routing_reasoning: str = ""

    documents: list[dict] = Field(default_factory=list)
    tool_results: list[dict] = Field(default_factory=list)

    intermediate_thoughts: list[str] = Field(default_factory=list)
    final_response: str = ""

    grounding_score: float = 0.0
    relevance_score: float = 0.0

    steps_executed: list[str] = Field(default_factory=list)
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    errors: list[str] = Field(default_factory=list)
    retries: int = 0
