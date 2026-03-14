"""
State Manager Models
====================
Pydantic models and exceptions for conversation state.
"""

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class SessionExpiredError(Exception):
    """
    Raised by rehydrate_voice_session() when the voice session has been
    stale longer than the reconnection tolerance window.
    """
    def __init__(self, voice_session_id: str, stale_seconds: float) -> None:
        self.voice_session_id = voice_session_id
        self.stale_seconds = stale_seconds
        super().__init__(
            f"Voice session '{voice_session_id}' expired after {stale_seconds:.1f}s of inactivity"
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


class ConversationContext(BaseModel):
    """Full conversation context for an agent session."""

    session_id: str
    user_id: Optional[str] = None

    # Conversation history
    messages: list[Message] = Field(default_factory=list)

    # Extracted entities from conversation
    entities: dict[str, Any] = Field(default_factory=dict)

    # User profile (farmer/buyer, location, preferences)
    user_profile: dict[str, Any] = Field(default_factory=dict)

    # Current agent state
    current_agent: Optional[str] = None
    agent_stack: list[str] = Field(default_factory=list)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Token tracking for cost management
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # NFR6: WebRTC voice session linkage
    voice_session_id: Optional[str] = None
    last_active_at: datetime = Field(default_factory=datetime.now)


class AgentExecutionState(BaseModel):
    """State during single agent execution."""

    # Identifiers
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str

    # Current query
    original_query: str
    rewritten_query: Optional[str] = None

    # Routing decision
    selected_agent: str = ""
    routing_confidence: float = 0.0
    routing_reasoning: str = ""

    # Retrieved context
    documents: list[dict] = Field(default_factory=list)
    tool_results: list[dict] = Field(default_factory=list)

    # Generation
    intermediate_thoughts: list[str] = Field(default_factory=list)
    final_response: str = ""

    # Quality metrics
    grounding_score: float = 0.0
    relevance_score: float = 0.0

    # Execution tracking
    steps_executed: list[str] = Field(default_factory=list)
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    # Error handling
    errors: list[str] = Field(default_factory=list)
    retries: int = 0
