"""Pydantic models for the shared chat API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Single chat message."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Chat request with optional session and context."""

    message: str
    session_id: Optional[str] = None
    context: Optional[dict] = None


class ChatResponse(BaseModel):
    """Chat response with metadata."""

    message: str
    session_id: str
    agent_used: str
    confidence: float
    sources: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    suggested_actions: list[str] = Field(default_factory=list)


class SessionInfo(BaseModel):
    """Session information."""

    session_id: str
    created_at: str
    message_count: int
    entities: dict = Field(default_factory=dict)


class StreamChunk(BaseModel):
    """Single chunk in a streaming chat response."""

    type: str
    content: str
    metadata: dict = Field(default_factory=dict)
