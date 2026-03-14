"""
LLM Provider Models
===================
Standardized models for LLM interactions.
"""

from pydantic import BaseModel


class LLMMessage(BaseModel):
    """Standard message format."""
    role: str  # system, user, assistant
    content: str


class LLMResponse(BaseModel):
    """Standard response format."""
    content: str
    model: str
    usage: dict[str, int]  # prompt_tokens, completion_tokens, total_tokens
    finish_reason: str
