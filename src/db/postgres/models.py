"""
Data models for the Aurora PostgreSQL client.
"""

from typing import Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str  # 'user' or 'assistant'
    content: str
    agent_name: Optional[str] = None
