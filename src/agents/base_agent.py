"""
Base Agent (Proxy)
==================
This file exists for backward compatibility and as a protected file interface.
The actual implementation has been modularized into `src.agents.base.*`.
"""

from src.agents.base import AgentConfig, AgentResponse, BaseAgent

__all__ = [
    "AgentConfig",
    "AgentResponse",
    "BaseAgent",
]
