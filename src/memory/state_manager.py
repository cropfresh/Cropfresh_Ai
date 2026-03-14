"""
Agent State Manager — Backward compatibility redirect.

! This file is kept for backward compatibility. The actual implementation
! has been split into the `src.memory.state_pkg` package.
! Import from `src.memory.state_pkg` directly in new code.
"""

from src.memory.state_pkg.entities import extract_entities
from src.memory.state_pkg.manager import AgentStateManager
from src.memory.state_pkg.models import (
    AgentExecutionState,
    ConversationContext,
    Message,
    SessionExpiredError,
)

__all__ = [
    "AgentExecutionState",
    "AgentStateManager",
    "ConversationContext",
    "Message",
    "SessionExpiredError",
    "extract_entities",
]
