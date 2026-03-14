"""
Agent State Manager
===================
Combines session, execution, and voice mixins into a single API structure.
"""

from .base import BaseStateManager
from .session import SessionManagerMixin
from .execution import ExecutionTrackerMixin
from .voice import VoiceSessionMixin
from loguru import logger


class AgentStateManager(
    VoiceSessionMixin,
    ExecutionTrackerMixin,
):
    """
    Centralized state manager for multi-agent RAG system.
    
    Handles:
    - Session creation and management
    - Conversation history with windowing
    - User context persistence
    - Agent execution tracking
    - WebRTC voice session rehydration (NFR6: <1.0s SLA)
    """
    
    def __init__(self, redis_url: str | None = None):
        """
        Initialize state manager, combining all mixins.
        """
        super().__init__(redis_url=redis_url)
        logger.info("AgentStateManager initialized")
