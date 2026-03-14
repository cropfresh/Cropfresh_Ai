"""
State Manager Base Class
========================
Provides shared state (memory dicts, Redis connection) for the mixins.
"""

from datetime import timedelta
from typing import Optional
from loguru import logger

from .models import ConversationContext, AgentExecutionState


class BaseStateManager:
    """
    Base class holding common configuration and shared state.
    """
    # Maximum messages to keep in context window
    MAX_MESSAGES = 50

    # Session TTL (24 hours)
    SESSION_TTL = timedelta(hours=24)

    # Voice session Redis TTL (separate from conversation session)
    VOICE_SESSION_TTL_SECONDS: int = 300  # 5 minutes

    # Maximum age of a voice session that can still be rehydrated
    VOICE_SESSION_MAX_STALE_SECONDS: float = 300.0  # 5 minutes

    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize the base state manager.
        """
        self.redis_url = redis_url
        self._redis_client = None

        # In-memory fallback structures
        self._sessions: dict[str, ConversationContext] = {}
        self._executions: dict[str, AgentExecutionState] = {}
        
        # in-memory voice_session_id → session_id mapping
        self._voice_sessions: dict[str, str] = {}

    async def _get_redis(self):
        """Lazy Redis connection."""
        if self._redis_client is None and self.redis_url:
            try:
                # We do this lazily to avoid forcing redis as a rigid requirement
                import redis.asyncio as redis
                self._redis_client = redis.from_url(self.redis_url)
                await self._redis_client.ping()
                logger.info("Connected to Redis for session storage")
            except Exception as e:
                logger.warning(f"Redis unavailable, using in-memory: {e}")
                self._redis_client = None
        return self._redis_client
