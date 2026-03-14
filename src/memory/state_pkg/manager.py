"""
Agent State Manager — session, conversation, and voice session management.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from loguru import logger

from src.memory.state_pkg.entities import extract_entities
from src.memory.state_pkg.models import (
    AgentExecutionState,
    ConversationContext,
    Message,
    SessionExpiredError,
)


class AgentStateManager:
    """
    Centralized state manager for multi-agent RAG system.

    Handles session creation, conversation history with windowing,
    agent execution tracking, and WebRTC voice session rehydration (NFR6).
    """

    MAX_MESSAGES = 50
    SESSION_TTL = timedelta(hours=24)
    VOICE_SESSION_TTL_SECONDS: int = 300
    VOICE_SESSION_MAX_STALE_SECONDS: float = 300.0

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url
        self._redis_client = None
        self._sessions: dict[str, ConversationContext] = {}
        self._executions: dict[str, AgentExecutionState] = {}
        self._voice_sessions: dict[str, str] = {}
        logger.info("AgentStateManager initialized")

    async def _get_redis(self):
        """Lazy Redis connection."""
        if self._redis_client is None and self.redis_url:
            try:
                import redis.asyncio as redis
                self._redis_client = redis.from_url(self.redis_url)
                await self._redis_client.ping()
                logger.info("Connected to Redis for session storage")
            except Exception as e:
                logger.warning(f"Redis unavailable, using in-memory: {e}")
                self._redis_client = None
        return self._redis_client

    # ── Session Management ────────────────────────────────────

    async def create_session(
        self, user_id: Optional[str] = None, user_profile: Optional[dict] = None,
    ) -> ConversationContext:
        """Create a new conversation session."""
        session_id = str(uuid.uuid4())
        context = ConversationContext(
            session_id=session_id, user_id=user_id, user_profile=user_profile or {},
        )
        redis = await self._get_redis()
        if redis:
            await redis.setex(
                f"session:{session_id}",
                int(self.SESSION_TTL.total_seconds()),
                context.model_dump_json(),
            )
        else:
            self._sessions[session_id] = context
        logger.debug(f"Created session: {session_id}")
        return context

    async def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get conversation context for a session."""
        redis = await self._get_redis()
        if redis:
            data = await redis.get(f"session:{session_id}")
            if data:
                return ConversationContext.model_validate_json(data)
        else:
            return self._sessions.get(session_id)
        return None

    async def add_message(self, session_id: str, message: Message) -> bool:
        """Add a message to conversation history."""
        context = await self.get_context(session_id)
        if not context:
            logger.warning(f"Session not found: {session_id}")
            return False

        context.messages.append(message)

        if len(context.messages) > self.MAX_MESSAGES:
            system_msgs = [m for m in context.messages if m.role == "system"]
            other_msgs = [m for m in context.messages if m.role != "system"]
            context.messages = system_msgs + other_msgs[-(self.MAX_MESSAGES - len(system_msgs)):]

        context.updated_at = datetime.now()
        await self._persist_context(session_id, context)
        return True

    async def update_entities(self, session_id: str, entities: dict[str, Any]) -> bool:
        """Update extracted entities for a session."""
        context = await self.get_context(session_id)
        if not context:
            return False
        context.entities.update(entities)
        context.updated_at = datetime.now()
        await self._persist_context(session_id, context)
        return True

    async def extract_and_merge_entities(self, session_id: str, text: str) -> dict[str, Any]:
        """Extract agricultural entities from text and merge into session."""
        found = extract_entities(text)
        if found:
            await self.update_entities(session_id, found)
            logger.debug("Extracted entities for {}: {}", session_id, list(found.keys()))
        return found

    # ── Execution Tracking ────────────────────────────────────

    def create_execution(self, session_id: str, query: str) -> AgentExecutionState:
        """Create execution state for a single query."""
        execution = AgentExecutionState(session_id=session_id, original_query=query)
        self._executions[execution.execution_id] = execution
        logger.debug(f"Created execution: {execution.execution_id}")
        return execution

    def update_execution(self, execution_id: str, **updates) -> Optional[AgentExecutionState]:
        """Update execution state."""
        execution = self._executions.get(execution_id)
        if not execution:
            return None
        for key, value in updates.items():
            if hasattr(execution, key):
                setattr(execution, key, value)
        return execution

    def add_step(self, execution_id: str, step: str) -> None:
        """Add step to execution trace."""
        execution = self._executions.get(execution_id)
        if execution:
            execution.steps_executed.append(step)

    def complete_execution(self, execution_id: str, response: str) -> Optional[AgentExecutionState]:
        """Mark execution as complete."""
        execution = self._executions.get(execution_id)
        if execution:
            execution.final_response = response
            execution.end_time = datetime.now()
        return execution

    # ── Conversation Summary ──────────────────────────────────

    def get_conversation_summary(
        self, context: ConversationContext, max_messages: int = 10,
    ) -> str:
        """Generate conversation summary for context injection."""
        if not context.messages:
            return ""
        recent = context.messages[-max_messages:]
        lines = ["Previous conversation:"]
        for msg in recent:
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            lines.append(f"{msg.role.upper()}: {content}")
        return "\n".join(lines)

    def cleanup_old_sessions(self) -> int:
        """Remove expired in-memory sessions."""
        now = datetime.now()
        expired = [
            sid for sid, ctx in self._sessions.items()
            if now - ctx.updated_at > self.SESSION_TTL
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
        return len(expired)

    # ── NFR6: Voice Session Rehydration ───────────────────────

    async def register_voice_session(self, session_id: str, voice_session_id: str) -> None:
        """Link a WebRTC voice_session_id to a conversation session_id."""
        context = await self.get_context(session_id)
        if context:
            context.voice_session_id = voice_session_id
            context.last_active_at = datetime.now()
            redis = await self._get_redis()
            if redis:
                await redis.setex(
                    f"session:{session_id}",
                    int(self.SESSION_TTL.total_seconds()),
                    context.model_dump_json(),
                )
                await redis.setex(
                    f"voice:{voice_session_id}",
                    self.VOICE_SESSION_TTL_SECONDS, session_id,
                )
            else:
                self._sessions[session_id] = context
                self._voice_sessions[voice_session_id] = session_id
        logger.debug("Voice session registered: voice_id={} session_id={}",
                      voice_session_id, session_id)

    async def rehydrate_voice_session(
        self, voice_session_id: str, timeout_sec: float = 1.0,
    ) -> Optional[ConversationContext]:
        """Rehydrate a dropped WebRTC session (NFR6: <1.0s SLA)."""
        try:
            context = await asyncio.wait_for(
                self._lookup_voice_session(voice_session_id), timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            logger.warning("NFR6 SLA breach: rehydration exceeded {:.1f}s for {}",
                           timeout_sec, voice_session_id)
            return None

        if context is None:
            return None

        stale_sec = (datetime.now() - context.last_active_at).total_seconds()
        if stale_sec > self.VOICE_SESSION_MAX_STALE_SECONDS:
            raise SessionExpiredError(voice_session_id, stale_sec)

        logger.info("NFR6: voice session {} rehydrated — session={} stale={:.1f}s",
                     voice_session_id, context.session_id, stale_sec)
        return context

    async def deregister_voice_session(self, voice_session_id: str) -> None:
        """Remove voice-session mapping on clean disconnect."""
        redis = await self._get_redis()
        if redis:
            await redis.delete(f"voice:{voice_session_id}")
        else:
            self._voice_sessions.pop(voice_session_id, None)

    async def touch_voice_session(self, session_id: str) -> None:
        """Refresh last_active_at for a voice session."""
        context = await self.get_context(session_id)
        if not context:
            return
        context.last_active_at = datetime.now()
        await self._persist_context(session_id, context)

    # ── Private helpers ───────────────────────────────────────

    async def _persist_context(self, session_id: str, context: ConversationContext) -> None:
        """Write context to Redis or in-memory store."""
        redis = await self._get_redis()
        if redis:
            await redis.setex(
                f"session:{session_id}",
                int(self.SESSION_TTL.total_seconds()),
                context.model_dump_json(),
            )
        else:
            self._sessions[session_id] = context

    async def _lookup_voice_session(
        self, voice_session_id: str,
    ) -> Optional[ConversationContext]:
        """Resolve voice_session_id → session_id → ConversationContext."""
        redis = await self._get_redis()
        if redis:
            raw = await redis.get(f"voice:{voice_session_id}")
            if not raw:
                return None
            session_id = raw.decode() if isinstance(raw, bytes) else raw
        else:
            session_id = self._voice_sessions.get(voice_session_id)
            if not session_id:
                return None
        return await self.get_context(session_id)
