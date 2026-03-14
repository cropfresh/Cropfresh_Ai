"""
Session Manager Mixin
=====================
Handles conversation context, message history, and entity updates.
"""

import uuid
from datetime import datetime
from typing import Any, Optional
from loguru import logger

from .models import ConversationContext, Message
from .base import BaseStateManager


class SessionManagerMixin(BaseStateManager):
    """Mixin for core session logic."""

    async def create_session(
        self,
        user_id: Optional[str] = None,
        user_profile: Optional[dict] = None,
    ) -> ConversationContext:
        """Create a new conversation session."""
        session_id = str(uuid.uuid4())

        context = ConversationContext(
            session_id=session_id,
            user_id=user_id,
            user_profile=user_profile or {},
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
        """Add a message to conversation history with sliding window."""
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

        redis = await self._get_redis()
        if redis:
            await redis.setex(
                f"session:{session_id}",
                int(self.SESSION_TTL.total_seconds()),
                context.model_dump_json(),
            )
        else:
            self._sessions[session_id] = context

        return True

    async def update_entities(self, session_id: str, entities: dict[str, Any]) -> bool:
        """Update extracted entities for a session."""
        context = await self.get_context(session_id)
        if not context:
            return False

        context.entities.update(entities)
        context.updated_at = datetime.now()

        redis = await self._get_redis()
        if redis:
            await redis.setex(
                f"session:{session_id}",
                int(self.SESSION_TTL.total_seconds()),
                context.model_dump_json(),
            )
        else:
            self._sessions[session_id] = context

        return True

    def get_conversation_summary(
        self,
        context: ConversationContext,
        max_messages: int = 10,
    ) -> str:
        """Generate plain text conversation summary for prompt injection."""
        if not context.messages:
            return ""

        recent = context.messages[-max_messages:]
        lines = ["Previous conversation:"]
        
        for msg in recent:
            role = msg.role.upper()
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            lines.append(f"{role}: {content}")

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
