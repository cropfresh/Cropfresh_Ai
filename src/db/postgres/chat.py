"""
Chat history operations mixin for Aurora PostgreSQL.

Replaces SupabaseClient.save_chat_message / get_chat_history.
"""

from typing import Any

from loguru import logger

from src.db.postgres.models import ChatMessage


class ChatOperationsMixin:
    """
    Mixin providing chat history persistence.

    Requires the parent class to expose a `pool` property
    returning an asyncpg connection pool.
    """

    async def save_chat_message(
        self,
        user_id: str,
        session_id: str,
        message: ChatMessage,
    ) -> dict[str, Any]:
        """Save a chat message to history."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO chat_history (user_id, session_id, role, content, agent_name)
                VALUES ($1::uuid, $2, $3, $4, $5)
                RETURNING id, user_id, session_id, role, content, agent_name, created_at
                """,
                user_id, session_id, message.role, message.content, message.agent_name,
            )

        logger.debug(f"Saved chat message for session {session_id}")
        return dict(row) if row else {}

    async def get_chat_history(
        self,
        session_id: str,
        limit: int = 20,
    ) -> list[ChatMessage]:
        """Get chat history for a session."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT role, content, agent_name
                FROM chat_history
                WHERE session_id = $1
                ORDER BY created_at ASC
                LIMIT $2
                """,
                session_id, limit,
            )

        return [
            ChatMessage(
                role=row["role"],
                content=row["content"],
                agent_name=row.get("agent_name"),
            )
            for row in rows
        ]
