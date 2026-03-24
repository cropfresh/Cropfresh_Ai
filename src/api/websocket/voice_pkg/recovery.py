"""Helpers for reconnect-aware duplex websocket recovery."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from loguru import logger

from src.memory.state_manager import AgentStateManager, ConversationContext, SessionExpiredError

if TYPE_CHECKING:
    from src.voice.duplex_pipeline import DuplexPipeline


async def resolve_duplex_session(
    *,
    state_manager: AgentStateManager | None,
    requested_session_id: str | None,
    user_id: str,
    reconnect_token: str | None,
    language: str,
    transport_mode: str,
) -> tuple[str, ConversationContext | None, bool, str]:
    """Resolve whether a duplex websocket should recover an existing session or start fresh."""
    session_id = requested_session_id or str(uuid.uuid4())
    if state_manager is None:
        return session_id, None, False, "no_state_manager"

    context = await state_manager.get_context(session_id)
    if context is None:
        created = await state_manager.ensure_session(session_id, user_id=user_id)
        await state_manager.register_voice_session(
            session_id=created.session_id,
            voice_session_id=created.session_id,
            reconnect_token=reconnect_token,
            transport_mode=transport_mode,
            language=language,
        )
        return created.session_id, created, False, "fresh"

    if context.reconnect_token_hash:
        if not reconnect_token:
            fresh = await state_manager.create_session(user_id=user_id)
            await state_manager.register_voice_session(
                session_id=fresh.session_id,
                voice_session_id=fresh.session_id,
                reconnect_token=reconnect_token,
                transport_mode=transport_mode,
                language=language,
            )
            return fresh.session_id, fresh, False, "missing_reconnect_token"

        if not await state_manager.validate_reconnect_token(session_id, reconnect_token):
            fresh = await state_manager.create_session(user_id=user_id)
            await state_manager.register_voice_session(
                session_id=fresh.session_id,
                voice_session_id=fresh.session_id,
                reconnect_token=reconnect_token,
                transport_mode=transport_mode,
                language=language,
            )
            return fresh.session_id, fresh, False, "invalid_reconnect_token"

    try:
        recovered = await state_manager.rehydrate_voice_session(session_id)
    except SessionExpiredError as exc:
        logger.info("Voice reconnect expired for {} after {:.1f}s", exc.voice_session_id, exc.stale_seconds)
        fresh = await state_manager.create_session(user_id=user_id)
        await state_manager.register_voice_session(
            session_id=fresh.session_id,
            voice_session_id=fresh.session_id,
            reconnect_token=reconnect_token,
            transport_mode=transport_mode,
            language=language,
        )
        return fresh.session_id, fresh, False, "expired"

    if recovered is None:
        fresh = await state_manager.create_session(user_id=user_id)
        await state_manager.register_voice_session(
            session_id=fresh.session_id,
            voice_session_id=fresh.session_id,
            reconnect_token=reconnect_token,
            transport_mode=transport_mode,
            language=language,
        )
        return fresh.session_id, fresh, False, "missing_mapping"

    await state_manager.register_voice_session(
        session_id=recovered.session_id,
        voice_session_id=recovered.session_id,
        reconnect_token=reconnect_token,
        transport_mode=transport_mode,
        language=language,
    )
    return recovered.session_id, recovered, True, "recovered"


def apply_recovery_context(
    pipeline: "DuplexPipeline",
    context: ConversationContext | None,
) -> int:
    """Hydrate duplex conversation history from reconnect-safe voice turns."""
    if context is None:
        return 0

    history: list[dict[str, str]] = []
    for turn in context.recent_turns:
        history.append({"role": "user", "content": turn.user_text})
        history.append({"role": "assistant", "content": turn.assistant_text})

    pipeline.load_conversation_history(history)
    return len(context.recent_turns)
