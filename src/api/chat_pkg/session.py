"""Chat session helpers shared by sync and streaming routes."""

from __future__ import annotations

import uuid
from typing import Any

from src.shared.language import ensure_language_context, split_session_context


async def prepare_chat_execution(
    supervisor: Any,
    message: str,
    session_id: str | None = None,
    context: dict | None = None,
) -> tuple[str, dict | None]:
    """Prepare session state and return normalized context for stateless mode."""
    state_manager = supervisor.state_manager
    if not state_manager:
        resolved_id = session_id or str(uuid.uuid4())
        user_profile, entities = split_session_context(context, query=message)
        return resolved_id, ensure_language_context(
            {"user_profile": user_profile, "entities": entities},
            query=message,
        )

    if session_id:
        session = await state_manager.get_context(session_id)
        if not session:
            session = await state_manager.create_session()
    else:
        session = await state_manager.create_session()

    resolved_id = session.session_id
    await _persist_request_context(state_manager, resolved_id, message, context)
    return resolved_id, None


async def execute_chat_request(
    supervisor: Any,
    message: str,
    session_id: str | None = None,
    context: dict | None = None,
):
    """Run a chat request after preparing the session context."""
    resolved_id, normalized_context = await prepare_chat_execution(
        supervisor,
        message,
        session_id=session_id,
        context=context,
    )

    if supervisor.state_manager:
        response = await supervisor.process_with_session(
            query=message,
            session_id=resolved_id,
        )
    else:
        response = await supervisor.process(
            query=message,
            context=normalized_context,
        )

    return resolved_id, response


async def _persist_request_context(
    state_manager: Any,
    session_id: str,
    message: str,
    context: dict | None,
) -> None:
    """Persist request-scoped user profile and entity data into the session."""
    if not context:
        return

    user_profile, entities = split_session_context(context, query=message)
    if user_profile:
        await state_manager.update_user_profile(session_id, user_profile)
    if entities:
        await state_manager.update_entities(session_id, entities)
