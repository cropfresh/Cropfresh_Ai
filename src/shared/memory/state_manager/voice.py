"""
Voice Session Mixin
===================
Handles NFR6 WebRTC voice session rehydration and linkage.
"""

import asyncio
from datetime import datetime
from typing import Optional

from loguru import logger

from .models import ConversationContext, SessionExpiredError
from .session import SessionManagerMixin


class VoiceSessionMixin(SessionManagerMixin):
    """
    Mixin for WebRTC Voice Session reconnect SLA (< 1.0s).
    Inherits from SessionManagerMixin since we need access to `get_context`.
    """

    async def register_voice_session(
        self,
        session_id: str,
        voice_session_id: str,
    ) -> None:
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
                    self.VOICE_SESSION_TTL_SECONDS,
                    session_id,
                )
            else:
                self._sessions[session_id] = context
                self._voice_sessions[voice_session_id] = session_id

        logger.debug(
            "Voice session registered: voice_id={} session_id={}",
            voice_session_id, session_id,
        )

    async def rehydrate_voice_session(
        self,
        voice_session_id: str,
        timeout_sec: float = 1.0,
    ) -> Optional[ConversationContext]:
        """
        Rehydrate a dropped WebRTC session upon client reconnection (NFR6).
        Returns within timeout_sec (default 1.0s).
        """
        try:
            context = await asyncio.wait_for(
                self._lookup_voice_session(voice_session_id),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "NFR6 SLA breach: voice session rehydration exceeded {:.1f}s for {}",
                timeout_sec, voice_session_id,
            )
            return None

        if context is None:
            return None

        stale_sec = (datetime.now() - context.last_active_at).total_seconds()
        if stale_sec > self.VOICE_SESSION_MAX_STALE_SECONDS:
            raise SessionExpiredError(voice_session_id, stale_sec)

        logger.info(
            "NFR6: voice session {} rehydrated — session={} stale={:.1f}s",
            voice_session_id, context.session_id, stale_sec,
        )
        return context

    async def deregister_voice_session(self, voice_session_id: str) -> None:
        """Remove the voice-session mapping on clean disconnect."""
        redis = await self._get_redis()
        if redis:
            await redis.delete(f"voice:{voice_session_id}")
        else:
            self._voice_sessions.pop(voice_session_id, None)

        logger.debug("Voice session deregistered: {}", voice_session_id)

    async def touch_voice_session(self, session_id: str) -> None:
        """Refresh last_active_at for a voice session frame."""
        context = await self.get_context(session_id)
        if not context:
            return

        context.last_active_at = datetime.now()

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
        self, voice_session_id: str
    ) -> Optional[ConversationContext]:
        """Internal lookup for voice_session_id → session_id → Context."""
        redis = await self._get_redis()
        if redis:
            raw_session_id = await redis.get(f"voice:{voice_session_id}")
            if not raw_session_id:
                return None
            session_id = (
                raw_session_id.decode()
                if isinstance(raw_session_id, bytes)
                else raw_session_id
            )
        else:
            session_id = self._voice_sessions.get(voice_session_id)
            if not session_id:
                return None

        return await self.get_context(session_id)
