"""
Agent State Manager — session, conversation, and voice session management.
"""

import asyncio
import hashlib
import hmac
import re
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
    VoicePlaybackState,
    VoiceSpeakerProfile,
    VoiceSessionState,
    VoiceStateEvent,
    VoiceTurn,
)


class AgentStateManager:
    """
    Centralized state manager for multi-agent RAG system.

    Handles session creation, conversation history with windowing,
    agent execution tracking, and WebRTC voice session rehydration (NFR6).
    """

    MAX_MESSAGES = 50
    MAX_RECENT_VOICE_TURNS = 10
    MAX_VOICE_STATE_EVENTS = 50
    SESSION_TTL = timedelta(hours=24)
    VOICE_SESSION_TTL_SECONDS: int = 300
    VOICE_SESSION_MAX_STALE_SECONDS: float = 300.0

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url
        self._redis_client = None
        self._sessions: dict[str, ConversationContext] = {}
        self._executions: dict[str, AgentExecutionState] = {}
        self._voice_sessions: dict[str, str] = {}
        self._voice_state_events: dict[str, list[VoiceStateEvent]] = {}
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
        self,
        user_id: Optional[str] = None,
        user_profile: Optional[dict] = None,
        session_id: Optional[str] = None,
    ) -> ConversationContext:
        """Create a new conversation session."""
        session_id = session_id or str(uuid.uuid4())
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

    async def ensure_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        user_profile: Optional[dict] = None,
    ) -> ConversationContext:
        """Return an existing session or create one with the requested id."""
        context = await self.get_context(session_id)
        if context is not None:
            return context
        return await self.create_session(
            user_id=user_id,
            user_profile=user_profile,
            session_id=session_id,
        )

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
        self._trim_messages(context)
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

    async def update_user_profile(self, session_id: str, user_profile: dict[str, Any]) -> bool:
        """Merge user profile fields into the session context."""
        context = await self.get_context(session_id)
        if not context:
            return False
        context.user_profile.update(user_profile)
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

    async def register_voice_session(
        self,
        session_id: str,
        voice_session_id: str,
        reconnect_token: Optional[str] = None,
        transport_mode: Optional[str] = None,
        language: Optional[str] = None,
    ) -> None:
        """Link a voice session id to a conversation session and persist reconnect metadata."""
        context = await self.ensure_session(session_id)
        context.voice_session_id = voice_session_id
        context.last_active_at = datetime.now()
        context.last_heartbeat_at = datetime.now()
        context.voice_state = VoiceSessionState.IDLE
        if transport_mode is not None:
            context.transport_mode = transport_mode
        if language is not None:
            context.language = language
        if reconnect_token:
            context.reconnect_token_hash = self._hash_reconnect_token(reconnect_token)

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
        session_id: Optional[str] = None
        if redis:
            raw = await redis.get(f"voice:{voice_session_id}")
            session_id = raw.decode() if isinstance(raw, bytes) else raw
            await redis.delete(f"voice:{voice_session_id}")
        else:
            session_id = self._voice_sessions.pop(voice_session_id, None)

        if session_id:
            context = await self.get_context(session_id)
            if context:
                context.voice_session_id = None
                context.reconnect_token_hash = None
                context.transport_mode = None
                context.playback_state = VoicePlaybackState.IDLE
                context.voice_state = VoiceSessionState.IDLE
                await self._persist_context(session_id, context)

    async def touch_voice_session(self, session_id: str, *, heartbeat: bool = False) -> None:
        """Refresh last_active_at for a voice session."""
        context = await self.get_context(session_id)
        if not context:
            return
        next_active_at = datetime.now()
        if next_active_at <= context.last_active_at:
            next_active_at = context.last_active_at + timedelta(microseconds=1)

        context.last_active_at = next_active_at
        context.updated_at = next_active_at
        if heartbeat:
            next_heartbeat_at = datetime.now()
            if next_heartbeat_at <= context.last_heartbeat_at:
                next_heartbeat_at = context.last_heartbeat_at + timedelta(microseconds=1)
            if next_heartbeat_at < context.last_active_at:
                next_heartbeat_at = context.last_active_at
            context.last_heartbeat_at = next_heartbeat_at
        await self._persist_context(session_id, context)

    async def update_voice_runtime(self, session_id: str, **updates: Any) -> bool:
        """Patch reconnect-aware voice runtime fields on a session."""
        context = await self.get_context(session_id)
        if not context:
            return False

        for key, value in updates.items():
            if key == "reconnect_token":
                context.reconnect_token_hash = (
                    self._hash_reconnect_token(value) if value else None
                )
                continue
            if key == "voice_state" and value is not None:
                context.voice_state = self._coerce_voice_state(value)
                context.playback_state = self._map_voice_state_to_playback(context.voice_state)
                continue
            if hasattr(context, key):
                setattr(context, key, value)

        context.updated_at = datetime.now()
        context.last_active_at = datetime.now()
        if "last_heartbeat_at" not in updates:
            context.last_heartbeat_at = context.last_active_at
        await self._persist_context(session_id, context)
        return True

    async def update_active_workflow(
        self,
        session_id: str,
        workflow: Optional[dict[str, Any]],
    ) -> bool:
        """Replace the active voice workflow context for a session."""
        context = await self.get_context(session_id)
        if not context:
            return False

        context.active_workflow = dict(workflow or {})
        context.updated_at = datetime.now()
        context.last_active_at = context.updated_at
        await self._persist_context(session_id, context)
        return True

    async def update_active_speaker(
        self,
        session_id: str,
        *,
        speaker_id: str | None = None,
        speaker_label: str | None = None,
        speaker_role: str | None = None,
        speaker_confidence: float | None = None,
        speaker_metadata: Optional[dict[str, Any]] = None,
    ) -> VoiceSpeakerProfile | None:
        """Upsert the active speaker for a grouped voice session."""
        context = await self.get_context(session_id)
        if not context:
            return None

        profile = self._upsert_speaker_profile(
            context,
            speaker_id=speaker_id,
            speaker_label=speaker_label,
            speaker_role=speaker_role,
            speaker_confidence=speaker_confidence,
            speaker_metadata=speaker_metadata,
            increment_turn_count=False,
        )
        if profile is None:
            return None

        context.active_speaker_id = profile.speaker_id
        context.updated_at = datetime.now()
        context.last_active_at = context.updated_at
        await self._persist_context(session_id, context)
        return profile

    async def transition_voice_state(
        self,
        session_id: str,
        state: VoiceSessionState | str,
        *,
        source: str,
        reason: Optional[str] = None,
        actor: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[VoiceStateEvent]:
        """Move a session through the canonical Sprint 10 voice state machine."""
        context = await self.get_context(session_id)
        if context is None:
            return None

        next_state = self._coerce_voice_state(state)
        previous_state = context.voice_state
        if not self._is_valid_voice_state_transition(previous_state, next_state):
            raise ValueError(
                f"Invalid voice state transition for {session_id}: "
                f"{previous_state.value} -> {next_state.value}"
            )

        context.voice_state_sequence += 1
        context.voice_state = next_state
        context.playback_state = self._map_voice_state_to_playback(next_state)
        context.updated_at = datetime.now()
        context.last_active_at = context.updated_at
        await self._persist_context(session_id, context)

        event = VoiceStateEvent(
            session_id=session_id,
            sequence=context.voice_state_sequence,
            state=next_state,
            previous_state=previous_state,
            source=source,
            correlation_id=session_id,
            reason=reason,
            actor=actor,
            metadata=dict(metadata or {}),
        )
        self._record_voice_state_event(event)
        await self._publish_voice_state_event(event)
        return event

    def get_voice_state_events(self, session_id: str) -> list[VoiceStateEvent]:
        """Return the in-memory voice-state event history for a session."""
        return list(self._voice_state_events.get(session_id, ()))

    async def append_recent_voice_turn(self, session_id: str, turn: VoiceTurn) -> bool:
        """Persist a compact reconnect-safe voice turn and mirror it into message history."""
        context = await self.get_context(session_id)
        if not context:
            return False

        profile = self._upsert_speaker_profile(
            context,
            speaker_id=turn.speaker_id,
            speaker_label=turn.speaker_label,
            speaker_role=turn.speaker_role,
            speaker_confidence=turn.speaker_confidence,
            speaker_metadata=turn.speaker_metadata,
            increment_turn_count=bool(
                turn.speaker_id or turn.speaker_label or context.active_speaker_id
            ),
        )
        if profile is not None:
            turn.speaker_id = profile.speaker_id
            turn.speaker_label = profile.label
            turn.speaker_role = profile.role
            turn.speaker_confidence = profile.confidence
            turn.speaker_metadata = dict(profile.metadata)
            context.active_speaker_id = profile.speaker_id

        context.recent_turns.append(turn)
        if len(context.recent_turns) > self.MAX_RECENT_VOICE_TURNS:
            context.recent_turns = context.recent_turns[-self.MAX_RECENT_VOICE_TURNS:]

        context.last_turn_id = turn.turn_id
        context.pending_transcript = None
        context.pending_segment_id = None
        context.language = turn.language or context.language
        context.updated_at = datetime.now()
        context.last_active_at = context.updated_at
        context.last_heartbeat_at = context.updated_at

        context.messages.append(
            Message(
                role="user",
                content=turn.user_text,
                metadata={
                    "transport": "voice",
                    "turn_id": turn.turn_id,
                    "language": turn.language,
                    "speaker_id": turn.speaker_id,
                    "speaker_label": turn.speaker_label,
                    "speaker_role": turn.speaker_role,
                    "speaker_confidence": turn.speaker_confidence,
                    "speaker_metadata": turn.speaker_metadata,
                },
            )
        )
        context.messages.append(
            Message(
                role="assistant",
                content=turn.assistant_text,
                metadata={
                    "transport": "voice",
                    "turn_id": turn.turn_id,
                    "language": turn.language,
                    "interrupted": turn.interrupted,
                    "timing": turn.timing,
                    "speaker_id": turn.speaker_id,
                    "speaker_label": turn.speaker_label,
                    "speaker_role": turn.speaker_role,
                    "speaker_confidence": turn.speaker_confidence,
                    "speaker_metadata": turn.speaker_metadata,
                },
            )
        )
        self._trim_messages(context)
        await self._persist_context(session_id, context)
        return True

    async def validate_reconnect_token(self, session_id: str, reconnect_token: str) -> bool:
        """Return whether the provided reconnect token matches the stored hash."""
        context = await self.get_context(session_id)
        if not context or not context.reconnect_token_hash:
            return False
        provided_hash = self._hash_reconnect_token(reconnect_token)
        return hmac.compare_digest(context.reconnect_token_hash, provided_hash)

    def _upsert_speaker_profile(
        self,
        context: ConversationContext,
        *,
        speaker_id: str | None = None,
        speaker_label: str | None = None,
        speaker_role: str | None = None,
        speaker_confidence: float | None = None,
        speaker_metadata: Optional[dict[str, Any]] = None,
        increment_turn_count: bool = False,
    ) -> VoiceSpeakerProfile | None:
        resolved_id = self._normalize_speaker_id(
            speaker_id=speaker_id,
            speaker_label=speaker_label,
            fallback_speaker_id=context.active_speaker_id,
        )
        if not resolved_id:
            return None

        profile = context.speaker_profiles.get(resolved_id)
        if profile is None:
            profile = VoiceSpeakerProfile(
                speaker_id=resolved_id,
                label=speaker_label,
                role=speaker_role,
                confidence=speaker_confidence,
                metadata=dict(speaker_metadata or {}),
            )
        else:
            if speaker_label:
                profile.label = speaker_label
            if speaker_role:
                profile.role = speaker_role
            if speaker_confidence is not None:
                profile.confidence = speaker_confidence
            if speaker_metadata:
                profile.metadata.update(speaker_metadata)

        if increment_turn_count:
            profile.turn_count += 1
        profile.last_seen_at = datetime.now()
        context.speaker_profiles[resolved_id] = profile
        return profile

    def _normalize_speaker_id(
        self,
        *,
        speaker_id: str | None = None,
        speaker_label: str | None = None,
        fallback_speaker_id: str | None = None,
    ) -> str:
        if speaker_id:
            return speaker_id.strip()
        if speaker_label:
            normalized = re.sub(r"[^a-z0-9]+", "-", speaker_label.lower()).strip("-")
            return f"speaker:{normalized or 'group'}"
        if fallback_speaker_id:
            return fallback_speaker_id
        return "speaker:primary"

    # ── Private helpers ───────────────────────────────────────

    async def _publish_voice_state_event(self, event: VoiceStateEvent) -> None:
        """Broadcast voice-state changes for bridge and UI consumers."""
        redis = await self._get_redis()
        if not redis:
            return

        payload = event.model_dump_json()
        await redis.publish("voice:state", payload)
        await redis.publish(f"voice:state:{event.session_id}", payload)

    def _record_voice_state_event(self, event: VoiceStateEvent) -> None:
        events = self._voice_state_events.setdefault(event.session_id, [])
        events.append(event)
        if len(events) > self.MAX_VOICE_STATE_EVENTS:
            self._voice_state_events[event.session_id] = events[-self.MAX_VOICE_STATE_EVENTS:]

    def _coerce_voice_state(self, value: VoiceSessionState | str) -> VoiceSessionState:
        if isinstance(value, VoiceSessionState):
            return value
        return VoiceSessionState(value)

    def _is_valid_voice_state_transition(
        self,
        previous: VoiceSessionState,
        next_state: VoiceSessionState,
    ) -> bool:
        allowed = {
            VoiceSessionState.IDLE: {
                VoiceSessionState.IDLE,
                VoiceSessionState.LISTENING,
                VoiceSessionState.TRANSCRIBING,
                VoiceSessionState.THINKING,
            },
            VoiceSessionState.LISTENING: {
                VoiceSessionState.LISTENING,
                VoiceSessionState.VAD_TRIGGERED,
                VoiceSessionState.BARGE_IN,
                VoiceSessionState.IDLE,
            },
            VoiceSessionState.VAD_TRIGGERED: {
                VoiceSessionState.VAD_TRIGGERED,
                VoiceSessionState.TRANSCRIBING,
                VoiceSessionState.BARGE_IN,
                VoiceSessionState.IDLE,
            },
            VoiceSessionState.TRANSCRIBING: {
                VoiceSessionState.TRANSCRIBING,
                VoiceSessionState.THINKING,
                VoiceSessionState.BARGE_IN,
                VoiceSessionState.IDLE,
            },
            VoiceSessionState.THINKING: {
                VoiceSessionState.THINKING,
                VoiceSessionState.SPEAKING,
                VoiceSessionState.BARGE_IN,
                VoiceSessionState.IDLE,
            },
            VoiceSessionState.SPEAKING: {
                VoiceSessionState.SPEAKING,
                VoiceSessionState.BARGE_IN,
                VoiceSessionState.LISTENING,
                VoiceSessionState.IDLE,
            },
            VoiceSessionState.BARGE_IN: {
                VoiceSessionState.BARGE_IN,
                VoiceSessionState.LISTENING,
                VoiceSessionState.TRANSCRIBING,
                VoiceSessionState.IDLE,
            },
        }
        return next_state in allowed.get(previous, {next_state})

    def _map_voice_state_to_playback(
        self,
        state: VoiceSessionState,
    ) -> VoicePlaybackState:
        mapping = {
            VoiceSessionState.IDLE: VoicePlaybackState.IDLE,
            VoiceSessionState.LISTENING: VoicePlaybackState.LISTENING,
            VoiceSessionState.VAD_TRIGGERED: VoicePlaybackState.LISTENING,
            VoiceSessionState.TRANSCRIBING: VoicePlaybackState.TRANSCRIBING,
            VoiceSessionState.THINKING: VoicePlaybackState.THINKING,
            VoiceSessionState.SPEAKING: VoicePlaybackState.SPEAKING,
            VoiceSessionState.BARGE_IN: VoicePlaybackState.INTERRUPTED,
        }
        return mapping[state]

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

    def _trim_messages(self, context: ConversationContext) -> None:
        """Keep system messages plus the most recent non-system messages."""
        if len(context.messages) <= self.MAX_MESSAGES:
            return
        system_msgs = [m for m in context.messages if m.role == "system"]
        other_msgs = [m for m in context.messages if m.role != "system"]
        context.messages = system_msgs + other_msgs[-(self.MAX_MESSAGES - len(system_msgs)):]

    def _hash_reconnect_token(self, reconnect_token: str) -> str:
        """Hash reconnect tokens before persisting them to session state."""
        return hashlib.sha256(reconnect_token.encode("utf-8")).hexdigest()

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
