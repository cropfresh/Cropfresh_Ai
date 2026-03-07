"""
Agent State Manager
===================
Production-grade state management for multi-agent RAG system.

Provides:
- Conversation memory with Redis/in-memory fallback
- Agent execution context tracking
- User session management
- Tool execution history
- WebRTC voice session rehydration (NFR6: <1.0s SLA)

Author: CropFresh AI Team
Version: 2.1.0
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field


# * ─── Custom Exceptions ──────────────────────────────────────────────────────


class SessionExpiredError(Exception):
    """
    Raised by rehydrate_voice_session() when the voice session has been
    stale longer than the reconnection tolerance window (default: 5 minutes).

    Client should show a 'session expired — please start a new call' message.
    """
    def __init__(self, voice_session_id: str, stale_seconds: float) -> None:
        self.voice_session_id = voice_session_id
        self.stale_seconds = stale_seconds
        super().__init__(
            f"Voice session '{voice_session_id}' expired after {stale_seconds:.1f}s of inactivity"
        )


class Message(BaseModel):
    """Single message in conversation."""

    role: str  # user, assistant, system, tool
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # For tool messages
    tool_name: Optional[str] = None
    tool_result: Optional[dict] = None


class ConversationContext(BaseModel):
    """Full conversation context for an agent session."""

    session_id: str
    user_id: Optional[str] = None

    # Conversation history
    messages: list[Message] = Field(default_factory=list)

    # Extracted entities from conversation
    entities: dict[str, Any] = Field(default_factory=dict)

    # User profile (farmer/buyer, location, preferences)
    user_profile: dict[str, Any] = Field(default_factory=dict)

    # Current agent state
    current_agent: Optional[str] = None
    agent_stack: list[str] = Field(default_factory=list)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Token tracking for cost management
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # * NFR6: WebRTC voice session linkage
    # voice_session_id is the WebRTC peer connection / client session ID
    # supplied by the client on connect. Used for reconnection lookup.
    voice_session_id: Optional[str] = None
    # last_active_at: updated on every incoming voice frame.
    # Used to detect stale sessions during reconnection attempts.
    last_active_at: datetime = Field(default_factory=datetime.now)


class AgentExecutionState(BaseModel):
    """State during single agent execution."""

    # Identifiers
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str

    # Current query
    original_query: str
    rewritten_query: Optional[str] = None

    # Routing decision
    selected_agent: str = ""
    routing_confidence: float = 0.0
    routing_reasoning: str = ""

    # Retrieved context
    documents: list[dict] = Field(default_factory=list)
    tool_results: list[dict] = Field(default_factory=list)

    # Generation
    intermediate_thoughts: list[str] = Field(default_factory=list)
    final_response: str = ""

    # Quality metrics
    grounding_score: float = 0.0
    relevance_score: float = 0.0

    # Execution tracking
    steps_executed: list[str] = Field(default_factory=list)
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    # Error handling
    errors: list[str] = Field(default_factory=list)
    retries: int = 0


class AgentStateManager:
    """
    Centralized state manager for multi-agent RAG system.

    Handles:
    - Session creation and management
    - Conversation history with windowing
    - User context persistence
    - Agent execution tracking
    - WebRTC voice session rehydration (NFR6: <1.0s SLA)

    Usage:
        manager = AgentStateManager()
        session = await manager.create_session(user_id="farmer_123")
        context = await manager.get_context(session.session_id)
        await manager.add_message(session.session_id, Message(...))

        # NFR6: Voice session reconnection
        await manager.register_voice_session(session.session_id, voice_session_id)
        ctx = await manager.rehydrate_voice_session(voice_session_id)
    """

    # Maximum messages to keep in context window
    MAX_MESSAGES = 50

    # Session TTL (24 hours)
    SESSION_TTL = timedelta(hours=24)

    # * NFR6: Voice session Redis TTL (separate from conversation session)
    # WebRTC connections can be recovered up to 5 minutes after drop (client UX budget)
    VOICE_SESSION_TTL_SECONDS: int = 300  # 5 minutes

    # * NFR6: Maximum age of a voice session that can still be rehydrated
    VOICE_SESSION_MAX_STALE_SECONDS: float = 300.0  # 5 minutes

    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize state manager.

        Args:
            redis_url: Optional Redis URL for distributed sessions
        """
        self.redis_url = redis_url
        self._redis_client = None

        # In-memory fallback
        self._sessions: dict[str, ConversationContext] = {}
        self._executions: dict[str, AgentExecutionState] = {}
        # * NFR6: in-memory voice_session_id → session_id mapping (Redis when available)
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

    async def create_session(
        self,
        user_id: Optional[str] = None,
        user_profile: Optional[dict] = None,
    ) -> ConversationContext:
        """
        Create a new conversation session.

        Args:
            user_id: Optional user identifier
            user_profile: Optional user profile data

        Returns:
            New ConversationContext
        """
        session_id = str(uuid.uuid4())

        context = ConversationContext(
            session_id=session_id,
            user_id=user_id,
            user_profile=user_profile or {},
        )

        # Store session
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
        """
        Get conversation context for a session.

        Args:
            session_id: Session identifier

        Returns:
            ConversationContext or None if not found
        """
        redis = await self._get_redis()

        if redis:
            data = await redis.get(f"session:{session_id}")
            if data:
                return ConversationContext.model_validate_json(data)
        else:
            return self._sessions.get(session_id)

        return None

    async def add_message(
        self,
        session_id: str,
        message: Message,
    ) -> bool:
        """
        Add a message to conversation history.

        Args:
            session_id: Session identifier
            message: Message to add

        Returns:
            True if successful
        """
        context = await self.get_context(session_id)
        if not context:
            logger.warning(f"Session not found: {session_id}")
            return False

        # Add message
        context.messages.append(message)

        # Apply windowing if needed
        if len(context.messages) > self.MAX_MESSAGES:
            # Keep system messages and last N messages
            system_msgs = [m for m in context.messages if m.role == "system"]
            other_msgs = [m for m in context.messages if m.role != "system"]
            context.messages = system_msgs + other_msgs[-(self.MAX_MESSAGES - len(system_msgs)):]

        # Update timestamp
        context.updated_at = datetime.now()

        # Persist
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

    async def update_entities(
        self,
        session_id: str,
        entities: dict[str, Any],
    ) -> bool:
        """
        Update extracted entities for a session.

        Args:
            session_id: Session identifier
            entities: Entities to merge

        Returns:
            True if successful
        """
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

    # * ─── Entity Extraction (Phase 3 — G3) ─────────────────────────────────

    # Compiled once at class level for efficiency
    _ENTITY_PATTERNS: dict[str, Any] = {}

    @classmethod
    def _get_entity_patterns(cls) -> dict:
        """Lazy-compile entity regex patterns (threadsafe class-level cache)."""
        import re
        if not cls._ENTITY_PATTERNS:
            cls._ENTITY_PATTERNS = {
                "commodity": re.compile(
                    r"\b(tomato|tamatar|potato|aloo|alugedde|onion|pyaaz|eerulli|"
                    r"carrot|gajjari|okra|bhindi|bendekai|cauliflower|gobhi|"
                    r"beans|hurali|brinjal|cucumber|chilli|mirchi|capsicum|cabbage)\b",
                    re.IGNORECASE,
                ),
                "quantity_kg": re.compile(
                    r"(\d+(?:\.\d+)?)\s*(?:kg|kgs|kilo|kilogram)", re.IGNORECASE
                ),
                "quantity_quintal": re.compile(
                    r"(\d+(?:\.\d+)?)\s*(?:quintal|quintals|q\b)", re.IGNORECASE
                ),
                "district": re.compile(
                    r"\b(kolar|tumkur|tumakuru|hassan|mysuru|mysore|belagavi|belgaum|"
                    r"hubli|dharwad|gadag|bidar|raichur|bagalkot|mandya|shimoga|"
                    r"shivamogga|davangere|chitradurga|chikkaballapur|bangalore|bengaluru|"
                    r"udupi|mangalore|ballari|bellary)\b",
                    re.IGNORECASE,
                ),
                "price_per_kg": re.compile(
                    r"₹\s*(\d+(?:\.\d+)?)\s*/?\s*kg", re.IGNORECASE
                ),
            }
        return cls._ENTITY_PATTERNS

    async def extract_and_merge_entities(
        self,
        session_id: str,
        text: str,
    ) -> dict[str, Any]:
        """
        Extract agricultural entities from text and merge into session.

        Runs on every user query and agent response to keep entities
        fresh without any LLM call. Returns the dict of newly found entities.

        Args:
            session_id: Session to merge into
            text:       User query or agent response text

        Returns:
            Dict of newly extracted entity key/value pairs
        """
        patterns = self._get_entity_patterns()
        found: dict[str, Any] = {}

        commodity_match = patterns["commodity"].search(text)
        if commodity_match:
            raw_term = commodity_match.group(0).lower()
            mapping = {
                "tamatar": "tomato",
                "aloo": "potato",
                "alugedde": "potato",
                "pyaaz": "onion",
                "eerulli": "onion",
                "gajjari": "carrot",
                "bhindi": "okra",
                "bendekai": "okra",
                "gobhi": "cauliflower",
                "hurali": "beans",
                "mirchi": "chilli",
            }
            canonical = mapping.get(raw_term, raw_term).capitalize()
            found["commodity"] = canonical

        qty_kg = patterns["quantity_kg"].search(text)
        if qty_kg:
            found["quantity_kg"] = float(qty_kg.group(1))

        qty_q = patterns["quantity_quintal"].search(text)
        if qty_q:
            found["quantity_quintal"] = float(qty_q.group(1))
            found["quantity_kg"] = found["quantity_quintal"] * 100.0

        district_match = patterns["district"].search(text)
        if district_match:
            found["district"] = district_match.group(0).title()

        price_match = patterns["price_per_kg"].search(text)
        if price_match:
            found["price_per_kg"] = float(price_match.group(1))

        if found:
            await self.update_entities(session_id, found)
            logger.debug("Extracted entities for {}: {}", session_id, list(found.keys()))

        return found

    def create_execution(
        self,
        session_id: str,
        query: str,
    ) -> AgentExecutionState:
        """
        Create execution state for a single query.

        Args:
            session_id: Parent session
            query: User query

        Returns:
            New AgentExecutionState
        """
        execution = AgentExecutionState(
            session_id=session_id,
            original_query=query,
        )

        self._executions[execution.execution_id] = execution
        logger.debug(f"Created execution: {execution.execution_id}")

        return execution

    def update_execution(
        self,
        execution_id: str,
        **updates,
    ) -> Optional[AgentExecutionState]:
        """
        Update execution state.

        Args:
            execution_id: Execution identifier
            **updates: Fields to update

        Returns:
            Updated state or None
        """
        execution = self._executions.get(execution_id)
        if not execution:
            return None

        for key, value in updates.items():
            if hasattr(execution, key):
                setattr(execution, key, value)

        return execution

    def add_step(
        self,
        execution_id: str,
        step: str,
    ) -> None:
        """Add step to execution trace."""
        execution = self._executions.get(execution_id)
        if execution:
            execution.steps_executed.append(step)

    def complete_execution(
        self,
        execution_id: str,
        response: str,
    ) -> Optional[AgentExecutionState]:
        """
        Mark execution as complete.

        Args:
            execution_id: Execution identifier
            response: Final response

        Returns:
            Completed execution state
        """
        execution = self._executions.get(execution_id)
        if execution:
            execution.final_response = response
            execution.end_time = datetime.now()
        return execution

    def get_conversation_summary(
        self,
        context: ConversationContext,
        max_messages: int = 10,
    ) -> str:
        """
        Generate conversation summary for context injection.

        Args:
            context: Conversation context
            max_messages: Maximum recent messages to include

        Returns:
            Formatted conversation summary
        """
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
        """
        Remove expired in-memory sessions.

        Returns:
            Number of sessions removed
        """
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

    # * ─── NFR6: WebRTC Voice Session Rehydration ──────────────────────────────

    async def register_voice_session(
        self,
        session_id: str,
        voice_session_id: str,
    ) -> None:
        """
        Link a WebRTC voice_session_id to a conversation session_id.

        Called when a client establishes a new WebRTC connection. Stores
        the mapping so that reconnecting clients can call
        rehydrate_voice_session() to resume their conversation context.

        Args:
            session_id:       Conversation session UUID.
            voice_session_id: WebRTC peer-connection / client-generated ID.
        """
        # Update the session's voice_session_id field
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
                # Store reverse mapping: voice -> session_id (short TTL)
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

        Looks up the conversation session linked to the given voice_session_id
        and returns the full ConversationContext for seamless resume.

        NFR6 SLA: returns within timeout_sec (default 1.0s). Redis GET
        round-trip is ~1–3ms; the asyncio.wait_for() guard enforces the SLA
        even if Redis is slow or the in-memory path is unexpectedly blocked.

        Args:
            voice_session_id: Client WebRTC peer-connection ID.
            timeout_sec:      Hard deadline for the lookup (default 1.0s).

        Returns:
            ConversationContext if session found and active, None if unknown.

        Raises:
            SessionExpiredError: If the session exists but last_active_at
                                 exceeds VOICE_SESSION_MAX_STALE_SECONDS.
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

        # * Staleness check — reject sessions idle > 5 minutes
        stale_sec = (datetime.now() - context.last_active_at).total_seconds()
        if stale_sec > self.VOICE_SESSION_MAX_STALE_SECONDS:
            raise SessionExpiredError(voice_session_id, stale_sec)

        logger.info(
            "NFR6: voice session {} rehydrated — session={} stale={:.1f}s",
            voice_session_id, context.session_id, stale_sec,
        )
        return context

    async def deregister_voice_session(self, voice_session_id: str) -> None:
        """
        Remove the voice-session mapping on clean disconnect.

        Should be called when WebRTC peer connection closes cleanly
        (not on drop — only on intentional disconnect).

        Args:
            voice_session_id: Client WebRTC peer-connection ID.
        """
        redis = await self._get_redis()
        if redis:
            await redis.delete(f"voice:{voice_session_id}")
        else:
            self._voice_sessions.pop(voice_session_id, None)

        logger.debug("Voice session deregistered: {}", voice_session_id)

    async def touch_voice_session(self, session_id: str) -> None:
        """
        Refresh last_active_at for a voice session.

        Call this on every incoming voice frame to keep the session
        fresh and prevent premature staleness expiry.

        Args:
            session_id: Conversation session UUID.
        """
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

    # ── private ──────────────────────────────────────────────────────────────

    async def _lookup_voice_session(
        self, voice_session_id: str
    ) -> Optional[ConversationContext]:
        """
        Resolve voice_session_id → session_id → ConversationContext.
        Internal coroutine wrapped by asyncio.wait_for in rehydrate_voice_session().
        """
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

