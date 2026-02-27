"""
Agent State Manager
===================
Production-grade state management for multi-agent RAG system.

Provides:
- Conversation memory with Redis/in-memory fallback
- Agent execution context tracking
- User session management
- Tool execution history

Author: CropFresh AI Team
Version: 2.0.0
"""

import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field


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
    
    Usage:
        manager = AgentStateManager()
        session = await manager.create_session(user_id="farmer_123")
        context = await manager.get_context(session.session_id)
        await manager.add_message(session.session_id, Message(...))
    """
    
    # Maximum messages to keep in context window
    MAX_MESSAGES = 50
    
    # Session TTL (24 hours)
    SESSION_TTL = timedelta(hours=24)
    
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
