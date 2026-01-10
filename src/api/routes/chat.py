"""
Chat API Routes
===============
Multi-turn conversation API with streaming support.

Provides:
- /chat - Multi-turn conversation endpoint
- /chat/stream - Server-Sent Events streaming
- /chat/session - Session management

Author: CropFresh AI Team
Version: 2.0.0
"""

import asyncio
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import get_settings


router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class ChatMessage(BaseModel):
    """Single chat message."""
    role: str  # user, assistant
    content: str


class ChatRequest(BaseModel):
    """Chat request with optional session."""
    message: str
    session_id: Optional[str] = None
    context: Optional[dict] = None


class ChatResponse(BaseModel):
    """Chat response with metadata."""
    message: str
    session_id: str
    agent_used: str
    confidence: float
    sources: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    suggested_actions: list[str] = Field(default_factory=list)


class SessionInfo(BaseModel):
    """Session information."""
    session_id: str
    created_at: str
    message_count: int
    entities: dict = Field(default_factory=dict)


class StreamChunk(BaseModel):
    """Single chunk in streaming response."""
    type: str  # token, tool, agent, done, error
    content: str
    metadata: dict = Field(default_factory=dict)


# =============================================================================
# Global Agent Instance
# =============================================================================

_supervisor_agent = None


async def get_supervisor_agent():
    """Get or create supervisor agent instance."""
    global _supervisor_agent
    
    if _supervisor_agent is None:
        from src.agents.supervisor_agent import SupervisorAgent
        from src.agents.agronomy_agent import AgronomyAgent
        from src.agents.commerce_agent import CommerceAgent
        from src.agents.platform_agent import PlatformAgent
        from src.agents.general_agent import GeneralAgent
        from src.orchestrator.llm_provider import create_llm_provider
        from src.rag.knowledge_base import KnowledgeBase
        from src.memory.state_manager import AgentStateManager
        from src.tools.registry import get_tool_registry
        
        settings = get_settings()
        
        # Create LLM
        llm = None
        if settings.groq_api_key:
            llm = create_llm_provider(
                provider=settings.llm_provider,
                api_key=settings.groq_api_key,
                model=settings.llm_model,
            )
        
        # Create knowledge base
        kb = KnowledgeBase(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        await kb.initialize()
        
        # Create state manager
        state_manager = AgentStateManager()
        
        # Get tool registry
        tool_registry = get_tool_registry()
        
        # Create supervisor
        _supervisor_agent = SupervisorAgent(
            llm=llm,
            tool_registry=tool_registry,
            state_manager=state_manager,
            knowledge_base=kb,
        )
        
        # Create and register specialized agents
        agronomy = AgronomyAgent(
            llm=llm,
            tool_registry=tool_registry,
            state_manager=state_manager,
            knowledge_base=kb,
        )
        
        commerce = CommerceAgent(
            llm=llm,
            tool_registry=tool_registry,
            state_manager=state_manager,
            knowledge_base=kb,
        )
        
        platform = PlatformAgent(
            llm=llm,
            tool_registry=tool_registry,
            state_manager=state_manager,
            knowledge_base=kb,
        )
        
        general = GeneralAgent(
            llm=llm,
            tool_registry=tool_registry,
            state_manager=state_manager,
            knowledge_base=kb,
        )
        
        # Register agents with supervisor
        _supervisor_agent.register_agent("agronomy_agent", agronomy)
        _supervisor_agent.register_agent("commerce_agent", commerce)
        _supervisor_agent.register_agent("platform_agent", platform)
        _supervisor_agent.register_agent("general_agent", general)
        _supervisor_agent.set_fallback_agent(general)
        
        # Initialize all
        await _supervisor_agent.initialize()
    
    return _supervisor_agent


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Multi-turn chat with the AI assistant.
    
    Maintains conversation context across messages.
    Routes to appropriate specialized agent.
    """
    supervisor = await get_supervisor_agent()
    
    # Get or create session
    session_id = request.session_id or str(uuid.uuid4())
    
    # Process with session context
    if request.session_id:
        response = await supervisor.process_with_session(
            query=request.message,
            session_id=session_id,
        )
    else:
        response = await supervisor.process(
            query=request.message,
            context=request.context,
        )
    
    return ChatResponse(
        message=response.content,
        session_id=session_id,
        agent_used=response.agent_name,
        confidence=response.confidence,
        sources=response.sources,
        tools_used=response.tools_used,
        steps=response.steps,
        suggested_actions=response.suggested_actions,
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat with Server-Sent Events.
    
    Returns a stream of response chunks for real-time display.
    """
    supervisor = await get_supervisor_agent()
    session_id = request.session_id or str(uuid.uuid4())
    
    async def generate_stream():
        """Generate SSE stream."""
        import json
        
        try:
            # Send session info
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
            
            # Process query
            response = await supervisor.process(
                query=request.message,
                context=request.context,
            )
            
            # Send agent info
            yield f"data: {json.dumps({'type': 'agent', 'agent': response.agent_name})}\n\n"
            
            # Stream content in chunks
            content = response.content
            chunk_size = 50  # Characters per chunk
            
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                await asyncio.sleep(0.02)  # Slight delay for streaming effect
            
            # Send completion with metadata
            yield f"data: {json.dumps({'type': 'done', 'confidence': response.confidence, 'sources': response.sources, 'steps': response.steps})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/chat/session", response_model=SessionInfo)
async def create_session(user_id: Optional[str] = None):
    """
    Create a new chat session.
    
    Returns session ID for use in subsequent requests.
    """
    supervisor = await get_supervisor_agent()
    
    if supervisor.state_manager:
        context = await supervisor.state_manager.create_session(user_id=user_id)
        return SessionInfo(
            session_id=context.session_id,
            created_at=context.created_at.isoformat(),
            message_count=len(context.messages),
            entities=context.entities,
        )
    
    # Fallback if no state manager
    return SessionInfo(
        session_id=str(uuid.uuid4()),
        created_at="",
        message_count=0,
    )


@router.get("/chat/session/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """
    Get session information and history.
    """
    supervisor = await get_supervisor_agent()
    
    if not supervisor.state_manager:
        raise HTTPException(status_code=500, detail="State manager not initialized")
    
    context = await supervisor.state_manager.get_context(session_id)
    
    if not context:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionInfo(
        session_id=context.session_id,
        created_at=context.created_at.isoformat(),
        message_count=len(context.messages),
        entities=context.entities,
    )


@router.get("/chat/agents")
async def list_agents():
    """
    List available specialized agents.
    """
    supervisor = await get_supervisor_agent()
    return {
        "agents": supervisor.get_available_agents(),
    }


@router.get("/chat/tools")
async def list_tools():
    """
    List available tools.
    """
    from src.tools.registry import get_tool_registry
    
    registry = get_tool_registry()
    tools = []
    
    for name in registry.list_tools():
        defn = registry.get_definition(name)
        if defn:
            tools.append({
                "name": defn.name,
                "description": defn.description,
                "category": defn.category,
            })
    
    return {"tools": tools}
