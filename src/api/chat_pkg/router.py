"""Shared chat router implementation."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.chat_pkg.models import (
    ChatRequest,
    ChatResponse,
    SessionInfo,
)
from src.api.chat_pkg.session import execute_chat_request, prepare_chat_execution
from src.api.chat_pkg.supervisor import get_supervisor_agent

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Multi-turn chat with shared language-aware session handling."""
    supervisor = await get_supervisor_agent()
    session_id, response = await execute_chat_request(
        supervisor,
        request.message,
        session_id=request.session_id,
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
    """Streaming chat with the same session preparation as the sync route."""
    supervisor = await get_supervisor_agent()
    session_id, normalized_context = await prepare_chat_execution(
        supervisor,
        request.message,
        session_id=request.session_id,
        context=request.context,
    )

    async def generate_stream():
        try:
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

            if supervisor.state_manager:
                response = await supervisor.process_with_session(
                    query=request.message,
                    session_id=session_id,
                )
            else:
                response = await supervisor.process(
                    query=request.message,
                    context=normalized_context,
                )

            yield f"data: {json.dumps({'type': 'agent', 'agent': response.agent_name})}\n\n"

            content = response.content
            for index in range(0, len(content), 50):
                chunk = content[index : index + 50]
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                await asyncio.sleep(0.02)

            yield (
                "data: "
                f"{json.dumps({'type': 'done', 'confidence': response.confidence, 'sources': response.sources, 'steps': response.steps})}\n\n"
            )
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"

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
    """Create a new chat session."""
    supervisor = await get_supervisor_agent()
    if supervisor.state_manager:
        context = await supervisor.state_manager.create_session(user_id=user_id)
        return SessionInfo(
            session_id=context.session_id,
            created_at=context.created_at.isoformat(),
            message_count=len(context.messages),
            entities=context.entities,
        )

    return SessionInfo(
        session_id=str(uuid.uuid4()),
        created_at="",
        message_count=0,
    )


@router.get("/chat/session/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get session information and history."""
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
    """List available specialized agents."""
    supervisor = await get_supervisor_agent()
    return {"agents": supervisor.get_available_agents()}


@router.get("/chat/tools")
async def list_tools():
    """List available tools."""
    from src.tools.registry import get_tool_registry

    registry = get_tool_registry()
    tools = []
    for name in registry.list_tools():
        definition = registry.get_definition(name)
        if definition:
            tools.append(
                {
                    "name": definition.name,
                    "description": definition.description,
                    "category": definition.category,
                }
            )

    return {"tools": tools}
