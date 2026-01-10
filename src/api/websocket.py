"""
WebSocket API for Real-time Voice Streaming

Enables continuous voice conversation with:
- Audio chunk streaming from client
- Real-time transcription
- Voice response streaming back
"""

import asyncio
import base64
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger
from pydantic import BaseModel

from ..agents.voice_agent import VoiceAgent
from ..voice import IndicWhisperSTT, IndicTTS


# Router
router = APIRouter(tags=["websocket"])

# Singleton instances
_voice_agent: Optional[VoiceAgent] = None


def get_voice_agent() -> VoiceAgent:
    """Get or create voice agent"""
    global _voice_agent
    if _voice_agent is None:
        _voice_agent = VoiceAgent()
    return _voice_agent


class AudioChunk(BaseModel):
    """Audio chunk from client"""
    type: str  # "audio" | "config" | "end"
    audio_base64: Optional[str] = None
    language: str = "auto"


class TranscriptionMessage(BaseModel):
    """Transcription result to client"""
    type: str  # "transcription" | "response" | "audio" | "error"
    text: Optional[str] = None
    is_final: bool = False
    audio_base64: Optional[str] = None
    intent: Optional[str] = None
    entities: Optional[dict] = None


@router.websocket("/ws/voice/{user_id}")
async def voice_websocket(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for real-time voice streaming.
    
    Protocol:
    1. Client connects
    2. Client sends audio chunks as JSON: {"type": "audio", "audio_base64": "..."}
    3. Server sends back transcriptions and responses
    4. Client sends {"type": "end"} to finish
    
    Messages from server:
    - {"type": "transcription", "text": "...", "is_final": true}
    - {"type": "response", "text": "...", "intent": "..."}
    - {"type": "audio", "audio_base64": "..."}
    - {"type": "error", "text": "..."}
    """
    await websocket.accept()
    
    session_id = str(uuid4())
    audio_buffer = bytearray()
    language = "auto"
    
    logger.info(f"Voice WebSocket connected: user={user_id}, session={session_id}")
    
    try:
        agent = get_voice_agent()
        
        while True:
            # Receive message
            data = await websocket.receive_json()
            msg_type = data.get("type", "")
            
            if msg_type == "config":
                # Configuration message
                language = data.get("language", "auto")
                await websocket.send_json({
                    "type": "config_ack",
                    "language": language,
                    "session_id": session_id,
                })
            
            elif msg_type == "audio":
                # Audio chunk
                audio_b64 = data.get("audio_base64", "")
                if audio_b64:
                    chunk = base64.b64decode(audio_b64)
                    audio_buffer.extend(chunk)
                
                # Process if we have enough audio (e.g., > 1 second at 16kHz, 16-bit)
                # ~32KB for 1 second of audio
                if len(audio_buffer) >= 32000:
                    # Process accumulated audio
                    result = await agent.process_voice(
                        audio=bytes(audio_buffer),
                        user_id=user_id,
                        session_id=session_id,
                        language=language,
                    )
                    
                    # Send transcription
                    await websocket.send_json({
                        "type": "transcription",
                        "text": result.transcription,
                        "is_final": True,
                        "language": result.detected_language,
                    })
                    
                    # Send response text
                    await websocket.send_json({
                        "type": "response",
                        "text": result.response_text,
                        "intent": result.intent,
                        "entities": result.entities,
                    })
                    
                    # Send audio response
                    audio_b64_response = base64.b64encode(result.response_audio).decode()
                    await websocket.send_json({
                        "type": "audio",
                        "audio_base64": audio_b64_response,
                    })
                    
                    # Clear buffer
                    audio_buffer.clear()
            
            elif msg_type == "process":
                # Force process current buffer
                if len(audio_buffer) > 0:
                    result = await agent.process_voice(
                        audio=bytes(audio_buffer),
                        user_id=user_id,
                        session_id=session_id,
                        language=language,
                    )
                    
                    await websocket.send_json({
                        "type": "transcription",
                        "text": result.transcription,
                        "is_final": True,
                    })
                    
                    await websocket.send_json({
                        "type": "response",
                        "text": result.response_text,
                        "intent": result.intent,
                        "entities": result.entities,
                    })
                    
                    audio_b64_response = base64.b64encode(result.response_audio).decode()
                    await websocket.send_json({
                        "type": "audio",
                        "audio_base64": audio_b64_response,
                    })
                    
                    audio_buffer.clear()
            
            elif msg_type == "end":
                # End of conversation
                logger.info(f"Voice WebSocket ended: session={session_id}")
                agent.clear_session(session_id)
                await websocket.send_json({
                    "type": "end_ack",
                    "session_id": session_id,
                })
                break
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "text": f"Unknown message type: {msg_type}",
                })
    
    except WebSocketDisconnect:
        logger.info(f"Voice WebSocket disconnected: session={session_id}")
        agent = get_voice_agent()
        agent.clear_session(session_id)
    
    except Exception as e:
        logger.error(f"Voice WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "text": str(e),
            })
        except:
            pass


@router.websocket("/ws/voice/stream/{user_id}")
async def voice_stream_websocket(websocket: WebSocket, user_id: str):
    """
    Simplified streaming WebSocket for continuous voice.
    
    Client sends raw audio bytes, server responds with audio bytes.
    For simpler integration with mobile apps.
    """
    await websocket.accept()
    
    session_id = str(uuid4())
    language = "hi"  # Default to Hindi
    
    logger.info(f"Voice stream connected: user={user_id}")
    
    try:
        agent = get_voice_agent()
        
        while True:
            # Receive raw audio bytes
            audio_bytes = await websocket.receive_bytes()
            
            if len(audio_bytes) == 0:
                continue
            
            # Process voice
            result = await agent.process_voice(
                audio=audio_bytes,
                user_id=user_id,
                session_id=session_id,
                language=language,
            )
            
            # Send back audio response
            if result.response_audio:
                await websocket.send_bytes(result.response_audio)
    
    except WebSocketDisconnect:
        logger.info(f"Voice stream disconnected: session={session_id}")
    
    except Exception as e:
        logger.error(f"Voice stream error: {e}")
