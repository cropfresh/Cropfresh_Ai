"""
WebSocket Voice Endpoint for CropFresh Voice Agent

Provides real-time bidirectional voice communication via WebSocket.
Integrates with WebRTC for audio streaming and VAD for speech detection.

Features:
- Real-time audio streaming (WebSocket + WebRTC)
- Automatic language detection
- Barge-in support
- Session management
"""

import asyncio
import base64
import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger
from pydantic import BaseModel

# Import voice components
try:
    from src.voice.vad import SileroVAD, VADState, BargeinDetector
    from src.voice.webrtc_transport import WebRTCTransport, WebRTCSignaling, ConnectionState
    from src.voice.stt import MultiProviderSTT, TranscriptionResult
    from src.voice.tts import IndicTTS
    from src.agents.voice_agent import VoiceAgent
    from src.voice.vad import bytes_to_wav
    VAD_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Voice components not fully available: {e}")
    VAD_AVAILABLE = False


# WebSocket message types
class MessageType(str, Enum):
    # Control messages
    INIT = "init"
    READY = "ready"
    START = "start"
    STOP = "stop"
    CLOSE = "close"
    ERROR = "error"
    
    # Audio messages
    AUDIO_CHUNK = "audio_chunk"
    AUDIO_END = "audio_end"
    
    # Voice activity
    VAD_START = "vad_start"
    VAD_END = "vad_end"
    VAD_SPEECH = "vad_speech"
    
    # Transcription
    TRANSCRIPT_PARTIAL = "transcript_partial"
    TRANSCRIPT_FINAL = "transcript_final"
    
    # Response
    RESPONSE_TEXT = "response_text"
    RESPONSE_AUDIO = "response_audio"
    RESPONSE_END = "response_end"
    
    # Barge-in
    BARGEIN = "bargein"
    
    # WebRTC signaling
    WEBRTC_OFFER = "webrtc_offer"
    WEBRTC_ANSWER = "webrtc_answer"
    WEBRTC_ICE = "webrtc_ice"
    
    # Language
    LANGUAGE_DETECTED = "language_detected"
    LANGUAGE_HINT = "language_hint"


class VoiceSession:
    """Manages a voice session with WebSocket and optional WebRTC"""
    
    def __init__(
        self,
        session_id: str,
        user_id: str,
        websocket: WebSocket,
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.websocket = websocket
        self.created_at = datetime.now()
        
        # Components
        self.vad: Optional[SileroVAD] = None
        self.bargein_detector: Optional[BargeinDetector] = None
        self.webrtc: Optional[WebRTCTransport] = None
        self.stt: Optional[MultiProviderSTT] = None
        self.tts: Optional[IndicTTS] = None
        self.voice_agent: Optional[VoiceAgent] = None
        
        # State
        self.is_active = False
        self.is_speaking = False
        self.is_responding = False
        self.detected_language = "auto"
        self.audio_buffer: list[bytes] = []
        
        # WebRTC state
        self.webrtc_connected = False
        
        # Cancellation
        self._cancel_response = False
        
        logger.info(f"Voice session created: {session_id} for user {user_id}")
    
    async def initialize(self) -> None:
        """Initialize session components"""
        if VAD_AVAILABLE:
            # Initialize VAD
            self.vad = SileroVAD()
            await self.vad.initialize()
            
            # Setup barge-in
            self.bargein_detector = BargeinDetector(self.vad)
            self.bargein_detector.on_bargein = self._on_bargein
            
            # Initialize STT
            self.stt = MultiProviderSTT(
                use_faster_whisper=True,
                use_indicconformer=True,
                use_groq=True,
            )
            
            # Initialize TTS
            self.tts = IndicTTS()
            
            # Initialize voice agent
            self.voice_agent = VoiceAgent(stt=self.stt, tts=self.tts)
        
        self.is_active = True
        logger.info(f"Session {self.session_id} initialized")
    
    def _on_bargein(self) -> None:
        """Handle barge-in event"""
        logger.info(f"Barge-in detected for session {self.session_id}")
        self._cancel_response = True
        asyncio.create_task(self._send_message(MessageType.BARGEIN, {}))
    
    async def _send_message(self, msg_type: MessageType, data: dict) -> None:
        """Send message via WebSocket"""
        try:
            message = {
                "type": msg_type.value,
                "timestamp": datetime.now().isoformat(),
                **data,
            }
            await self.websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def handle_audio_chunk(self, audio_b64: str) -> None:
        """Process incoming audio chunk"""
        if not self.is_active or not self.vad:
            return
        
        # Decode audio
        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            logger.error(f"Error decoding audio: {e}")
            return
        
        # Process through VAD
        event = self.vad.process_chunk(audio_bytes)
        
        # Check for barge-in during response
        if self.is_responding and self.bargein_detector:
            self.bargein_detector.check_bargein(event)
        
        # Handle VAD events
        if event.state == VADState.SPEECH_START:
            self.is_speaking = True
            self.audio_buffer = [audio_bytes]
            await self._send_message(MessageType.VAD_START, {
                "timestamp_ms": event.timestamp_ms,
            })
        
        elif event.state == VADState.SPEECH:
            self.audio_buffer.append(audio_bytes)
            await self._send_message(MessageType.VAD_SPEECH, {
                "probability": event.probability,
            })
        
        elif event.state == VADState.SPEECH_END:
            self.is_speaking = False
            await self._send_message(MessageType.VAD_END, {
                "timestamp_ms": event.timestamp_ms,
            })
            
            # Process the collected speech
            await self._process_speech()
    
    async def _process_speech(self) -> None:
        """Process collected speech segment"""
        if not self.audio_buffer or not self.stt or not self.voice_agent:
            return
        
        # Combine audio chunks
        audio = b"".join(self.audio_buffer)
        self.audio_buffer = []
        
        # Add WAV header
        audio_wav = bytes_to_wav(audio) if VAD_AVAILABLE else audio
        
        # Transcribe
        try:
            result = await self.stt.transcribe(audio_wav, language="auto")
            
            if result.is_successful:
                # Update detected language
                if result.language != self.detected_language:
                    self.detected_language = result.language
                    await self._send_message(MessageType.LANGUAGE_DETECTED, {
                        "language": result.language,
                        "confidence": result.confidence,
                    })
                
                # Send transcript
                await self._send_message(MessageType.TRANSCRIPT_FINAL, {
                    "text": result.text,
                    "language": result.language,
                    "confidence": result.confidence,
                    "provider": result.provider,
                })
                
                # Generate response
                await self._generate_response(result.text, result.language)
            
        except Exception as e:
            logger.error(f"Error processing speech: {e}")
            await self._send_message(MessageType.ERROR, {
                "error": str(e),
            })
    
    async def _generate_response(self, text: str, language: str) -> None:
        """Generate and stream voice response"""
        if not self.voice_agent or not self.tts:
            return
        
        self.is_responding = True
        self._cancel_response = False
        
        # Start barge-in monitoring
        if self.bargein_detector:
            self.bargein_detector.start_monitoring()
        
        try:
            # Get agent response
            response = await self.voice_agent.process_text(
                text=text,
                user_id=self.user_id,
                session_id=self.session_id,
                language=language,
            )
            
            if self._cancel_response:
                return
            
            # Send response text
            await self._send_message(MessageType.RESPONSE_TEXT, {
                "text": response.response_text,
                "intent": response.intent,
                "entities": response.entities,
            })
            
            if self._cancel_response:
                return
            
            # Synthesize speech in the SAME language as input
            synthesis = await self.tts.synthesize(
                text=response.response_text,
                language=language,  # Use detected language, not hardcoded
            )
            
            if self._cancel_response:
                return
            
            # Stream audio in chunks
            chunk_size = 4096
            audio = synthesis.audio
            
            for i in range(0, len(audio), chunk_size):
                if self._cancel_response:
                    break
                
                chunk = audio[i:i + chunk_size]
                chunk_b64 = base64.b64encode(chunk).decode("utf-8")
                
                await self._send_message(MessageType.RESPONSE_AUDIO, {
                    "audio_base64": chunk_b64,
                    "format": synthesis.format,
                    "sample_rate": synthesis.sample_rate,
                    "chunk_index": i // chunk_size,
                    "is_last": i + chunk_size >= len(audio),
                })
                
                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.02)
            
            await self._send_message(MessageType.RESPONSE_END, {
                "duration_seconds": synthesis.duration_seconds,
            })
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            await self._send_message(MessageType.ERROR, {
                "error": str(e),
            })
        
        finally:
            self.is_responding = False
            if self.bargein_detector:
                self.bargein_detector.stop_monitoring()
    
    async def handle_webrtc_offer(self, offer: dict) -> dict:
        """Handle WebRTC offer and return answer"""
        if not self.webrtc:
            self.webrtc = WebRTCTransport()
        
        answer = await self.webrtc.create_answer(offer)
        self.webrtc_connected = True
        
        return answer
    
    async def handle_ice_candidate(self, candidate: dict) -> None:
        """Handle ICE candidate"""
        if self.webrtc:
            await self.webrtc.add_ice_candidate(candidate)
    
    async def close(self) -> None:
        """Close the session"""
        self.is_active = False
        
        if self.webrtc:
            await self.webrtc.close()
        
        if self.vad:
            self.vad.reset()
        
        logger.info(f"Session {self.session_id} closed")


# Session manager
class SessionManager:
    """Manages active voice sessions"""
    
    def __init__(self):
        self._sessions: Dict[str, VoiceSession] = {}
    
    def create_session(
        self,
        user_id: str,
        websocket: WebSocket,
    ) -> VoiceSession:
        """Create new voice session"""
        session_id = str(uuid.uuid4())
        session = VoiceSession(session_id, user_id, websocket)
        self._sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[VoiceSession]:
        """Get session by ID"""
        return self._sessions.get(session_id)
    
    async def remove_session(self, session_id: str) -> None:
        """Remove and close session"""
        session = self._sessions.pop(session_id, None)
        if session:
            await session.close()
    
    def get_active_count(self) -> int:
        """Get count of active sessions"""
        return len(self._sessions)


# Router
router = APIRouter(prefix="/api/v1/voice", tags=["voice-realtime"])

# Global session manager
_session_manager = SessionManager()


@router.websocket("/ws")
async def voice_websocket(
    websocket: WebSocket,
    user_id: str = "anonymous",
):
    """
    WebSocket endpoint for real-time voice communication.
    
    Message protocol:
    - Client sends: init, audio_chunk, language_hint, webrtc_offer, webrtc_ice
    - Server sends: ready, vad_start/end, transcript_*, response_*, error
    
    Audio format:
    - 16-bit PCM
    - Mono (1 channel)
    - 16kHz sample rate
    - Encoded as base64 in audio_chunk messages
    """
    await websocket.accept()
    
    session = _session_manager.create_session(user_id, websocket)
    
    try:
        # Initialize session
        await session.initialize()
        
        # Send ready message
        await websocket.send_json({
            "type": MessageType.READY.value,
            "session_id": session.session_id,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Message loop
        while True:
            try:
                message = await websocket.receive_json()
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                break
            
            msg_type = message.get("type")
            
            if msg_type == MessageType.AUDIO_CHUNK.value:
                # Process audio chunk
                audio_b64 = message.get("audio_base64", "")
                await session.handle_audio_chunk(audio_b64)
            
            elif msg_type == MessageType.LANGUAGE_HINT.value:
                # Set language hint
                session.detected_language = message.get("language", "auto")
                logger.info(f"Language hint set: {session.detected_language}")
            
            elif msg_type == MessageType.WEBRTC_OFFER.value:
                # Handle WebRTC offer
                offer = message.get("offer", {})
                answer = await session.handle_webrtc_offer(offer)
                await websocket.send_json({
                    "type": MessageType.WEBRTC_ANSWER.value,
                    "answer": answer,
                })
            
            elif msg_type == MessageType.WEBRTC_ICE.value:
                # Handle ICE candidate
                candidate = message.get("candidate", {})
                await session.handle_ice_candidate(candidate)
            
            elif msg_type == MessageType.STOP.value:
                # Stop the session
                logger.info(f"Session {session.session_id} stopped by client")
                break
            
            elif msg_type == MessageType.CLOSE.value:
                # Close the session
                break
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session.session_id}")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": MessageType.ERROR.value,
                "error": str(e),
            })
        except:
            pass
    
    finally:
        await _session_manager.remove_session(session.session_id)


@router.get("/ws/sessions")
async def get_sessions():
    """Get count of active WebSocket sessions"""
    return {
        "active_sessions": _session_manager.get_active_count(),
    }
