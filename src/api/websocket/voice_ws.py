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
    from src.voice.tts import IndicTTS, EdgeTTSProvider
    from src.agents.voice_agent import VoiceAgent
    from src.voice.vad import bytes_to_wav
    from src.orchestrator.llm_provider import create_llm_provider
    VAD_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Voice components not fully available: {e}")
    VAD_AVAILABLE = False

# Import duplex pipeline
try:
    from src.voice.duplex_pipeline import (
        DuplexPipeline, PipelineState, PipelineEvent, AudioOutputChunk,
    )
    DUPLEX_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Duplex pipeline not available: {e}")
    DUPLEX_AVAILABLE = False


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
        self.tts: Optional[EdgeTTSProvider] = None
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
        """Initialize session components — STT/TTS always, VAD non-fatal."""
        # STT + TTS always initialize (no GPU, no model cache needed)
        self.stt = MultiProviderSTT(
            use_faster_whisper=True,
            use_indicconformer=False,  # CPU-safe
        )
        self.tts = EdgeTTSProvider()

        # Voice agent (uses default LLM from settings, not hardcoded vLLM)
        self.voice_agent = VoiceAgent(stt=self.stt, tts=self.tts)

        # VAD — non-fatal, falls back to manual chunking
        if VAD_AVAILABLE:
            try:
                self.vad = SileroVAD()
                await self.vad.initialize()
                self.bargein_detector = BargeinDetector(self.vad)
                self.bargein_detector.on_bargein = self._on_bargein
                logger.info(f"Session {self.session_id}: VAD ready")
            except Exception as e:
                logger.warning(f"Session {self.session_id}: VAD unavailable ({e}) — using manual flush")
                self.vad = None

        self.is_active = True
        logger.info(f"Session {self.session_id} initialized (vad={'on' if self.vad else 'off'})")

    
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
    language: str = "hi",
):
    """
    WebSocket endpoint for real-time voice communication.

    Audio format (JSON text frames):
        { "type": "audio_chunk", "audio_base64": "<base64 PCM>" }
        { "type": "audio_end" }   ← flush: trigger STT on buffered audio
        { "type": "close" }       ← graceful close

    Server sends back JSON frames for every pipeline event:
        ready | vad_start | vad_end | transcript_final | response_text | response_audio | response_end | error
    """
    await websocket.accept()

    session = _session_manager.create_session(user_id, websocket)
    logger.info(f"WebSocket accepted for user={user_id} lang={language}")

    try:
        # Initialize STT + TTS + VAD (non-fatal VAD)
        await session.initialize()
        await websocket.send_json({"type": "ready", "session_id": session.session_id})

        # Message loop — keep running until disconnect or close message
        while True:
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                break

            try:
                msg = json.loads(raw)
            except Exception:
                continue

            msg_type = msg.get("type", "")

            if msg_type == "audio_chunk":
                await session.handle_audio_chunk(msg.get("audio_base64", ""))

            elif msg_type == "audio_end":
                # Manual flush — force STT processing without VAD silence detection
                await session._process_speech()

            elif msg_type == "close":
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket session error for {session.session_id}: {e}")
    finally:
        await _session_manager.remove_session(session.session_id)
        logger.info(f"WebSocket session {session.session_id} cleaned up")


# ═══════════════════════════════════════════════════════════════
# Duplex WebSocket Endpoint (Streaming Pipeline)
# ═══════════════════════════════════════════════════════════════


@router.websocket("/ws/duplex")
async def voice_duplex_websocket(
    websocket: WebSocket,
    user_id: str = "anonymous",
    language: str = "hi",
):
    """
    Full-duplex WebSocket endpoint with streaming LLM + TTS.

    This endpoint uses the DuplexPipeline for speculative TTS:
    - LLM streams sentences in real-time
    - Each sentence is immediately synthesized to audio
    - Audio chunks are streamed back as they are generated
    - Barge-in interrupts both LLM and TTS instantly

    Audio format (JSON text frames):
        { "type": "audio_chunk", "audio_base64": "<base64 PCM>" }
        { "type": "audio_end" }   — flush: trigger STT on buffered audio
        { "type": "bargein" }     — interrupt AI response
        { "type": "close" }       — graceful close

    Server sends back JSON frames:
        ready | pipeline_state | transcript_final | response_audio |
        response_sentence | response_end | error
    """
    if not DUPLEX_AVAILABLE:
        await websocket.accept()
        await websocket.send_json({
            "type": "error",
            "error": "Duplex pipeline not available. Missing dependencies.",
        })
        await websocket.close()
        return

    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"Duplex WebSocket accepted: user={user_id}, lang={language}")

    # Initialize the duplex pipeline
    pipeline = DuplexPipeline(
        llm_provider="groq",
        tts_provider="edge",
        stt_provider="groq",
    )

    # VAD for server-side speech detection
    vad = None
    bargein_detector = None
    audio_buffer: list[bytes] = []
    detected_language = language

    async def send_msg(msg_type: str, data: dict) -> None:
        """Send JSON message to client."""
        try:
            await websocket.send_json({
                "type": msg_type,
                "timestamp": datetime.now().isoformat(),
                **data,
            })
        except Exception:
            pass

    # Pipeline event callback
    async def on_pipeline_event(event: PipelineEvent) -> None:
        await send_msg("pipeline_state", {
            "state": event.state.value,
            **event.data,
        })

    pipeline.on_event(on_pipeline_event)

    try:
        # Initialize pipeline components
        await pipeline.initialize()

        # Initialize VAD (non-fatal)
        if VAD_AVAILABLE:
            try:
                vad = SileroVAD()
                await vad.initialize()
                bargein_detector = BargeinDetector(vad)
                logger.info(f"Duplex {session_id}: VAD ready")
            except Exception as e:
                logger.warning(f"Duplex {session_id}: VAD unavailable: {e}")
                vad = None

        await send_msg("ready", {
            "session_id": session_id,
            "mode": "duplex",
            "features": {
                "vad": vad is not None,
                "streaming_llm": True,
                "streaming_tts": True,
                "bargein": True,
            },
        })

        # ── Message loop ──
        while True:
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                break

            try:
                msg = json.loads(raw)
            except Exception:
                continue

            msg_type = msg.get("type", "")

            if msg_type == "audio_chunk":
                # Decode and buffer audio
                try:
                    audio_bytes = base64.b64decode(msg.get("audio_base64", ""))
                except Exception:
                    continue

                audio_buffer.append(audio_bytes)

                # Process through VAD if available
                if vad:
                    event = vad.process_chunk(audio_bytes)

                    # Check for barge-in during AI response
                    if pipeline.state == PipelineState.SPEAKING:
                        if event.state in (VADState.SPEECH_START, VADState.SPEECH):
                            if event.probability > 0.7:
                                pipeline.interrupt()
                                await send_msg("bargein", {})
                                audio_buffer = [audio_bytes]  # Start new buffer

                    # Speech ended — process the segment
                    if event.state == VADState.SPEECH_END:
                        await _process_duplex_speech(
                            pipeline, audio_buffer, detected_language,
                            websocket, send_msg,
                        )
                        audio_buffer = []

            elif msg_type == "audio_end":
                # Manual flush — process buffered audio
                if audio_buffer:
                    await _process_duplex_speech(
                        pipeline, audio_buffer, detected_language,
                        websocket, send_msg,
                    )
                    audio_buffer = []

            elif msg_type == "bargein":
                # Client-side barge-in signal
                pipeline.interrupt()
                await send_msg("bargein", {})

            elif msg_type == "language_hint":
                detected_language = msg.get("language", language)

            elif msg_type == "close":
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Duplex session error {session_id}: {e}")
    finally:
        await pipeline.close()
        if vad:
            vad.reset()
        logger.info(f"Duplex session {session_id} cleaned up")


async def _process_duplex_speech(
    pipeline: "DuplexPipeline",
    audio_buffer: list[bytes],
    language: str,
    websocket: WebSocket,
    send_msg,
) -> None:
    """
    Process buffered speech through the duplex pipeline.

    Streams LLM sentences → TTS audio chunks back to the client.
    """
    if not audio_buffer:
        return

    # Combine and convert audio
    audio = b"".join(audio_buffer)
    if VAD_AVAILABLE:
        audio_wav = bytes_to_wav(audio)
    else:
        audio_wav = audio

    chunk_count = 0
    response_text_parts = []

    try:
        async for audio_chunk in pipeline.process_speech(
            audio_wav, language=language
        ):
            # Send each audio chunk immediately
            await send_msg("response_audio", {
                "audio_base64": audio_chunk.audio_base64,
                "format": audio_chunk.format,
                "sample_rate": audio_chunk.sample_rate,
                "chunk_index": audio_chunk.chunk_index,
                "is_last": audio_chunk.is_last,
            })
            chunk_count += 1

            # Track sentence text for transcript
            if audio_chunk.text and audio_chunk.text not in response_text_parts:
                response_text_parts.append(audio_chunk.text)
                await send_msg("response_sentence", {
                    "text": audio_chunk.text,
                })

        # Signal response complete
        await send_msg("response_end", {
            "chunks_sent": chunk_count,
            "full_text": " ".join(response_text_parts),
        })

    except Exception as e:
        logger.error(f"Duplex speech processing error: {e}")
        await send_msg("error", {"error": str(e)})


@router.get("/ws/sessions")
async def get_sessions():
    """Get count of active WebSocket sessions"""
    return {
        "active_sessions": _session_manager.get_active_count(),
    }
