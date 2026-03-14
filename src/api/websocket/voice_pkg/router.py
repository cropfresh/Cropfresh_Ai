"""
WebSocket Voice Router.

Defines the FastAPI APIRouter and endpoint handlers for voice WebSockets.
"""

import base64
import json
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from src.api.websocket.voice_pkg.duplex import process_duplex_speech
from src.api.websocket.voice_pkg.session import VAD_AVAILABLE, SessionManager

try:
    from src.voice.vad import BargeinDetector, SileroVAD, VADState
except ImportError:
    pass

try:
    from src.voice.duplex_pipeline import DuplexPipeline, PipelineEvent, PipelineState
    DUPLEX_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Duplex pipeline not available: {e}")
    DUPLEX_AVAILABLE = False


router = APIRouter(prefix="/api/v1/voice", tags=["voice-realtime"])

# Global session manager
_session_manager = SessionManager()


@router.websocket("/ws")
async def voice_websocket(
    websocket: WebSocket,
    user_id: str = "anonymous",
    language: str = "hi",
    session_id: Optional[str] = None,
):
    """
    WebSocket endpoint for real-time voice communication.
    """
    await websocket.accept()
    session = _session_manager.create_session(user_id, websocket, session_id)
    logger.info(f"WebSocket accepted for user={user_id} lang={language} session={session.session_id}")

    try:
        await session.initialize()
        await websocket.send_json({"type": "ready", "session_id": session.session_id})

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
                await session.process_speech()
            elif msg_type == "close":
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket session error for {session.session_id}: {e}")
    finally:
        await _session_manager.remove_session(session.session_id)
        logger.info(f"WebSocket session {session.session_id} cleaned up")


@router.websocket("/ws/duplex")
async def voice_duplex_websocket(
    websocket: WebSocket,
    user_id: str = "anonymous",
    language: str = "hi",
    session_id: Optional[str] = None,
):
    """
    Full-duplex WebSocket endpoint with streaming LLM + TTS.
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

    pipeline = DuplexPipeline(llm_provider="groq", tts_provider="edge", stt_provider="groq")
    vad = None
    audio_buffer: list[bytes] = []
    connection_language = language
    last_detected_language = None

    async def send_msg(msg_type: str, data: dict) -> None:
        try:
            await websocket.send_json({
                "type": msg_type,
                "timestamp": datetime.now().isoformat(),
                **data,
            })
        except Exception:
            pass

    async def on_pipeline_event(event: PipelineEvent) -> None:
        nonlocal last_detected_language, connection_language
        if event.state == PipelineState.THINKING:
            lang = event.data.get("language")
            is_switched = event.data.get("language_switched", False)

            if is_switched and lang:
                connection_language = lang
                last_detected_language = lang
                await send_msg("language_detected", {"language": lang, "locked": True})
            elif lang and lang != last_detected_language:
                last_detected_language = lang
                await send_msg("language_detected", {"language": lang})

        await send_msg("pipeline_state", {"state": event.state.value, **event.data})

    pipeline.on_event(on_pipeline_event)

    try:
        await pipeline.initialize()
        if VAD_AVAILABLE:
            try:
                vad = SileroVAD()
                await vad.initialize()
                BargeinDetector(vad)
                logger.info(f"Duplex {session_id}: VAD ready")
            except Exception as e:
                logger.warning(f"Duplex {session_id}: VAD unavailable: {e}")
                vad = None

        await send_msg("ready", {
            "session_id": session_id,
            "mode": "duplex",
            "features": {"vad": vad is not None, "streaming_llm": True, "streaming_tts": True, "bargein": True},
        })

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
                try:
                    audio_bytes = base64.b64decode(msg.get("audio_base64", ""))
                except Exception:
                    continue
                audio_buffer.append(audio_bytes)

                if vad:
                    event = vad.process_chunk(audio_bytes)
                    if pipeline.state == PipelineState.SPEAKING:
                        if event.state in (VADState.SPEECH_START, VADState.SPEECH) and event.probability > 0.7:
                            pipeline.interrupt()
                            await send_msg("bargein", {})
                            audio_buffer = [audio_bytes]

                    if event.state == VADState.SPEECH_END:
                        await process_duplex_speech(pipeline, audio_buffer, connection_language, websocket, send_msg)
                        audio_buffer = []

            elif msg_type == "audio_end":
                if audio_buffer:
                    await process_duplex_speech(pipeline, audio_buffer, connection_language, websocket, send_msg)
                    audio_buffer = []
            elif msg_type == "bargein":
                pipeline.interrupt()
                await send_msg("bargein", {})
            elif msg_type == "language_hint":
                connection_language = msg.get("language", language)
            elif msg_type == "close":
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Duplex session error {session_id}: {e}")
    finally:
        await pipeline.close()
        if vad: vad.reset()
        logger.info(f"Duplex session {session_id} cleaned up")


@router.get("/ws/sessions")
async def get_sessions():
    """Get count of active WebSocket sessions"""
    return {"active_sessions": _session_manager.get_active_count()}
