"""
WebSocket Voice Router.

Defines the FastAPI APIRouter and endpoint handlers for voice WebSockets.
"""

import asyncio
import base64
import json
import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from src.api.config import get_settings
from src.api.websocket.voice_pkg.duplex import process_duplex_speech
from src.api.websocket.voice_pkg.recovery import apply_recovery_context, resolve_duplex_session
from src.api.websocket.voice_pkg.session import VAD_AVAILABLE, SessionManager
from src.memory.state_pkg.models import VoicePlaybackState, VoiceTurn
from src.voice.semantic_endpointing import evaluate_semantic_flush

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
    settings = get_settings()
    state_manager = getattr(getattr(websocket, "app", None), "state", None)
    state_manager = getattr(state_manager, "state_manager", None)
    llm_provider = getattr(getattr(websocket, "app", None), "state", None)
    llm_provider = getattr(llm_provider, "llm", None)
    reconnect_token = websocket.query_params.get("reconnect_token")
    heartbeat_interval_ms = settings.voice_heartbeat_interval_ms
    dead_peer_timeout_ms = settings.voice_dead_peer_timeout_ms
    recovery_ttl_ms = int((getattr(state_manager, "VOICE_SESSION_MAX_STALE_SECONDS", 300.0)) * 1000)

    session_id, recovery_context, recovered, recovery_outcome = await resolve_duplex_session(
        state_manager=state_manager,
        requested_session_id=session_id,
        user_id=user_id,
        reconnect_token=reconnect_token,
        language=language,
        transport_mode="duplex_ws",
    )
    logger.info(
        "Duplex WebSocket accepted: user={} lang={} session={} recovered={} outcome={}",
        user_id,
        language,
        session_id,
        recovered,
        recovery_outcome,
    )

    pipeline = DuplexPipeline(llm_provider="groq", tts_provider="edge", stt_provider="groq")
    vad = None
    audio_buffer: list[bytes] = []
    pending_audio: bytes | None = None
    pending_hold_started_at: float | None = None
    connection_language = recovery_context.language if recovery_context else language
    last_detected_language = connection_language
    last_client_activity = time.monotonic()
    clean_disconnect = False
    recovered_turn_count = 0

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

        if state_manager is not None:
            await state_manager.update_voice_runtime(
                session_id,
                playback_state=VoicePlaybackState(event.state.value),
                language=connection_language,
            )
        await send_msg("pipeline_state", {"state": event.state.value, **event.data})

    pipeline.on_event(on_pipeline_event)

    async def persist_turn(semantic_hold_ms: int | None = None) -> None:
        if state_manager is None or not pipeline.last_user_text or not pipeline.last_response_text:
            return

        timing = dict(pipeline.last_turn_timing)
        if semantic_hold_ms is not None:
            timing["semantic_hold_ms"] = semantic_hold_ms
            timing["continuity_gap_ms"] = semantic_hold_ms

        await state_manager.append_recent_voice_turn(
            session_id,
            VoiceTurn(
                turn_id=pipeline.last_turn_id,
                user_text=pipeline.last_user_text,
                assistant_text=pipeline.last_response_text,
                language=connection_language,
                interrupted=timing.get("interrupted_ms") is not None,
                timing=timing,
            ),
        )
        await state_manager.update_voice_runtime(
            session_id,
            playback_state=VoicePlaybackState.IDLE,
            pending_transcript=None,
            pending_segment_id=None,
            last_turn_id=pipeline.last_turn_id,
            language=connection_language,
        )

    async def handle_segment(*, force_flush: bool, segment_id: str | None = None) -> None:
        nonlocal audio_buffer, pending_audio, pending_hold_started_at, connection_language
        current_audio = b"".join(audio_buffer)
        combined_audio = b"".join(part for part in [pending_audio, current_audio] if part)
        audio_buffer = []
        if not combined_audio:
            return

        semantic_enabled = settings.voice_semantic_endpointing_enabled and not force_flush
        semantic_hold_ms: int | None = None
        transcript: str | None = None
        detected_language = connection_language

        if semantic_enabled:
            transcript, detected_language = await pipeline._transcribe(combined_audio, connection_language)
            if detected_language:
                connection_language = detected_language

            decision = await evaluate_semantic_flush(
                transcript=transcript,
                language=detected_language,
                llm_provider=llm_provider,
                enabled=settings.voice_semantic_endpointing_enabled,
                timeout_ms=settings.voice_semantic_timeout_ms,
                max_hold_ms=settings.voice_semantic_hold_max_ms,
            )
            if pending_hold_started_at is not None:
                semantic_hold_ms = int((time.monotonic() - pending_hold_started_at) * 1000)

            if not decision.should_flush:
                pending_audio = combined_audio
                pending_hold_started_at = pending_hold_started_at or time.monotonic()
                semantic_hold_ms = int((time.monotonic() - pending_hold_started_at) * 1000)
                if semantic_hold_ms < settings.voice_semantic_hold_max_ms:
                    if state_manager is not None:
                        await state_manager.update_voice_runtime(
                            session_id,
                            pending_transcript=decision.transcript,
                            pending_segment_id=segment_id,
                            playback_state=VoicePlaybackState.LISTENING,
                            language=connection_language,
                        )
                    await send_msg(
                        "pipeline_state",
                        {
                            "state": PipelineState.LISTENING.value,
                            "semantic_hold_ms": semantic_hold_ms,
                            "semantic_reason": decision.reason,
                            "pending_transcript": decision.transcript,
                        },
                    )
                    return

        result = await process_duplex_speech(
            pipeline,
            [combined_audio],
            connection_language,
            websocket,
            send_msg,
            transcription=transcript,
            detected_language=detected_language,
        )
        if result is not None and pending_hold_started_at is not None:
            semantic_hold_ms = int((time.monotonic() - pending_hold_started_at) * 1000)
        await persist_turn(semantic_hold_ms)
        pending_audio = None
        pending_hold_started_at = None

    async def heartbeat_watchdog() -> None:
        nonlocal last_client_activity
        while True:
            await asyncio.sleep(max(1.0, heartbeat_interval_ms / 1000))
            idle_ms = (time.monotonic() - last_client_activity) * 1000
            if idle_ms <= dead_peer_timeout_ms:
                continue
            logger.info("Duplex session {} exceeded heartbeat deadline at {:.0f}ms", session_id, idle_ms)
            if state_manager is not None:
                await state_manager.update_voice_runtime(
                    session_id,
                    playback_state=VoicePlaybackState.IDLE,
                )
            await send_msg("error", {"error": "heartbeat_timeout"})
            await websocket.close(code=1011, reason="heartbeat_timeout")
            return

    heartbeat_task = asyncio.create_task(heartbeat_watchdog())

    try:
        await pipeline.initialize()
        recovered_turn_count = apply_recovery_context(
            pipeline,
            recovery_context if recovered else None,
        )
        if VAD_AVAILABLE:
            try:
                vad = SileroVAD()
                await vad.initialize()
                BargeinDetector(vad)
                logger.info(f"Duplex {session_id}: VAD ready")
            except Exception as e:
                logger.warning(f"Duplex {session_id}: VAD unavailable: {e}")
                vad = None

        if state_manager is not None:
            await state_manager.update_voice_runtime(
                session_id,
                transport_mode="duplex_ws",
                language=connection_language,
                playback_state=VoicePlaybackState.RECOVERING if recovered else VoicePlaybackState.IDLE,
                reconnect_token=reconnect_token,
            )
        await send_msg("ready", {
            "session_id": session_id,
            "mode": "duplex",
            "recovered": recovered,
            "recovered_turn_count": recovered_turn_count,
            "heartbeat_interval_ms": heartbeat_interval_ms,
            "session_recovery_ttl_ms": recovery_ttl_ms,
            "recovery_outcome": recovery_outcome,
            "features": {
                "vad": vad is not None,
                "streaming_llm": True,
                "streaming_tts": True,
                "bargein": True,
                "heartbeat": True,
                "semantic_endpointing": settings.voice_semantic_endpointing_enabled,
            },
        })

        while True:
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                break

            last_client_activity = time.monotonic()
            if state_manager is not None:
                await state_manager.touch_voice_session(session_id)

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
                            await send_msg("bargein", {"session_id": session_id})
                            audio_buffer = [audio_bytes]

                    if event.state == VADState.SPEECH_END:
                        await handle_segment(force_flush=False, segment_id=getattr(event, "segment_id", None))

            elif msg_type == "audio_end":
                if audio_buffer or pending_audio:
                    await handle_segment(force_flush=True)
            elif msg_type == "bargein":
                pipeline.interrupt()
                await send_msg("bargein", {"session_id": session_id})
            elif msg_type == "heartbeat":
                if state_manager is not None:
                    await state_manager.touch_voice_session(session_id, heartbeat=True)
                await send_msg("heartbeat_ack", {
                    "session_id": session_id,
                    "heartbeat_interval_ms": heartbeat_interval_ms,
                    "session_recovery_ttl_ms": recovery_ttl_ms,
                })
            elif msg_type == "language_hint":
                connection_language = msg.get("language", language)
                if state_manager is not None:
                    await state_manager.update_voice_runtime(
                        session_id,
                        language=connection_language,
                    )
            elif msg_type == "close":
                clean_disconnect = True
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Duplex session error {session_id}: {e}")
    finally:
        heartbeat_task.cancel()
        if clean_disconnect and state_manager is not None:
            await state_manager.deregister_voice_session(session_id)
        elif state_manager is not None:
            await state_manager.update_voice_runtime(
                session_id,
                playback_state=VoicePlaybackState.IDLE,
            )
        await pipeline.close()
        if vad:
            vad.reset()
        logger.info(f"Duplex session {session_id} cleaned up")


@router.get("/ws/sessions")
async def get_sessions():
    """Get count of active WebSocket sessions"""
    return {"active_sessions": _session_manager.get_active_count()}
