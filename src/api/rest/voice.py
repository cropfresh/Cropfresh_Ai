"""REST endpoints for voice processing."""

from __future__ import annotations

import base64
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from loguru import logger

from src.api.rest.voice_models import (
    LanguagesResponse,
    SynthesizeRequest,
    SynthesizeResponse,
    TranscribeResponse,
    VoiceProcessResponse,
)
from src.api.rest.voice_runtime import resolve_stt, resolve_tts, resolve_voice_agent

router = APIRouter(prefix="/api/v1/voice", tags=["voice"])


async def _workflow_context(voice_agent, session_id: str) -> dict[str, Any]:
    session = voice_agent.get_session(session_id) if hasattr(voice_agent, "get_session") else None
    context = getattr(session, "context", {}) if session is not None else {}
    payload: dict[str, Any] = {
        "last_listing_id": context.get("last_listing_id"),
        "pending_intent": context.get("pending_intent"),
        "active_speaker_id": context.get("active_speaker_id"),
        "known_speakers": context.get("known_speakers") or [],
    }
    state_manager = getattr(voice_agent, "state_manager", None)
    if state_manager is None:
        return payload

    persisted = await state_manager.get_context(session_id)
    if persisted is None:
        return payload

    payload["active_speaker_id"] = persisted.active_speaker_id or payload["active_speaker_id"]
    payload["known_speakers"] = sorted(persisted.speaker_profiles.keys())
    return payload


@router.post("/process", response_model=VoiceProcessResponse)
async def process_voice(
    request: Request,
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, OGG)"),
    user_id: str = Form(..., description="User identifier"),
    session_id: str | None = Form(None, description="Session ID for context"),
    language: str = Form("auto", description="Language code or 'auto'"),
    speaker_id: str | None = Form(None, description="Stable speaker ID for grouped turns"),
    speaker_label: str | None = Form(None, description="Human-readable speaker label"),
    speaker_role: str | None = Form(None, description="Speaker role such as farmer or buyer"),
):
    """Process voice input and return a synthesized voice response."""
    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")
        voice_agent = resolve_voice_agent(request)
        result = await voice_agent.process_voice(
            audio=audio_bytes,
            user_id=user_id,
            session_id=session_id,
            language=language,
            speaker_id=speaker_id,
            speaker_label=speaker_label,
            speaker_role=speaker_role,
        )
        return VoiceProcessResponse(
            transcription=result.transcription,
            language=result.detected_language,
            intent=result.intent,
            entities=result.entities,
            response_text=result.response_text,
            response_audio_base64=base64.b64encode(result.response_audio).decode("utf-8"),
            session_id=result.session_id,
            confidence=result.confidence,
            workflow_context=await _workflow_context(voice_agent, result.session_id),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Voice processing error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    request: Request,
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, OGG)"),
    language: str = Form("auto", description="Language code or 'auto'"),
):
    """Transcribe audio to text only."""
    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")
        result = await resolve_stt(request).transcribe(audio_bytes, language=language)
        return TranscribeResponse(
            text=result.text,
            language=result.language,
            confidence=result.confidence,
            duration_seconds=result.duration_seconds,
            provider=result.provider,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Transcription error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize_speech(request: Request, payload: SynthesizeRequest):
    """Synthesize text to speech audio."""
    try:
        if not payload.text:
            raise HTTPException(status_code=400, detail="Empty text")
        result = await resolve_tts(request).synthesize(
            text=payload.text,
            language=payload.language,
            voice=payload.voice,
            emotion=payload.emotion,
        )
        return SynthesizeResponse(
            audio_base64=base64.b64encode(result.audio).decode("utf-8"),
            format=result.format,
            duration_seconds=result.duration_seconds,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Synthesis error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/languages", response_model=LanguagesResponse)
async def get_languages(request: Request):
    """Return STT and TTS supported language lists."""
    stt = resolve_stt(request)
    tts = resolve_tts(request)
    return LanguagesResponse(
        stt_languages=stt.get_supported_languages(),
        tts_languages=tts.get_supported_languages(),
    )


@router.delete("/session/{session_id}")
async def clear_session(request: Request, session_id: str):
    """Clear a voice conversation session."""
    resolve_voice_agent(request).clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}


@router.get("/health")
async def voice_health(request: Request):
    """Report voice component health."""
    stt = resolve_stt(request)
    tts = resolve_tts(request)
    return {
        "status": "healthy" if stt.get_available_providers() else "degraded",
        "stt_providers": stt.get_available_providers(),
        "tts_provider": tts.__class__.__name__,
        "languages": stt.get_supported_languages()[:5],
        "version": "1.0.0",
    }
