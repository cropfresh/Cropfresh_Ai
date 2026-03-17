"""REST endpoints for voice processing."""

from __future__ import annotations

import base64

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


@router.post("/process", response_model=VoiceProcessResponse)
async def process_voice(
    request: Request,
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, OGG)"),
    user_id: str = Form(..., description="User identifier"),
    session_id: str | None = Form(None, description="Session ID for context"),
    language: str = Form("auto", description="Language code or 'auto'"),
):
    """Process voice input and return a synthesized voice response."""
    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")
        result = await resolve_voice_agent(request).process_voice(
            audio=audio_bytes,
            user_id=user_id,
            session_id=session_id,
            language=language,
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
