"""
Voice API Endpoints for CropFresh AI Service

REST endpoints for voice processing:
- /api/v1/voice/process - Full voice-in → voice-out
- /api/v1/voice/transcribe - Audio → Text only
- /api/v1/voice/synthesize - Text → Audio only
- /api/v1/voice/languages - Supported languages
"""

import base64
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from pydantic import BaseModel
from loguru import logger

from ...agents.voice_agent import VoiceAgent, VoiceResponse
from ...voice import IndicWhisperSTT, IndicTTS, VoiceEntityExtractor


# Router
router = APIRouter(prefix="/api/v1/voice", tags=["voice"])

# Initialize voice components (lazy loaded)
_voice_agent: Optional[VoiceAgent] = None
_stt: Optional[IndicWhisperSTT] = None
_tts: Optional[IndicTTS] = None


def get_voice_agent() -> VoiceAgent:
    """Get or create voice agent instance"""
    global _voice_agent, _stt, _tts
    
    if _voice_agent is None:
        _stt = IndicWhisperSTT()
        _tts = IndicTTS()
        _voice_agent = VoiceAgent(stt=_stt, tts=_tts)
        logger.info("Voice agent initialized")
    
    return _voice_agent


def get_stt() -> IndicWhisperSTT:
    """Get or create STT instance"""
    global _stt
    if _stt is None:
        _stt = IndicWhisperSTT()
    return _stt


def get_tts() -> IndicTTS:
    """Get or create TTS instance"""
    global _tts
    if _tts is None:
        _tts = IndicTTS()
    return _tts


# ===== Request/Response Models =====

class VoiceProcessResponse(BaseModel):
    """Response from voice processing"""
    transcription: str
    language: str
    intent: str
    entities: dict
    response_text: str
    response_audio_base64: str
    session_id: str
    confidence: float


class TranscribeResponse(BaseModel):
    """Response from transcription"""
    text: str
    language: str
    confidence: float
    duration_seconds: float
    provider: str


class SynthesizeRequest(BaseModel):
    """Request for text-to-speech"""
    text: str
    language: str = "hi"
    voice: str = "default"
    emotion: str = "neutral"


class SynthesizeResponse(BaseModel):
    """Response from synthesis"""
    audio_base64: str
    format: str
    duration_seconds: float


class LanguagesResponse(BaseModel):
    """Response with supported languages"""
    stt_languages: list[str]
    tts_languages: list[str]


# ===== Endpoints =====

@router.post("/process", response_model=VoiceProcessResponse)
async def process_voice(
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, OGG)"),
    user_id: str = Form(..., description="User identifier"),
    session_id: Optional[str] = Form(None, description="Session ID for context"),
    language: str = Form("auto", description="Language code or 'auto'"),
):
    """
    Process voice input and return voice response.
    
    Complete flow:
    1. Transcribe audio to text
    2. Extract intent and entities
    3. Generate response
    4. Synthesize response audio
    
    Returns transcription, intent, entities, and audio response.
    """
    try:
        # Read audio bytes
        audio_bytes = await audio.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        logger.info(f"Processing voice: {len(audio_bytes)} bytes, user={user_id}")
        
        # Process with voice agent
        agent = get_voice_agent()
        result = await agent.process_voice(
            audio=audio_bytes,
            user_id=user_id,
            session_id=session_id,
            language=language,
        )
        
        # Encode audio as base64
        audio_b64 = base64.b64encode(result.response_audio).decode("utf-8")
        
        return VoiceProcessResponse(
            transcription=result.transcription,
            language=result.detected_language,
            intent=result.intent,
            entities=result.entities,
            response_text=result.response_text,
            response_audio_base64=audio_b64,
            session_id=result.session_id,
            confidence=result.confidence,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, OGG)"),
    language: str = Form("auto", description="Language code or 'auto'"),
):
    """
    Transcribe audio to text only.
    
    Uses AI4Bharat IndicWhisper for Indian languages,
    with Groq Whisper API as fallback.
    """
    try:
        audio_bytes = await audio.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        logger.info(f"Transcribing: {len(audio_bytes)} bytes")
        
        stt = get_stt()
        result = await stt.transcribe(audio_bytes, language=language)
        
        return TranscribeResponse(
            text=result.text,
            language=result.language,
            confidence=result.confidence,
            duration_seconds=result.duration_seconds,
            provider=result.provider,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize_speech(request: SynthesizeRequest):
    """
    Synthesize text to speech audio.
    
    Uses AI4Bharat IndicTTS for Indian languages,
    with Edge TTS as fallback.
    """
    try:
        if not request.text:
            raise HTTPException(status_code=400, detail="Empty text")
        
        logger.info(f"Synthesizing: {len(request.text)} chars, lang={request.language}")
        
        tts = get_tts()
        result = await tts.synthesize(
            text=request.text,
            language=request.language,
            voice=request.voice,
            emotion=request.emotion,
        )
        
        audio_b64 = base64.b64encode(result.audio).decode("utf-8")
        
        return SynthesizeResponse(
            audio_base64=audio_b64,
            format=result.format,
            duration_seconds=result.duration_seconds,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/languages", response_model=LanguagesResponse)
async def get_languages():
    """
    Get list of supported languages for STT and TTS.
    """
    stt = get_stt()
    tts = get_tts()
    
    return LanguagesResponse(
        stt_languages=stt.get_supported_languages(),
        tts_languages=tts.get_supported_languages(),
    )


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """
    Clear a voice conversation session.
    """
    agent = get_voice_agent()
    agent.clear_session(session_id)
    
    return {"message": f"Session {session_id} cleared"}


@router.get("/health")
async def voice_health():
    """
    Health check for voice service.
    """
    return {
        "status": "healthy",
        "stt": "indicwhisper",
        "tts": "indictts",
        "languages": ["hi", "kn", "te", "ta", "en"],
    }
