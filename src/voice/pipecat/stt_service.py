import asyncio
from typing import AsyncGenerator

from loguru import logger
from pipecat.services.ai_services import STTService
from pipecat.frames.frames import Frame, TextFrame, TranscriptionFrame

from src.voice.stt import MultiProviderSTT

class LocalBhashiniSTTService(STTService):
    """
    Pipecat STT Service wrapper for CropFresh's Local Bhashini (IndicConformer)
    and Faster-Whisper models.
    """

    def __init__(self, language: str = "auto", **kwargs):
        super().__init__(**kwargs)
        self.language = language
        
        # Initialize our local STT engine (no cloud fallbacks)
        self._stt = MultiProviderSTT(
            use_faster_whisper=True,
            use_indicconformer=True,
            use_groq=False,  # Enforce local
        )
        logger.info(f"Initialized LocalBhashiniSTTService for language: {language}")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """
        Run speech-to-text on the provided audio buffer representing a single utterance.
        Called automatically by Pipecat when Silero VAD detects the end of speech.
        """
        try:
            logger.debug(f"Running local STT on {len(audio)} bytes of audio...")
            
            # The transcribe method in our STT engine takes bytes and language
            result = await self._stt.transcribe(audio, language=self.language)
            
            if result.is_successful and result.text.strip():
                logger.info(f"[Local STT] Transcribed text: {result.text}")
                
                # Yield the transcription as a TextFrame / TranscriptionFrame so Pipecat
                # can pass it downstream to the LLM (Sarvam-1)
                yield TranscriptionFrame(text=result.text, user_id="user", timestamp="")
                yield TextFrame(text=result.text)
            else:
                logger.debug("Local STT returned empty or failed result for this utterance.")
                
        except Exception as e:
            logger.error(f"Error in LocalBhashiniSTTService: {e}")
