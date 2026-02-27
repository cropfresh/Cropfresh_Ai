import asyncio
from typing import AsyncGenerator

from loguru import logger
from pipecat.services.ai_services import TTSService
from pipecat.frames.frames import Frame, AudioRawFrame

from src.voice.tts import IndicTTS

class LocalBhashiniTTSService(TTSService):
    """
    Pipecat TTS Service wrapper for CropFresh's Local Bhashini (IndicF5/IndicParler)
    models.
    
    This service takes text chunks yielded by the LLM and synthesizes them
    into `AudioRawFrame`s which are instantly streamed back over WebRTC.
    """

    def __init__(self, language: str = "hi", voice: str = "default", **kwargs):
        super().__init__(**kwargs)
        self.language = language
        self.voice = voice
        
        # Initialize our purely local TTS engine
        self._tts = IndicTTS()
        logger.info(f"Initialized LocalBhashiniTTSService for language: {language}, voice: {voice}")

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """
        Run text-to-speech synthesis on the provided text block.
        Yields raw PCM audio frames for the transport to stream.
        """
        try:
            logger.debug(f"Running local TTS for text: '{text[:30]}...'")
            
            # The synthesize method natively blocks/awaits on the torch inference
            result = await self._tts.synthesize(
                text=text, 
                language=self.language,
                voice=self.voice
            )
            
            # Yield the synthesized audio as an AudioRawFrame to Pipecat
            if result and result.audio:
                # result.audio is WAV bytes (from _array_to_wav).
                # We need to extract just the raw PCM payload. But actually, Pipecat supports standard PCM.
                # However, our IndicTTS currently produces complete .wav buffers containing headers.
                # Let's extract the raw PCM from the WAV to avoid header clicks for continuous streams.
                
                # We know the TTS returns 16-bit PCM at self._tts.OUTPUT_SAMPLE_RATE Hz mono
                import wave
                import io
                
                wav_io = io.BytesIO(result.audio)
                with wave.open(wav_io, 'rb') as wav_file:
                    raw_pcm = wav_file.readframes(wav_file.getnframes())
                
                yield AudioRawFrame(
                    audio=raw_pcm,
                    sample_rate=result.sample_rate,
                    num_channels=1
                )
            else:
                logger.debug("Local TTS returned empty audio.")
                
        except Exception as e:
            logger.error(f"Error in LocalBhashiniTTSService: {e}")
