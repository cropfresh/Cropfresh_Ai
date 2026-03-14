"""
Duplex Pipeline Processing Mixin
================================
Internal transcription and synthesis logic for the duplex pipeline.
"""

import base64
from typing import AsyncIterator

from loguru import logger

from .models import AudioOutputChunk


class ProcessingMixin:
    """Mixin for internal STT transcription and TTS synthesis logics."""

    async def _transcribe(
        self, audio_bytes: bytes, language: str
    ) -> tuple[str, str]:
        """Transcribe audio and return (text, language)."""
        # self._stt is provided by the main pipeline class
        if not self._stt:
            return "", language

        try:
            # We are currently skipping the actual implementation of transcribe error
            # as it was hard-coded to ignore failure before in duplex_pipeline.py
            # (Wait, actually the STT result has `is_successful`, `text`, `language`).
            result = await self._stt.transcribe(audio_bytes, language=language) # This should work
            if getattr(result, "is_successful", True):  # Default to True if not present
                return getattr(result, "text", ""), getattr(result, "language", language)
            else:
                logger.warning("[DuplexPipeline] STT returned empty result")
                return "", language
        except Exception as e:
            logger.error(f"[DuplexPipeline] STT error: {e}")
            return "", language

    async def _synthesize_sentence(
        self,
        sentence: str,
        language: str,
        start_chunk_index: int,
        is_final_sentence: bool,
    ) -> AsyncIterator[AudioOutputChunk]:
        """
        Synthesize a single sentence and yield audio chunks.
        """
        if not self._tts or not sentence.strip():
            return

        try:
            from src.voice.streaming_tts import CancellationToken

            cancel_token = CancellationToken()

            # If pipeline is cancelled, propagate to TTS
            if self._cancelled:
                cancel_token.cancel()
                return

            chunk_idx = start_chunk_index
            async for audio_chunk in self._tts.synthesize_stream(
                text=sentence,
                language=language,
                cancel_token=cancel_token,
                voice="female",
            ):
                if self._cancelled:
                    cancel_token.cancel()
                    break

                # Encode audio to base64 for WebSocket transport
                audio_b64 = base64.b64encode(audio_chunk.data).decode("utf-8")

                yield AudioOutputChunk(
                    audio_base64=audio_b64,
                    format=audio_chunk.format,
                    sample_rate=audio_chunk.sample_rate,
                    chunk_index=chunk_idx,
                    is_last=audio_chunk.is_last and is_final_sentence,
                    text=sentence,
                )
                chunk_idx += 1

        except Exception as e:
            logger.error(f"[DuplexPipeline] TTS error for '{sentence[:30]}': {e}")
            # Fallback: try full synthesis instead of streaming
            try:
                full_audio = await self._tts.synthesize_full(
                    text=sentence,
                    language=language,
                    voice="female",
                )
                if full_audio:
                    audio_b64 = base64.b64encode(full_audio).decode("utf-8")
                    yield AudioOutputChunk(
                        audio_base64=audio_b64,
                        format="mp3",
                        sample_rate=24000,
                        chunk_index=start_chunk_index,
                        is_last=is_final_sentence,
                        text=sentence,
                    )
            except Exception as e2:
                logger.error(f"[DuplexPipeline] TTS fallback also failed: {e2}")
