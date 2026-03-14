"""
Duplex Pipeline Core
====================
The main DuplexPipeline orchestrator for CropFresh Voice Agent.
"""

import asyncio
import re
import time
from typing import AsyncIterator, Callable, Optional

from loguru import logger

from .initializers import InitializersMixin
from .models import AudioOutputChunk, PipelineEvent, PipelineState
from .processing import ProcessingMixin


class DuplexPipeline(InitializersMixin, ProcessingMixin):
    """
    Full-duplex voice pipeline orchestrator.

    Manages the flow: Audio → STT → LLM (streaming) → TTS (streaming)
    with support for barge-in interruption at any stage.
    """

    def __init__(
        self,
        llm_provider: str = "groq",
        tts_provider: str = "edge",
        stt_provider: str = "groq",
    ) -> None:
        self._llm_provider = llm_provider
        self._tts_provider = tts_provider
        self._stt_provider = stt_provider

        # Components (lazy initialized)
        self._llm = None
        self._stt = None
        self._tts = None

        # State
        self._state = PipelineState.IDLE
        self._cancelled = False
        self._conversation_history: list[dict] = []

        # Callbacks
        self._on_event: Optional[Callable] = None

        logger.info(
            f"[DuplexPipeline] Created: llm={llm_provider}, "
            f"tts={tts_provider}, stt={stt_provider}"
        )

    @property
    def state(self) -> PipelineState:
        """Current pipeline state."""
        return self._state

    def on_event(self, callback: Callable) -> None:
        """Set callback for pipeline events."""
        self._on_event = callback

    async def _emit(self, state: PipelineState, **data) -> None:
        """Emit a pipeline event and update state."""
        self._state = state
        if self._on_event:
            event = PipelineEvent(state=state, data=data)
            if asyncio.iscoroutinefunction(self._on_event):
                await self._on_event(event)
            else:
                self._on_event(event)

    # ──────────────────────────────────────────────────────────
    # Initialization (handled by InitializersMixin)
    # ──────────────────────────────────────────────────────────
    async def initialize(self) -> None:
        """Initialize pipeline components."""
        await self._init_llm()
        await self._init_stt()
        await self._init_tts()
        logger.info("[DuplexPipeline] All components initialized")

    # ──────────────────────────────────────────────────────────
    # Barge-in / Cancellation
    # ──────────────────────────────────────────────────────────

    def interrupt(self) -> None:
        """Cancel both LLM generation and TTS synthesis immediately."""
        if self._state in (PipelineState.THINKING, PipelineState.SPEAKING):
            self._cancelled = True
            if self._llm:
                self._llm.cancel()
            logger.info("[DuplexPipeline] Pipeline interrupted (barge-in)")

    def reset(self) -> None:
        """Reset pipeline for a new turn."""
        self._cancelled = False
        if self._llm:
            self._llm.reset()
        self._state = PipelineState.IDLE

    # ──────────────────────────────────────────────────────────
    # Main pipeline: Speech → Response Audio
    # ──────────────────────────────────────────────────────────

    async def process_speech(
        self,
        audio_bytes: bytes,
        language: str = "auto",
    ) -> AsyncIterator[AudioOutputChunk]:
        """
        Process speech audio through the full pipeline.
        Audio → STT → LLM (streaming) → TTS (streaming)
        """
        time.time()
        self.reset()

        # ── Stage 1: Transcribe ──
        await self._emit(PipelineState.TRANSCRIBING)
        transcription, detected_language = await self._transcribe(
            audio_bytes, language
        )

        if not transcription or self._cancelled:
            return

        logger.info(
            f"[DuplexPipeline] Transcribed: '{transcription}' "
            f"(lang={detected_language})"
        )

        # ── Stage 2+3: Stream LLM → Stream TTS ──
        async for chunk in self.process_text(
            transcription, detected_language
        ):
            yield chunk

    async def process_text(
        self,
        text: str,
        language: str = "en",
    ) -> AsyncIterator[AudioOutputChunk]:
        """
        Process text through LLM → TTS pipeline (speculative TTS loop).
        """
        if not self._llm or not self._tts:
            logger.error("[DuplexPipeline] LLM or TTS not initialized")
            return

        self.reset()
        start_time = time.time()

        # ── Stage 2: Stream LLM ──
        await self._emit(PipelineState.THINKING, text=text, language=language)

        full_response = ""
        chunk_index = 0

        try:
            async for sentence_chunk in self._llm.stream_sentences(
                user_message=text,
                conversation_history=self._conversation_history,
                language=language,
            ):
                if self._cancelled:
                    await self._emit(PipelineState.INTERRUPTED)
                    break

                # Handle Explicit Language Switch Tag
                lang_match = re.search(r'\[LANG:([a-z]{2})\]', sentence_chunk.text)
                if lang_match:
                    new_lang = lang_match.group(1)
                    language = new_lang  # Overwrite TTS language
                    sentence_chunk.text = sentence_chunk.text.replace(lang_match.group(0), "").strip()
                    logger.info(f"[DuplexPipeline] LLM explicit language switch to: {new_lang}")
                    await self._emit(PipelineState.THINKING, language=new_lang, language_switched=True)

                # Skip if empty after stripping
                if not sentence_chunk.text.strip():
                    continue

                full_response += sentence_chunk.text + " "

                # ── Stage 3: Speculative TTS ──
                await self._emit(
                    PipelineState.SPEAKING,
                    sentence=sentence_chunk.text,
                )

                async for audio_chunk in self._synthesize_sentence(
                    sentence_chunk.text,
                    language,
                    chunk_index,
                    sentence_chunk.is_final,
                ):
                    if self._cancelled:
                        break
                    yield audio_chunk
                    chunk_index += 1

        except Exception as e:
            logger.error(f"[DuplexPipeline] Pipeline error: {e}")
            await self._emit(PipelineState.IDLE, error=str(e))
            return

        # ── Update conversation history ──
        if full_response.strip():
            self._conversation_history.append({"role": "user", "content": text})
            self._conversation_history.append({"role": "assistant", "content": full_response.strip()})
            if len(self._conversation_history) > 20:
                self._conversation_history = self._conversation_history[-20:]

        latency = (time.time() - start_time) * 1000
        logger.info(
            f"[DuplexPipeline] Complete: {chunk_index} audio chunks, "
            f"{latency:.0f}ms total"
        )
        await self._emit(PipelineState.IDLE, latency_ms=latency)

    # ──────────────────────────────────────────────────────────
    # Session management
    # ──────────────────────────────────────────────────────────

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._conversation_history.clear()
        logger.info("[DuplexPipeline] Conversation history cleared")

    async def close(self) -> None:
        """Clean up pipeline resources."""
        self._cancelled = True
        self._conversation_history.clear()
        self._state = PipelineState.IDLE
        logger.info("[DuplexPipeline] Pipeline closed")
