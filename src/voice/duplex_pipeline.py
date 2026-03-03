"""
Full-Duplex Voice Pipeline for CropFresh Voice Agent
=====================================================

Central orchestrator that wires together:
    VAD → STT → Streaming LLM → Streaming TTS

Features:
- Speculative TTS: sends first sentence to TTS while LLM generates the rest
- Barge-in: cancels TTS+LLM when user interrupts
- Provider fallback: Groq (primary) → Bedrock (fallback)
- Streaming audio output via callback

Usage:
    pipeline = DuplexPipeline()
    await pipeline.initialize()

    # Process a complete speech segment
    async for audio_chunk in pipeline.process_speech(audio_bytes, language="hi"):
        await websocket.send(audio_chunk)

    # Or process text directly
    async for audio_chunk in pipeline.process_text("What is wheat price?", "hi"):
        await websocket.send(audio_chunk)
"""

import asyncio
import base64
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Callable, Optional

from loguru import logger


# ═══════════════════════════════════════════════════════════════
# Pipeline State
# ═══════════════════════════════════════════════════════════════

class PipelineState(str, Enum):
    """Current state of the duplex pipeline."""
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


@dataclass
class PipelineEvent:
    """Event emitted by the pipeline for UI updates."""
    state: PipelineState
    data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AudioOutputChunk:
    """Chunk of synthesized audio to send to the client."""
    audio_base64: str
    format: str = "mp3"
    sample_rate: int = 24000
    chunk_index: int = 0
    is_last: bool = False
    text: str = ""  # The sentence that was synthesized


@dataclass
class PipelineResult:
    """Result of a complete pipeline run."""
    transcription: str = ""
    language: str = "en"
    response_text: str = ""
    audio_chunks_sent: int = 0
    was_interrupted: bool = False
    latency_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════
# Duplex Pipeline
# ═══════════════════════════════════════════════════════════════

class DuplexPipeline:
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
    # Initialization
    # ──────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Initialize pipeline components."""
        # Initialize LLM
        await self._init_llm()

        # Initialize STT
        await self._init_stt()

        # Initialize TTS
        await self._init_tts()

        logger.info("[DuplexPipeline] All components initialized")

    async def _init_llm(self) -> None:
        """Initialize the LLM streaming provider."""
        try:
            if self._llm_provider == "groq":
                from .groq_streaming import GroqLLMStreaming
                self._llm = GroqLLMStreaming()
                logger.info("[DuplexPipeline] LLM: Groq initialized")
            elif self._llm_provider == "bedrock":
                from .bedrock_streaming import BedrockLLMStreaming
                self._llm = BedrockLLMStreaming()
                logger.info("[DuplexPipeline] LLM: Bedrock initialized")
            else:
                raise ValueError(f"Unknown LLM provider: {self._llm_provider}")
        except Exception as e:
            logger.warning(f"[DuplexPipeline] Primary LLM ({self._llm_provider}) failed: {e}")
            # Fallback
            if self._llm_provider == "groq":
                try:
                    from .bedrock_streaming import BedrockLLMStreaming
                    self._llm = BedrockLLMStreaming()
                    self._llm_provider = "bedrock"
                    logger.info("[DuplexPipeline] LLM: Fell back to Bedrock")
                except Exception as e2:
                    logger.error(f"[DuplexPipeline] Bedrock fallback also failed: {e2}")
                    raise

    async def _init_stt(self) -> None:
        """Initialize the STT provider."""
        try:
            from .stt import MultiProviderSTT
            self._stt = MultiProviderSTT(
                use_faster_whisper=True,
                use_indicconformer=False,
            )
            logger.info("[DuplexPipeline] STT: MultiProvider initialized")
        except Exception as e:
            logger.error(f"[DuplexPipeline] STT init failed: {e}")
            raise

    async def _init_tts(self) -> None:
        """Initialize the TTS streaming provider."""
        try:
            from .streaming_tts import StreamingTTS, StreamingTTSProvider
            provider = StreamingTTSProvider.EDGE
            self._tts = StreamingTTS(preferred_provider=provider)
            logger.info("[DuplexPipeline] TTS: Edge TTS initialized")
        except Exception as e:
            logger.error(f"[DuplexPipeline] TTS init failed: {e}")
            raise

    # ──────────────────────────────────────────────────────────
    # Barge-in / Cancellation
    # ──────────────────────────────────────────────────────────

    def interrupt(self) -> None:
        """
        Interrupt the current pipeline (barge-in).

        Cancels both LLM generation and TTS synthesis immediately.
        """
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

        Flow: Audio → STT → LLM (streaming) → TTS (streaming)

        Args:
            audio_bytes: Raw audio bytes (WAV format).
            language: Language code or "auto" for detection.

        Yields:
            AudioOutputChunk with synthesized response audio.
        """
        start_time = time.time()
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
        Process text through LLM → TTS pipeline.

        This is the core speculative TTS loop:
        1. LLM streams sentences
        2. Each sentence is immediately sent to TTS
        3. TTS audio chunks are yielded as they are generated

        Args:
            text: User's transcribed text.
            language: Detected language code.

        Yields:
            AudioOutputChunk with synthesized audio.
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

                full_response += sentence_chunk.text + " "

                # ── Stage 3: Speculative TTS ──
                await self._emit(
                    PipelineState.SPEAKING,
                    sentence=sentence_chunk.text,
                )

                # Synthesize this sentence immediately
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
            self._conversation_history.append(
                {"role": "user", "content": text}
            )
            self._conversation_history.append(
                {"role": "assistant", "content": full_response.strip()}
            )
            # Keep history bounded
            if len(self._conversation_history) > 12:
                self._conversation_history = self._conversation_history[-12:]

        latency = (time.time() - start_time) * 1000
        logger.info(
            f"[DuplexPipeline] Complete: {chunk_index} audio chunks, "
            f"{latency:.0f}ms total"
        )

        await self._emit(PipelineState.IDLE, latency_ms=latency)

    # ──────────────────────────────────────────────────────────
    # Internal: STT
    # ──────────────────────────────────────────────────────────

    async def _transcribe(
        self, audio_bytes: bytes, language: str
    ) -> tuple[str, str]:
        """Transcribe audio and return (text, language)."""
        if not self._stt:
            return "", language

        try:
            result = await self._stt.transcribe(audio_bytes, language=language)
            if result.is_successful:
                return result.text, result.language
            else:
                logger.warning("[DuplexPipeline] STT returned empty result")
                return "", language
        except Exception as e:
            logger.error(f"[DuplexPipeline] STT error: {e}")
            return "", language

    # ──────────────────────────────────────────────────────────
    # Internal: Streaming TTS for a single sentence
    # ──────────────────────────────────────────────────────────

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
            from .streaming_tts import CancellationToken

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
