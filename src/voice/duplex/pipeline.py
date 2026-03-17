"""
Duplex Pipeline Core
====================
The main DuplexPipeline orchestrator for CropFresh Voice Agent.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable, Optional

from loguru import logger

from .initializers import InitializersMixin
from .models import AudioOutputChunk, PipelineEvent, PipelineState
from .processing import ProcessingMixin
from .turns import run_speech_turn, run_text_turn


class DuplexPipeline(InitializersMixin, ProcessingMixin):
    """Full-duplex voice pipeline orchestrator."""

    def __init__(
        self,
        llm_provider: str = "groq",
        tts_provider: str = "edge",
        stt_provider: str = "groq",
    ) -> None:
        self._llm_provider = llm_provider
        self._tts_provider = tts_provider
        self._stt_provider = stt_provider
        self._llm = None
        self._stt = None
        self._tts = None
        self._state = PipelineState.IDLE
        self._cancelled = False
        self._conversation_history: list[dict] = []
        self._last_turn_timing: dict[str, float | None] = {}
        self._on_event: Optional[Callable] = None

        logger.info(
            "[DuplexPipeline] Created: llm={}, tts={}, stt={}",
            llm_provider,
            tts_provider,
            stt_provider,
        )

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def last_turn_timing(self) -> dict[str, float | None]:
        return dict(self._last_turn_timing)

    def on_event(self, callback: Callable) -> None:
        self._on_event = callback

    async def _emit(self, state: PipelineState, **data) -> None:
        self._state = state
        if not self._on_event:
            return

        event = PipelineEvent(state=state, data=data)
        if asyncio.iscoroutinefunction(self._on_event):
            await self._on_event(event)
            return
        self._on_event(event)

    async def initialize(self) -> None:
        await self._init_llm()
        await self._init_stt()
        await self._init_tts()
        logger.info("[DuplexPipeline] All components initialized")

    def interrupt(self) -> None:
        if self._state not in (PipelineState.THINKING, PipelineState.SPEAKING):
            return

        self._cancelled = True
        if self._llm:
            self._llm.cancel()
        logger.info("[DuplexPipeline] Pipeline interrupted (barge-in)")

    def reset(self) -> None:
        self._cancelled = False
        self._last_turn_timing = {}
        if self._llm:
            self._llm.reset()
        self._state = PipelineState.IDLE

    async def process_speech(
        self,
        audio_bytes: bytes,
        language: str = "auto",
    ) -> AsyncIterator[AudioOutputChunk]:
        async for chunk in run_speech_turn(self, audio_bytes, language):
            yield chunk

    async def process_text(
        self,
        text: str,
        language: str = "en",
    ) -> AsyncIterator[AudioOutputChunk]:
        async for chunk in run_text_turn(self, text, language):
            yield chunk

    def clear_history(self) -> None:
        self._conversation_history.clear()
        logger.info("[DuplexPipeline] Conversation history cleared")

    async def close(self) -> None:
        self._cancelled = True
        self._conversation_history.clear()
        self._state = PipelineState.IDLE
        logger.info("[DuplexPipeline] Pipeline closed")
