"""
Duplex Turn Runners
===================
Shared turn-processing helpers for the duplex voice pipeline.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, AsyncIterator

from loguru import logger

from .lifecycle import emit_interrupted, finish_turn, update_history
from .models import AudioOutputChunk, PipelineState
from .timing import TurnTiming

if TYPE_CHECKING:
    from .pipeline import DuplexPipeline


LANGUAGE_SWITCH_PATTERN = re.compile(r"\[LANG:([a-z]{2})\]")


async def run_speech_turn(
    pipeline: "DuplexPipeline",
    audio_bytes: bytes,
    language: str,
) -> AsyncIterator[AudioOutputChunk]:
    """Run the speech -> STT -> LLM -> TTS turn."""
    pipeline.reset()
    timing = TurnTiming()
    timing.mark_transcription_started()
    await pipeline._emit(PipelineState.TRANSCRIBING, timing=timing.snapshot())

    transcription, detected_language = await pipeline._transcribe(
        audio_bytes,
        language,
    )
    timing.mark_transcription_completed()

    if not transcription or pipeline._cancelled:
        if pipeline._cancelled:
            await emit_interrupted(pipeline, timing)
        await finish_turn(pipeline, timing)
        return

    logger.info(
        "[DuplexPipeline] Transcribed: '{}' (lang={})",
        transcription,
        detected_language,
    )
    async for chunk in run_text_turn(
        pipeline,
        transcription,
        detected_language,
        timing=timing,
        reset_pipeline=False,
    ):
        yield chunk


async def run_text_turn(
    pipeline: "DuplexPipeline",
    text: str,
    language: str,
    *,
    timing: TurnTiming | None = None,
    reset_pipeline: bool = True,
) -> AsyncIterator[AudioOutputChunk]:
    """Run the LLM -> TTS turn for a text input."""
    if not pipeline._llm or not pipeline._tts:
        logger.error("[DuplexPipeline] LLM or TTS not initialized")
        return

    if reset_pipeline:
        pipeline.reset()

    turn_timing = timing or TurnTiming()
    turn_timing.mark_thinking_started()
    await pipeline._emit(
        PipelineState.THINKING,
        text=text,
        language=language,
        timing=turn_timing.snapshot(),
    )

    full_response_parts: list[str] = []
    chunk_index = 0

    try:
        async for sentence_chunk in pipeline._llm.stream_sentences(
            user_message=text,
            conversation_history=pipeline._conversation_history,
            language=language,
        ):
            if pipeline._cancelled:
                await emit_interrupted(pipeline, turn_timing)
                break

            cleaned_text, language, language_switched = _parse_sentence(
                sentence_chunk.text,
                language,
            )
            if language_switched:
                logger.info(
                    "[DuplexPipeline] LLM explicit language switch to: {}",
                    language,
                )
                await pipeline._emit(
                    PipelineState.THINKING,
                    language=language,
                    language_switched=True,
                    timing=turn_timing.snapshot(),
                )

            if not cleaned_text:
                continue

            turn_timing.mark_first_sentence()
            turn_timing.mark_speaking_started()
            full_response_parts.append(cleaned_text)
            await pipeline._emit(
                PipelineState.SPEAKING,
                sentence=cleaned_text,
                timing=turn_timing.snapshot(),
            )

            async for audio_chunk in pipeline._synthesize_sentence(
                cleaned_text,
                language,
                chunk_index,
                sentence_chunk.is_final,
            ):
                if pipeline._cancelled:
                    await emit_interrupted(pipeline, turn_timing)
                    break

                if turn_timing.first_audio_at is None:
                    turn_timing.mark_first_audio()
                    audio_chunk.timing = turn_timing.snapshot()

                yield audio_chunk
                chunk_index += 1

            if pipeline._cancelled:
                break

    except Exception as exc:
        logger.error("[DuplexPipeline] Pipeline error: {}", exc)
        await finish_turn(pipeline, turn_timing, error=str(exc))
        return

    if full_response_parts:
        update_history(pipeline, text, " ".join(full_response_parts))

    await finish_turn(
        pipeline,
        turn_timing,
        interrupted=pipeline._cancelled,
        chunk_count=chunk_index,
    )


def _parse_sentence(text: str, language: str) -> tuple[str, str, bool]:
    match = LANGUAGE_SWITCH_PATTERN.search(text)
    if not match:
        return text.strip(), language, False

    next_language = match.group(1)
    cleaned = LANGUAGE_SWITCH_PATTERN.sub("", text).strip()
    return cleaned, next_language, True
