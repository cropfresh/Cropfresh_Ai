"""
Duplex WebSocket Endpoint Utilities (Streaming Pipeline).

Handles full-duplex WebSocket connection with streaming LLM + TTS.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from fastapi import WebSocket
from loguru import logger

if TYPE_CHECKING:
    from src.voice.duplex_pipeline import DuplexPipeline

try:
    from src.voice.vad import bytes_to_wav
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False

from src.voice.duplex.lifecycle import emit_interrupted, finish_turn, update_history
from src.voice.duplex.models import PipelineState
from src.voice.duplex.timing import TurnTiming


async def process_duplex_speech(
    pipeline: "DuplexPipeline",
    audio_buffer: list[bytes],
    language: str,
    websocket: WebSocket,
    send_msg,
    *,
    transcription: str | None = None,
    detected_language: str | None = None,
) -> dict[str, object] | None:
    """
    Process buffered speech through the duplex pipeline.

    Streams LLM sentences → TTS audio chunks back to the client.
    """
    if not audio_buffer:
        return None

    # Combine and convert audio
    audio = b"".join(audio_buffer)
    if VAD_AVAILABLE:
        audio_wav = bytes_to_wav(audio)
    else:
        audio_wav = audio

    chunk_count = 0
    response_text_parts = []

    try:
        if transcription:
            stream = pipeline.process_text(
                transcription,
                language=detected_language or language,
            )
        else:
            stream = pipeline.process_speech(audio_wav, language=language)

        async for audio_chunk in stream:
            # Send each audio chunk immediately
            payload = {
                "audio_base64": audio_chunk.audio_base64,
                "format": audio_chunk.format,
                "sample_rate": audio_chunk.sample_rate,
                "chunk_index": audio_chunk.chunk_index,
                "is_last": audio_chunk.is_last,
            }
            if audio_chunk.timing:
                payload["timing"] = audio_chunk.timing
            await send_msg("response_audio", payload)
            chunk_count += 1

            # Track sentence text for transcript
            if audio_chunk.text and audio_chunk.text not in response_text_parts:
                response_text_parts.append(audio_chunk.text)
                await send_msg("response_sentence", {
                    "text": audio_chunk.text,
                })

        # Signal response complete
        payload = {
            "chunks_sent": chunk_count,
            "full_text": " ".join(response_text_parts),
        }
        if pipeline.last_turn_timing:
            payload["timing"] = pipeline.last_turn_timing
        await send_msg("response_end", payload)
        return payload

    except Exception as e:
        logger.error(f"Duplex speech processing error: {e}")
        await send_msg("error", {"error": str(e)})
        return None


async def process_duplex_text_response(
    pipeline: "DuplexPipeline",
    text: str,
    response_text: str,
    language: str,
    websocket: WebSocket,
    send_msg,
) -> dict[str, object] | None:
    """
    Synthesize an already-generated response through the duplex transport.

    This is used by the Sprint 10 voice orchestrator to bypass the LLM stream
    while preserving the websocket event contract and reconnect-safe history.
    """
    del websocket
    if not response_text.strip():
        return None

    pipeline.reset()
    pipeline._last_user_text = text
    timing = TurnTiming()
    timing.mark_thinking_started()
    await pipeline._emit(
        PipelineState.THINKING,
        text=text,
        language=language,
        orchestrated=True,
        timing=timing.snapshot(),
    )

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", response_text.strip())
        if sentence.strip()
    ] or [response_text.strip()]

    chunk_count = 0
    response_text_parts: list[str] = []

    try:
        for index, sentence in enumerate(sentences):
            if pipeline._cancelled:
                await emit_interrupted(pipeline, timing)
                break

            timing.mark_first_sentence()
            timing.mark_speaking_started()
            response_text_parts.append(sentence)
            await pipeline._emit(
                PipelineState.SPEAKING,
                sentence=sentence,
                language=language,
                orchestrated=True,
                timing=timing.snapshot(),
            )
            await send_msg("response_sentence", {"text": sentence})

            async for audio_chunk in pipeline._synthesize_sentence(
                sentence,
                language,
                chunk_count,
                index == len(sentences) - 1,
            ):
                if pipeline._cancelled:
                    await emit_interrupted(pipeline, timing)
                    break

                if timing.first_audio_at is None:
                    timing.mark_first_audio()
                    audio_chunk.timing = timing.snapshot()

                payload = {
                    "audio_base64": audio_chunk.audio_base64,
                    "format": audio_chunk.format,
                    "sample_rate": audio_chunk.sample_rate,
                    "chunk_index": audio_chunk.chunk_index,
                    "is_last": audio_chunk.is_last,
                }
                if audio_chunk.timing:
                    payload["timing"] = audio_chunk.timing
                await send_msg("response_audio", payload)
                chunk_count += 1

            if pipeline._cancelled:
                break

        if response_text_parts:
            pipeline._last_response_text = " ".join(response_text_parts)
            update_history(pipeline, text, pipeline._last_response_text)
        else:
            pipeline._last_response_text = ""

        await finish_turn(
            pipeline,
            timing,
            interrupted=pipeline._cancelled,
            chunk_count=chunk_count,
        )
        payload = {
            "chunks_sent": chunk_count,
            "full_text": pipeline.last_response_text,
            "orchestrated": True,
        }
        if pipeline.last_turn_timing:
            payload["timing"] = pipeline.last_turn_timing
        await send_msg("response_end", payload)
        return payload

    except Exception as e:
        logger.error(f"Duplex text response processing error: {e}")
        await send_msg("error", {"error": str(e)})
        return None
