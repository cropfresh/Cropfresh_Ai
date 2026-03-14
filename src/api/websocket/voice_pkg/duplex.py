"""
Duplex WebSocket Endpoint Utilities (Streaming Pipeline).

Handles full-duplex WebSocket connection with streaming LLM + TTS.
"""

from __future__ import annotations

import base64
from datetime import datetime
from typing import TYPE_CHECKING

from fastapi import WebSocket
from loguru import logger

if TYPE_CHECKING:
    from src.voice.duplex_pipeline import DuplexPipeline

try:
    from src.voice.vad import SileroVAD, VADState, BargeinDetector, bytes_to_wav
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False


async def process_duplex_speech(
    pipeline: "DuplexPipeline",
    audio_buffer: list[bytes],
    language: str,
    websocket: WebSocket,
    send_msg,
) -> None:
    """
    Process buffered speech through the duplex pipeline.

    Streams LLM sentences → TTS audio chunks back to the client.
    """
    if not audio_buffer:
        return

    # Combine and convert audio
    audio = b"".join(audio_buffer)
    if VAD_AVAILABLE:
        audio_wav = bytes_to_wav(audio)
    else:
        audio_wav = audio

    chunk_count = 0
    response_text_parts = []

    try:
        async for audio_chunk in pipeline.process_speech(
            audio_wav, language=language
        ):
            # Send each audio chunk immediately
            await send_msg("response_audio", {
                "audio_base64": audio_chunk.audio_base64,
                "format": audio_chunk.format,
                "sample_rate": audio_chunk.sample_rate,
                "chunk_index": audio_chunk.chunk_index,
                "is_last": audio_chunk.is_last,
            })
            chunk_count += 1

            # Track sentence text for transcript
            if audio_chunk.text and audio_chunk.text not in response_text_parts:
                response_text_parts.append(audio_chunk.text)
                await send_msg("response_sentence", {
                    "text": audio_chunk.text,
                })

        # Signal response complete
        await send_msg("response_end", {
            "chunks_sent": chunk_count,
            "full_text": " ".join(response_text_parts),
        })

    except Exception as e:
        logger.error(f"Duplex speech processing error: {e}")
        await send_msg("error", {"error": str(e)})
