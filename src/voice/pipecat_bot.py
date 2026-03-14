"""
Pipecat Orchestration Layer for CropFresh Voice Agent
=====================================================

This module provides an industry-standard Pipecat pipeline to handle
real-time, full-duplex voice communication using local Bhashini models.

Pipeline:
    WS → VAD(Silero) → LocalBhashiniSTT → LLMUserResponseAggregator
    → CropFreshAgentProcessor → LocalBhashiniTTS → WS
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger  # ← fixes NameError present in original prototype
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_response import LLMUserResponseAggregator
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from src.voice.pipecat.agent_processor import CropFreshAgentProcessor

# Local Bhashini Services
from src.voice.pipecat.stt_service import LocalBhashiniSTTService
from src.voice.pipecat.tts_service import LocalBhashiniTTSService

if TYPE_CHECKING:
    from src.agents.voice_agent import VoiceAgent


async def run_voice_bot(
    websocket,
    session_id: str,
    language: str = "hi",
    voice_agent: "VoiceAgent | None" = None,
) -> None:
    """
    Run the Pipecat pipeline using local FastAPI WebSockets.

    Provides true open-source, local duplex streaming without cloud APIs.
    All processing — VAD, STT, agent routing, TTS — runs on local models.

    Args:
        websocket:    FastAPI WebSocket connection.
        session_id:   Unique session identifier (used for VoiceSession context).
        language:     ISO language code: 'hi' (Hindi), 'kn' (Kannada), 'en'.
        voice_agent:  Optional pre-built VoiceAgent (created if not provided).
    """
    from src.agents.voice_agent import VoiceAgent as _VoiceAgent

    logger.info(f"Starting Pipecat Local Pipeline for session: {session_id}")

    # 1. Transport Layer (Binary WebSocket Audio)
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        ),
    )

    # 2. Local Inference Services
    stt = LocalBhashiniSTTService(language=language)
    tts = LocalBhashiniTTSService(language=language)

    # 3. CropFresh Agent Processor (replaces raw OpenAILLMService → vLLM)
    #    Routes transcribed text through VoiceAgent intent router,
    #    handles multi-turn sessions (listing, registration, buyer matching)
    #    and multi-language template responses (en / hi / kn).
    agent_proc = CropFreshAgentProcessor(
        voice_agent=voice_agent or _VoiceAgent(),
        session_id=session_id,
        language=language,
    )

    # 4. Context Management (accumulates conversation history)
    messages: list[dict] = []
    tma_in = LLMUserResponseAggregator(messages)

    # 5. Pipeline Assembly
    pipeline = Pipeline(
        [
            transport.input(),   # Receive PCM audio frames from browser
            stt,                 # IndicConformer / FasterWhisper STT
            tma_in,              # Aggregate conversation context
            agent_proc,          # Route intent → VoiceAgent response text
            tts,                 # IndicF5 / IndicParler TTS synthesis
            transport.output(),  # Stream PCM audio frames to browser
        ]
    )

    task = PipelineTask(pipeline)
    runner = PipelineRunner()

    logger.info(f"Executing Pipecat pipeline for session {session_id}")
    await runner.run(task)


if __name__ == "__main__":
    logger.error("Do not run directly. This is imported by the FastAPI router.")


