"""
Pipecat Orchestration Layer for CropFresh Voice Agent
=====================================================

This module provides an industry-standard Pipecat pipeline to handle
real-time, full-duplex voice communication using local Bhashini models.
"""

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.aggregators.llm_response import LLMUserResponseAggregator
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)

# Local Bhashini Services
from src.voice.pipecat.stt_service import LocalBhashiniSTTService
from src.voice.pipecat.tts_service import LocalBhashiniTTSService

from pipecat.services.openai import OpenAILLMService


async def run_voice_bot(websocket, session_id: str, language: str = "hi"):
    """
    Run the Pipecat pipeline using local FastAPI WebSockets.
    This provides true open-source, local duplex streaming without paid APIs like Daily.
    """
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

    # The Core Brain: Sarvam-1 via vLLM
    llm = OpenAILLMService(
        api_key="not-needed-for-local", 
        model="sarvam-1",
        base_url="http://localhost:8000/v1"
    )

    # 3. Context Management
    messages = [
        {
            "role": "system",
            "content": f"You are a helpful voice assistant for CropFresh, an agricultural marketplace. Respond in {language} language. Keep response short (1-2 sentences) as it will be spoken. If the farmer asks for prices, make one up for now. Be very polite."
        }
    ]
    
    tma_in = LLMUserResponseAggregator(messages)

    # 4. Pipeline Assembly
    # Notice how clean this is compared to writing raw WebSockets
    pipeline = Pipeline(
        [
            transport.input(),    # UDP Mic In
            stt,                  # Transcribe
            tma_in,               # Accumulate context
            llm,                  # Generate LLM Response stream
            tts,                  # Synthesize Speech
            transport.output(),   # UDP Speaker Out
        ]
    )

    task = PipelineTask(pipeline)
    runner = PipelineRunner()

    logger.info(f"Executing Pipecat pipeline for session {session_id}")
    await runner.run(task)

if __name__ == "__main__":
    logger.error("Do not run directly. This is imported by the FastAPI router.")

