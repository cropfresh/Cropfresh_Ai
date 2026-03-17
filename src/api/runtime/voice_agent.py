"""Shared voice-agent builder used by startup and REST fallbacks."""

from __future__ import annotations

from typing import Any

from src.agents.voice import VoiceAgent
from src.tools.weather import WeatherTool
from src.voice import MultiProviderSTT, VoiceEntityExtractor
from src.voice.tts import EdgeTTSProvider


def build_voice_agent(
    llm_provider: Any = None,
    listing_service: Any = None,
    order_service: Any = None,
    matching_agent: Any = None,
    quality_agent: Any = None,
    agronomy_agent: Any = None,
    adcl_agent: Any = None,
    registration_service: Any = None,
    weather_api_key: str = "",
) -> VoiceAgent:
    """Build a shared VoiceAgent with the app's runtime services."""
    stt = MultiProviderSTT(
        use_faster_whisper=True,
        use_indicconformer=False,
        faster_whisper_model="small",
    )
    tts = EdgeTTSProvider()
    extractor = VoiceEntityExtractor(llm_provider=llm_provider)
    weather_tool = WeatherTool(api_key=weather_api_key, use_mock=not bool(weather_api_key))
    return VoiceAgent(
        stt=stt,
        tts=tts,
        entity_extractor=extractor,
        llm_provider=llm_provider,
        listing_service=listing_service,
        order_service=order_service,
        matching_agent=matching_agent,
        quality_agent=quality_agent,
        weather_tool=weather_tool,
        agronomy_agent=agronomy_agent,
        adcl_agent=adcl_agent,
        registration_service=registration_service,
    )
