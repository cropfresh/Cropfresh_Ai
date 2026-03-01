"""Unit tests for VoiceAgent task 4 wiring."""

from types import SimpleNamespace

import pytest

from src.agents.voice_agent import VoiceAgent
from src.voice.entity_extractor import ExtractionResult, VoiceIntent
from src.voice.stt import TranscriptionResult
from src.voice.tts import SynthesisResult


class StubStt:
    def __init__(self, results: list[TranscriptionResult]):
        self._results = results
        self._index = 0

    async def transcribe(self, audio: bytes, language: str = "auto") -> TranscriptionResult:
        result = self._results[self._index]
        self._index += 1
        return result

    def get_supported_languages(self) -> list[str]:
        return ["en", "hi", "kn"]


class StubTts:
    async def synthesize(self, text: str, language: str) -> SynthesisResult:
        return SynthesisResult(
            audio=b"audio-bytes",
            format="wav",
            sample_rate=22050,
            duration_seconds=1.0,
            language=language,
            voice="default",
            provider="stub",
        )


class StubExtractor:
    def __init__(self, results: list[ExtractionResult]):
        self._results = results
        self._index = 0

    async def extract(self, text: str, language: str) -> ExtractionResult:
        result = self._results[self._index]
        self._index += 1
        return result


class StubListingService:
    def __init__(self):
        self.calls: list[dict] = []

    async def create_listing(self, **kwargs):
        self.calls.append(kwargs)
        return {"id": "lst-123"}

    async def get_farmer_listings(self, farmer_id: str):
        return [{"commodity": "tomato", "quantity_kg": 100}]


class StubOrderService:
    async def get_status(self, order_id: str, user_id: str):
        return {"eta": "45 minutes", "order_id": order_id, "user_id": user_id}


@pytest.mark.asyncio
async def test_create_listing_multiturn_collects_missing_fields():
    stt = StubStt(
        [
            TranscriptionResult("I want to sell tomato", "en", 0.95, 1.0, "stub"),
            TranscriptionResult("100 kg", "en", 0.95, 1.0, "stub"),
            TranscriptionResult("price 25", "en", 0.95, 1.0, "stub"),
        ]
    )
    extractor = StubExtractor(
        [
            ExtractionResult(VoiceIntent.CREATE_LISTING, {"crop": "tomato"}, 0.9, "I want to sell tomato", "en"),
            ExtractionResult(VoiceIntent.UNKNOWN, {"quantity": 100, "unit": "kg"}, 0.6, "100 kg", "en"),
            ExtractionResult(VoiceIntent.UNKNOWN, {"asking_price": 25}, 0.6, "price 25", "en"),
        ]
    )
    listing_service = StubListingService()
    agent = VoiceAgent(
        stt=stt,
        tts=StubTts(),
        entity_extractor=extractor,
        listing_service=listing_service,
    )

    first = await agent.process_voice(b"a", user_id="farmer-1", language="en")
    second = await agent.process_voice(b"b", user_id="farmer-1", session_id=first.session_id, language="en")
    third = await agent.process_voice(b"c", user_id="farmer-1", session_id=first.session_id, language="en")

    assert "How many" in first.response_text
    assert "asking price" in second.response_text
    assert "Listing ID: lst-123" in third.response_text
    assert len(listing_service.calls) == 1
    assert listing_service.calls[0]["commodity"] == "tomato"
    assert listing_service.calls[0]["quantity_kg"] == 100.0
    assert listing_service.calls[0]["asking_price_per_kg"] == 25.0


@pytest.mark.asyncio
async def test_check_price_uses_pricing_agent_data():
    pricing_agent = SimpleNamespace()

    async def get_recommendation(crop: str, location: str):
        return SimpleNamespace(
            current_price=32,
            recommended_action="sell",
            reason="Strong demand today.",
        )

    pricing_agent.get_recommendation = get_recommendation
    agent = VoiceAgent(stt=StubStt([]), tts=StubTts(), entity_extractor=StubExtractor([]), pricing_agent=pricing_agent)
    session = agent._get_or_create_session("farmer-2", None, "en")
    template = agent.RESPONSE_TEMPLATES[VoiceIntent.CHECK_PRICE]["en"]

    result = await agent._handle_check_price(template, {"crop": "tomato", "location": "Kolar"}, session)

    assert "32" in result
    assert "Recommendation: sell." in result


@pytest.mark.asyncio
async def test_track_order_uses_order_status_when_available():
    agent = VoiceAgent(
        stt=StubStt([]),
        tts=StubTts(),
        entity_extractor=StubExtractor([]),
        order_service=StubOrderService(),
    )
    session = agent._get_or_create_session("farmer-3", None, "en")
    template = agent.RESPONSE_TEMPLATES[VoiceIntent.TRACK_ORDER]["en"]

    result = await agent._handle_track_order(template, {"order_id": "ord-77"}, session)

    assert "45 minutes" in result
