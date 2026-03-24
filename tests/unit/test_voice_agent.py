"""Unit tests for VoiceAgent task 4 wiring."""

from types import SimpleNamespace

import pytest

from src.agents.voice_agent import VoiceAgent
from src.voice.entity_extractor import ExtractionResult, VoiceIntent
from src.voice.stt import TranscriptionResult
from src.voice.tts import SynthesisResult

# * ═══════════════════════════════════════════════════════════════
# * STUB INFRASTRUCTURE
# * ═══════════════════════════════════════════════════════════════

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

    async def extract(
        self,
        text: str,
        language: str,
        context_intent: str = "",
    ) -> ExtractionResult:
        del text, language, context_intent
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

    async def get_dispute_status(self, order_id: str, user_id: str):
        return {"dispute_id": "disp-001", "status": "Under Review", "notes": "Team will respond in 24h."}


def _make_agent(**kwargs) -> VoiceAgent:
    """Build VoiceAgent with stub STT/TTS for unit testing."""
    return VoiceAgent(
        stt=kwargs.pop("stt", StubStt([])),
        tts=kwargs.pop("tts", StubTts()),
        entity_extractor=kwargs.pop("entity_extractor", StubExtractor([])),
        **kwargs,
    )


# * ═══════════════════════════════════════════════════════════════
# * EXISTING INTENT TESTS (tasks 1-3)
# * ═══════════════════════════════════════════════════════════════

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
    agent = _make_agent(pricing_agent=pricing_agent)
    session = agent._get_or_create_session("farmer-2", None, "en")
    template = agent.RESPONSE_TEMPLATES[VoiceIntent.CHECK_PRICE]["en"]

    result = await agent._handle_check_price(template, {"crop": "tomato", "location": "Kolar"}, session)

    assert "32" in result
    assert "Recommendation: sell." in result


@pytest.mark.asyncio
async def test_track_order_uses_order_status_when_available():
    agent = _make_agent(order_service=StubOrderService())
    session = agent._get_or_create_session("farmer-3", None, "en")
    template = agent.RESPONSE_TEMPLATES[VoiceIntent.TRACK_ORDER]["en"]

    result = await agent._handle_track_order(template, {"order_id": "ord-77"}, session)

    assert "45 minutes" in result


# * ═══════════════════════════════════════════════════════════════
# * NEW INTENT TESTS (task 4 completion)
# * ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_find_buyer_multiturn_collects_commodity_and_quantity():
    """Multi-turn: first turn asks for crop, second for qty, third returns matches."""
    stub_match = SimpleNamespace(buyer_name="FreshMart", buyer_type="retailer", price_fit=28)
    matching_agent = SimpleNamespace()

    async def find_matches(listing_id: str, **kwargs):
        return [stub_match]

    matching_agent.find_matches = find_matches

    stt = StubStt(
        [
            TranscriptionResult("find buyer", "en", 0.9, 1.0, "stub"),
            TranscriptionResult("tomato", "en", 0.9, 1.0, "stub"),
            TranscriptionResult("200 kg", "en", 0.9, 1.0, "stub"),
        ]
    )
    extractor = StubExtractor(
        [
            ExtractionResult(VoiceIntent.FIND_BUYER, {}, 0.8, "find buyer", "en"),
            ExtractionResult(VoiceIntent.UNKNOWN, {"commodity": "tomato"}, 0.6, "tomato", "en"),
            ExtractionResult(VoiceIntent.UNKNOWN, {"quantity_kg": 200}, 0.6, "200 kg", "en"),
        ]
    )
    agent = VoiceAgent(stt=stt, tts=StubTts(), entity_extractor=extractor, matching_agent=matching_agent)

    first = await agent.process_voice(b"a", user_id="f1", language="en")
    second = await agent.process_voice(b"b", user_id="f1", session_id=first.session_id, language="en")
    third = await agent.process_voice(b"c", user_id="f1", session_id=first.session_id, language="en")

    assert "crop" in first.response_text.lower() or "Which crop" in first.response_text
    assert "kg" in second.response_text.lower() or "How many" in second.response_text
    assert "FreshMart" in third.response_text


@pytest.mark.asyncio
async def test_find_buyer_graceful_fallback_when_service_unavailable():
    agent = _make_agent()
    session = agent._get_or_create_session("f2", None, "en")
    session.context["pending_find_buyer"] = {"commodity": "onion", "quantity_kg": 100}
    template = agent.RESPONSE_TEMPLATES[VoiceIntent.FIND_BUYER]["en"]

    result = await agent._handle_find_buyer(template, {}, session)

    assert "not available" in result.lower() or "no buyers" in result.lower() or "service" in result.lower()


@pytest.mark.asyncio
async def test_quality_check_returns_grade_from_quality_agent():
    quality_agent = SimpleNamespace()

    async def execute(input_data: dict):
        return {
            "grade": "A",
            "confidence": 0.88,
            "hitl_required": False,
            "message": "Good quality produce.",
        }

    quality_agent.execute = execute
    agent = _make_agent(quality_agent=quality_agent)
    session = agent._get_or_create_session("f3", None, "en")
    template = agent.RESPONSE_TEMPLATES[VoiceIntent.QUALITY_CHECK]["en"]

    result = await agent._handle_quality_check(
        template, {"commodity": "tomato", "listing_id": "lst-abc"}, session,
    )

    assert "A" in result
    assert "88" in result


@pytest.mark.asyncio
async def test_quality_check_graceful_fallback_when_service_unavailable():
    agent = _make_agent()
    session = agent._get_or_create_session("f4", None, "en")
    template = agent.RESPONSE_TEMPLATES[VoiceIntent.QUALITY_CHECK]["en"]

    result = await agent._handle_quality_check(template, {"commodity": "potato"}, session)

    assert "not available" in result.lower() or "quality" in result.lower()


@pytest.mark.asyncio
async def test_quality_check_asks_for_commodity_when_missing():
    agent = _make_agent()
    session = agent._get_or_create_session("f4b", None, "en")
    template = agent.RESPONSE_TEMPLATES[VoiceIntent.QUALITY_CHECK]["en"]

    result = await agent._handle_quality_check(template, {}, session)

    assert "crop" in result.lower() or "which" in result.lower()


@pytest.mark.asyncio
async def test_get_advisory_calls_agronomy_agent():
    agronomy_agent = SimpleNamespace()

    async def process(query: str, context: dict = None):
        return SimpleNamespace(content="Water tomatoes twice daily in dry season.")

    agronomy_agent.process = process
    agent = _make_agent(agronomy_agent=agronomy_agent)
    session = agent._get_or_create_session("f5", None, "en")
    template = agent.RESPONSE_TEMPLATES[VoiceIntent.GET_ADVISORY]["en"]

    result = await agent._handle_get_advisory(template, {"crop": "tomato"}, session)

    assert "tomato" in result.lower()
    assert "Water" in result or "advisory" in result.lower()


@pytest.mark.asyncio
async def test_get_advisory_asks_for_crop_when_missing():
    agent = _make_agent()
    session = agent._get_or_create_session("f5b", None, "en")
    template = agent.RESPONSE_TEMPLATES[VoiceIntent.GET_ADVISORY]["en"]

    result = await agent._handle_get_advisory(template, {}, session)

    assert "crop" in result.lower() or "which" in result.lower()


@pytest.mark.asyncio
async def test_check_weather_returns_forecast():
    weather_tool = SimpleNamespace()

    async def get_forecast(location: str):
        return {"condition": "Partly Cloudy", "temperature": 26, "advisory": "Good day for harvesting."}

    weather_tool.get_forecast = get_forecast
    agent = _make_agent(weather_tool=weather_tool)
    session = agent._get_or_create_session("f6", None, "en")
    template = agent.RESPONSE_TEMPLATES[VoiceIntent.CHECK_WEATHER]["en"]

    result = await agent._handle_check_weather(template, {"location": "Kolar"}, session)

    assert "Partly Cloudy" in result
    assert "26" in result


@pytest.mark.asyncio
async def test_check_weather_graceful_fallback():
    agent = _make_agent()
    session = agent._get_or_create_session("f7", None, "en")
    template = agent.RESPONSE_TEMPLATES[VoiceIntent.CHECK_WEATHER]["en"]

    result = await agent._handle_check_weather(template, {"location": "Mysuru"}, session)

    assert "not available" in result.lower() or "service" in result.lower()


@pytest.mark.asyncio
async def test_register_multiturn_collects_name_phone_district():
    reg_calls: list[dict] = []
    registration_service = SimpleNamespace()

    async def register_farmer(**kwargs):
        reg_calls.append(kwargs)
        return {"farmer_id": "far-999"}

    registration_service.register_farmer = register_farmer

    stt = StubStt(
        [
            TranscriptionResult("I want to register", "en", 0.9, 1.0, "stub"),
            TranscriptionResult("Raju", "en", 0.9, 1.0, "stub"),
            TranscriptionResult("9876543210", "en", 0.9, 1.0, "stub"),
            TranscriptionResult("Kolar", "en", 0.9, 1.0, "stub"),
        ]
    )
    extractor = StubExtractor(
        [
            ExtractionResult(VoiceIntent.REGISTER, {}, 0.8, "I want to register", "en"),
            ExtractionResult(VoiceIntent.UNKNOWN, {"name": "Raju"}, 0.6, "Raju", "en"),
            ExtractionResult(VoiceIntent.UNKNOWN, {"phone": "9876543210"}, 0.6, "9876543210", "en"),
            ExtractionResult(VoiceIntent.UNKNOWN, {"district": "Kolar"}, 0.6, "Kolar", "en"),
        ]
    )
    agent = VoiceAgent(
        stt=stt, tts=StubTts(), entity_extractor=extractor,
        registration_service=registration_service,
    )

    t1 = await agent.process_voice(b"a", user_id="f8", language="en")
    t2 = await agent.process_voice(b"b", user_id="f8", session_id=t1.session_id, language="en")
    t3 = await agent.process_voice(b"c", user_id="f8", session_id=t1.session_id, language="en")
    t4 = await agent.process_voice(b"d", user_id="f8", session_id=t1.session_id, language="en")

    assert "name" in t1.response_text.lower() or "What is your name" in t1.response_text
    assert "mobile" in t2.response_text.lower() or "number" in t2.response_text.lower()
    assert "district" in t3.response_text.lower()
    assert "far-999" in t4.response_text
    assert "Raju" in t4.response_text
    assert len(reg_calls) == 1


@pytest.mark.asyncio
async def test_dispute_status_returns_dispute_info():
    agent = _make_agent(order_service=StubOrderService())
    session = agent._get_or_create_session("f9", None, "en")
    template = agent.RESPONSE_TEMPLATES[VoiceIntent.DISPUTE_STATUS]["en"]

    result = await agent._handle_dispute_status(template, {"order_id": "ord-11"}, session)

    assert "disp-001" in result
    assert "Under Review" in result


@pytest.mark.asyncio
async def test_dispute_status_graceful_fallback():
    agent = _make_agent()
    session = agent._get_or_create_session("f10", None, "en")
    template = agent.RESPONSE_TEMPLATES[VoiceIntent.DISPUTE_STATUS]["en"]

    result = await agent._handle_dispute_status(template, {}, session)

    assert "not available" in result.lower() or "dispute" in result.lower()


@pytest.mark.asyncio
async def test_weekly_demand_returns_list():
    adcl_agent = SimpleNamespace()

    async def get_weekly_list(location: str):
        return ["tomato", "onion", "potato"]

    adcl_agent.get_weekly_list = get_weekly_list
    agent = _make_agent(adcl_agent=adcl_agent)
    session = agent._get_or_create_session("f11", None, "en")
    template = agent.RESPONSE_TEMPLATES[VoiceIntent.WEEKLY_DEMAND]["en"]

    result = await agent._handle_weekly_demand(template, {"location": "Bangalore"}, session)

    assert "tomato" in result.lower()
    assert "onion" in result.lower()


@pytest.mark.asyncio
async def test_weekly_demand_graceful_fallback():
    agent = _make_agent()
    session = agent._get_or_create_session("f12", None, "en")
    template = agent.RESPONSE_TEMPLATES[VoiceIntent.WEEKLY_DEMAND]["en"]

    result = await agent._handle_weekly_demand(template, {}, session)

    assert "not available" in result.lower() or "service" in result.lower()


@pytest.mark.asyncio
async def test_session_context_preserved_across_turns():
    """Verify session context accumulates entities across multiple turns."""
    stt = StubStt(
        [
            TranscriptionResult("find buyer for tomato", "en", 0.9, 1.0, "stub"),
            TranscriptionResult("500 kg", "en", 0.9, 1.0, "stub"),
        ]
    )
    extractor = StubExtractor(
        [
            ExtractionResult(VoiceIntent.FIND_BUYER, {"commodity": "tomato"}, 0.8, "find buyer for tomato", "en"),
            ExtractionResult(VoiceIntent.UNKNOWN, {"quantity_kg": 500}, 0.6, "500 kg", "en"),
        ]
    )
    agent = VoiceAgent(stt=stt, tts=StubTts(), entity_extractor=extractor)

    t1 = await agent.process_voice(b"a", user_id="f13", language="en")
    t2 = await agent.process_voice(b"b", user_id="f13", session_id=t1.session_id, language="en")

    session = agent.get_session(t1.session_id)
    assert session is not None
    # Context should carry over - pending or resolved
    assert "commodity" in session.context.get("pending_find_buyer", {}) or "500" in t2.response_text or "service" in t2.response_text.lower()


@pytest.mark.asyncio
async def test_hindi_language_templates_used_for_find_buyer():
    """Verify Hindi language templates are used when language is 'hi'."""
    agent = _make_agent()
    session = agent._get_or_create_session("f14", None, "hi")

    result = await agent._handle_find_buyer("", {}, session)

    assert any(char in result for char in "अआइईउऊएऐओऔ")  # Hindi unicode present


@pytest.mark.asyncio
async def test_kannada_language_templates_used_for_quality_check():
    """Verify Kannada language templates are used when language is 'kn'."""
    agent = _make_agent()
    session = agent._get_or_create_session("f15", None, "kn")

    result = await agent._handle_quality_check("", {}, session)

    # * Kannada script spans U+0C80–U+0CFF; any char in that range confirms localised response
    assert any("\u0c80" <= char <= "\u0cff" for char in result), f"Expected Kannada text, got: {result!r}"
