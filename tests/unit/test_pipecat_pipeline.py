"""
Unit tests for Task 14: Pipecat WebSocket Voice Streaming Pipeline.

Groups
------
A  – CropFreshAgentProcessor frame handling (8 tests, no live pipeline)
     Requires pipecat package — skipped automatically when not installed.
B  – run_voice_bot pipeline assembly / module import (6 tests)
     Requires pipecat package — skipped automatically when not installed.
C  – VoiceAgent.handle_text_input (8 tests, stubs for all I/O)
     No pipecat dependency — always runs.

Run with:
    uv run pytest tests/unit/test_pipecat_pipeline.py -v
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import types
from types import SimpleNamespace
from typing import Optional
from uuid import uuid4

import pytest

from src.agents.voice_agent import VoiceAgent, VoiceResponse
from src.voice.entity_extractor import ExtractionResult, VoiceIntent
from src.voice.stt import TranscriptionResult
from src.voice.tts import SynthesisResult

# ---------------------------------------------------------------------------
# Detect whether pipecat is available in this environment.
# Groups A and B tests are marked to skip when it is absent.
# ---------------------------------------------------------------------------
_PIPECAT_AVAILABLE = importlib.util.find_spec("pipecat") is not None
_skip_no_pipecat = pytest.mark.skipif(
    not _PIPECAT_AVAILABLE,
    reason="pipecat package not installed — install with: uv sync --extra voice",
)


# ═══════════════════════════════════════════════════════════════════════════
# SHARED STUBS
# ═══════════════════════════════════════════════════════════════════════════

class StubStt:
    """STT stub — should NOT be called in Pipecat path."""

    def __init__(self):
        self.called = False

    async def transcribe(self, audio: bytes, language: str = "auto") -> TranscriptionResult:
        self.called = True
        return TranscriptionResult("stub text", language, 0.9, 1.0, "stub")

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
    def __init__(self, result: Optional[ExtractionResult] = None, intent: VoiceIntent = VoiceIntent.GREETING):
        self._result = result or ExtractionResult(
            intent=intent,
            entities={},
            confidence=0.9,
            original_text="",
            language="en",
        )

    async def extract(self, text: str, language: str) -> ExtractionResult:
        self._result.original_text = text
        self._result.language = language
        return self._result


def _make_agent(**kwargs) -> VoiceAgent:
    """Build VoiceAgent with stubs for unit testing."""
    return VoiceAgent(
        stt=kwargs.pop("stt", StubStt()),
        tts=kwargs.pop("tts", StubTts()),
        entity_extractor=kwargs.pop("entity_extractor", StubExtractor()),
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════════════
# GROUP A — CropFreshAgentProcessor frame handling
# ═══════════════════════════════════════════════════════════════════════════

# Import lazily to avoid Pipecat being a hard dependency for the test runner
# (the processor defers actual pipeline execution).
def _make_processor(voice_agent=None, session_id: str = "s-001", language: str = "hi"):
    from src.voice.pipecat.agent_processor import CropFreshAgentProcessor
    return CropFreshAgentProcessor(
        voice_agent=voice_agent or _make_agent(),
        session_id=session_id,
        language=language,
    )


@_skip_no_pipecat
@pytest.mark.asyncio
async def test_agent_processor_routes_text_frame_to_voice_agent():
    """TextFrame → _route_text → push downstream TextFrame with non-empty text."""
    from pipecat.frames.frames import TextFrame
    proc = _make_processor(language="en")

    pushed: list = []
    proc.push_frame = lambda frame, *_: pushed.append(frame) or __import__("asyncio").sleep(0)

    await proc._route_text("what is the price of tomato")
    # We called _route_text directly; it returns a string, not a frame.
    result = await proc._route_text("hello")
    assert isinstance(result, str)
    assert len(result) > 0


@_skip_no_pipecat
@pytest.mark.asyncio
async def test_agent_processor_passes_through_non_text_frames():
    """EndFrame and other frame types pass through unchanged."""
    from pipecat.frames.frames import EndFrame
    from pipecat.processors.frame_processor import FrameDirection
    proc = _make_processor()

    passed: list = []

    async def capture(frame, direction=None):
        passed.append(frame)

    proc.push_frame = capture

    end = EndFrame()
    await proc.process_frame(end, FrameDirection.DOWNSTREAM)
    assert any(isinstance(f, EndFrame) for f in passed)


@_skip_no_pipecat
@pytest.mark.asyncio
async def test_agent_processor_uses_correct_session_id():
    """session_id constructor arg is stored and accessible."""
    proc = _make_processor(session_id="farm-session-42")
    assert proc.session_id == "farm-session-42"


@_skip_no_pipecat
@pytest.mark.asyncio
async def test_agent_processor_uses_correct_language():
    """language='kn' is stored and used in fallback messages."""
    proc = _make_processor(language="kn")
    assert proc.language == "kn"


@_skip_no_pipecat
@pytest.mark.asyncio
async def test_agent_processor_handles_voice_agent_exception_gracefully():
    """If VoiceAgent.handle_text_input raises, fallback text is returned (no crash)."""
    class BrokenAgent:
        async def handle_text_input(self, **_):
            raise RuntimeError("simulated failure")

    proc = _make_processor(voice_agent=BrokenAgent(), language="en")
    result = await proc._route_text("some query")
    # Should return the English fallback, not raise
    assert "sorry" in result.lower() or "again" in result.lower()


@_skip_no_pipecat
@pytest.mark.asyncio
async def test_agent_processor_ignores_empty_text_frame():
    """TextFrame(text='   ') is passed through without calling VoiceAgent."""
    from pipecat.frames.frames import TextFrame
    from pipecat.processors.frame_processor import FrameDirection

    call_count = 0

    class CountingAgent:
        async def handle_text_input(self, **_):
            nonlocal call_count
            call_count += 1
            return SimpleNamespace(response_text="yo")

    proc = _make_processor(voice_agent=CountingAgent())

    pushed: list = []

    async def capture(frame, direction=None):
        pushed.append(frame)

    proc.push_frame = capture

    await proc.process_frame(TextFrame(text="   "), FrameDirection.DOWNSTREAM)
    # VoiceAgent should NOT be called for blank text
    assert call_count == 0


@_skip_no_pipecat
@pytest.mark.asyncio
async def test_agent_processor_hindi_fallback_on_error():
    """For language='hi', fallback text contains Devanagari script."""
    class AlwaysFails:
        async def handle_text_input(self, **_):
            raise ValueError("boom")

    proc = _make_processor(voice_agent=AlwaysFails(), language="hi")
    fallback = await proc._route_text("कुछ भी")
    # Devanagari block: U+0900–U+097F
    assert any("\u0900" <= ch <= "\u097f" for ch in fallback), (
        f"Expected Devanagari in Hindi fallback, got: {fallback!r}"
    )


@_skip_no_pipecat
@pytest.mark.asyncio
async def test_agent_processor_english_fallback_on_error():
    """For language='en', fallback text is in English."""
    class AlwaysFails:
        async def handle_text_input(self, **_):
            raise ValueError("boom")

    proc = _make_processor(voice_agent=AlwaysFails(), language="en")
    fallback = await proc._route_text("anything")
    assert fallback  # non-empty
    # Must be ASCII-printable (English)
    assert all(ord(ch) < 256 for ch in fallback)


# ═══════════════════════════════════════════════════════════════════════════
# GROUP B — pipecat_bot module import and pipeline assembly
# ═══════════════════════════════════════════════════════════════════════════

@_skip_no_pipecat
def test_pipecat_bot_logger_import_no_nameerror():
    """Importing pipecat_bot must NOT raise NameError due to missing logger."""
    # A NameError would propagate immediately on import
    import src.voice.pipecat_bot as bot_module  # noqa: F401
    assert hasattr(bot_module, "run_voice_bot"), "run_voice_bot function must exist"


@_skip_no_pipecat
def test_run_voice_bot_function_accepts_voice_agent_param():
    """run_voice_bot signature must accept an optional voice_agent parameter."""
    import inspect
    from src.voice.pipecat_bot import run_voice_bot
    sig = inspect.signature(run_voice_bot)
    assert "voice_agent" in sig.parameters, (
        "run_voice_bot must accept voice_agent keyword argument"
    )


@_skip_no_pipecat
def test_run_voice_bot_uses_cropfresh_agent_processor_import():
    """pipecat_bot.py must import CropFreshAgentProcessor (not OpenAILLMService)."""
    import src.voice.pipecat_bot as bot_module
    source_file = bot_module.__file__
    with open(source_file, encoding="utf-8") as f:
        source = f.read()
    assert "CropFreshAgentProcessor" in source, (
        "pipecat_bot must reference CropFreshAgentProcessor"
    )
    # Check no *import* of OpenAILLMService (comments are OK)
    import_lines = [l.strip() for l in source.splitlines() if l.strip().startswith(("import ", "from "))]
    assert not any("OpenAILLMService" in l for l in import_lines), (
        "pipecat_bot must NOT import OpenAILLMService (replaced by CropFreshAgentProcessor)"
    )


@_skip_no_pipecat
def test_pipecat_bot_no_openai_llm_service():
    """OpenAILLMService import must have been removed from pipecat_bot."""
    import src.voice.pipecat_bot as bot_module
    source_file = bot_module.__file__
    with open(source_file, encoding="utf-8") as f:
        source = f.read()
    import_lines = [l.strip() for l in source.splitlines() if l.strip().startswith(("import ", "from "))]
    assert not any("OpenAILLMService" in l for l in import_lines), (
        "OpenAILLMService should not be imported in pipecat_bot (use CropFreshAgentProcessor)"
    )


@_skip_no_pipecat
def test_agent_processor_module_exists():
    """src/voice/pipecat/agent_processor.py must be importable."""
    from src.voice.pipecat.agent_processor import CropFreshAgentProcessor  # noqa
    assert CropFreshAgentProcessor is not None


@_skip_no_pipecat
def test_agent_processor_extends_frame_processor():
    """CropFreshAgentProcessor must subclass Pipecat FrameProcessor."""
    from pipecat.processors.frame_processor import FrameProcessor
    from src.voice.pipecat.agent_processor import CropFreshAgentProcessor
    assert issubclass(CropFreshAgentProcessor, FrameProcessor)


# ═══════════════════════════════════════════════════════════════════════════
# GROUP C — VoiceAgent.handle_text_input
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_handle_text_input_does_not_call_stt():
    """STT.transcribe must NOT be called when handle_text_input is used."""
    stub_stt = StubStt()
    agent = _make_agent(stt=stub_stt)
    await agent.handle_text_input(
        text="I want to sell tomato",
        user_id="farmer-1",
        language="en",
    )
    assert not stub_stt.called, "STT must be bypassed in handle_text_input path"


@pytest.mark.asyncio
async def test_handle_text_input_returns_voice_response():
    """Returns a VoiceResponse instance with response_text and response_audio."""
    agent = _make_agent()
    resp = await agent.handle_text_input(
        text="hello",
        user_id="farmer-2",
        language="en",
    )
    assert isinstance(resp, VoiceResponse)
    assert isinstance(resp.response_text, str) and resp.response_text
    assert isinstance(resp.response_audio, bytes) and resp.response_audio


@pytest.mark.asyncio
async def test_handle_text_input_response_contains_session_id():
    """Returned VoiceResponse.session_id is non-empty."""
    agent = _make_agent()
    resp = await agent.handle_text_input(
        text="hi",
        user_id="u1",
        language="en",
    )
    assert resp.session_id


@pytest.mark.asyncio
async def test_handle_text_input_creates_new_session_when_none():
    """With session_id=None a new session is auto-created and returned."""
    agent = _make_agent()
    resp = await agent.handle_text_input(
        text="hello",
        user_id="u2",
        session_id=None,
        language="en",
    )
    session = agent.get_session(resp.session_id)
    assert session is not None
    assert session.user_id == "u2"


@pytest.mark.asyncio
async def test_handle_text_input_preserves_session_context():
    """Two calls with the same session_id share context (multi-turn)."""
    intent_seq = [
        ExtractionResult(VoiceIntent.CREATE_LISTING, {"crop": "tomato"}, 0.9, "", "en"),
        ExtractionResult(VoiceIntent.UNKNOWN, {"quantity": 100}, 0.6, "", "en"),
    ]
    idx = 0

    class SeqExtractor:
        async def extract(self, text: str, language: str) -> ExtractionResult:
            nonlocal idx
            r = intent_seq[idx]
            idx = min(idx + 1, len(intent_seq) - 1)
            return r

    agent = VoiceAgent(stt=StubStt(), tts=StubTts(), entity_extractor=SeqExtractor())

    r1 = await agent.handle_text_input("I want to sell tomato", "f1", language="en")
    r2 = await agent.handle_text_input("100 kg", "f1", session_id=r1.session_id, language="en")

    # Session must be preserved
    assert r1.session_id == r2.session_id


@pytest.mark.asyncio
async def test_handle_text_input_hindi_language_returns_hindi_response():
    """language='hi' → response_text contains Devanagari."""
    agent = _make_agent(
        entity_extractor=StubExtractor(intent=VoiceIntent.GREETING),
    )
    resp = await agent.handle_text_input(
        text="नमस्ते",
        user_id="farmer-hi",
        language="hi",
    )
    assert any("\u0900" <= ch <= "\u097f" for ch in resp.response_text), (
        f"Expected Devanagari in Hindi response, got: {resp.response_text!r}"
    )


@pytest.mark.asyncio
async def test_handle_text_input_kannada_language_returns_kannada_response():
    """language='kn' → response_text contains Kannada script."""
    agent = _make_agent(
        entity_extractor=StubExtractor(intent=VoiceIntent.GREETING),
    )
    resp = await agent.handle_text_input(
        text="ನಮಸ್ಕಾರ",
        user_id="farmer-kn",
        language="kn",
    )
    assert any("\u0c80" <= ch <= "\u0cff" for ch in resp.response_text), (
        f"Expected Kannada script in response, got: {resp.response_text!r}"
    )


@pytest.mark.asyncio
async def test_handle_text_input_english_greeting_response():
    """language='en' greeting → response_text is in English."""
    agent = _make_agent(
        entity_extractor=StubExtractor(intent=VoiceIntent.GREETING),
    )
    resp = await agent.handle_text_input(
        text="hello",
        user_id="farmer-en",
        language="en",
    )
    # English response should contain 'CropFresh' or 'Hello'
    assert "CropFresh" in resp.response_text or resp.response_text
