from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.rag.agentic.speculative import SpeculativeDraftEngine
from src.rag.graph_runtime.services import generate_answer
from src.rag.language_support import (
    build_generation_language_instruction,
    detect_response_language,
)


class StubLLM:
    def __init__(self, content: str = "ಉತ್ತರ ಸಿದ್ಧವಾಗಿದೆ."):
        self.content = content
        self.messages = None

    async def generate(self, messages, temperature: float = 0.0, max_tokens: int = 0):
        del temperature, max_tokens
        self.messages = messages
        return SimpleNamespace(content=self.content)


def test_detect_response_language_for_kannada_script():
    assert detect_response_language("ಟೊಮೆಟೊ ಬೆಲೆ ಎಷ್ಟು?") == "kn"


def test_detect_response_language_for_transliterated_kannada():
    assert detect_response_language("Tomato bele yaavaga Kolar mandi alli?") == "kn"


def test_build_generation_language_instruction_mentions_kannada_market_terms():
    instruction = build_generation_language_instruction(
        "ಟೊಮೆಟೊ ಬೆಲೆ ಎಷ್ಟು?",
        route="live_price_api",
    )
    assert "Kannada" in instruction
    assert "ಬೆಲೆ" in instruction
    assert "ಮಾದರಿ ದರ" in instruction


@pytest.mark.asyncio
async def test_generate_answer_localizes_no_information_for_kannada():
    answer, model = await generate_answer("ಟೊಮೆಟೊ ಬೆಲೆ ಎಷ್ಟು?", [], llm=None)
    assert model == "none"
    assert any("\u0c80" <= char <= "\u0cff" for char in answer)


@pytest.mark.asyncio
async def test_generate_answer_localizes_extractive_fallback_prefix_for_kannada():
    docs = [SimpleNamespace(text="Tomato modal price is Rs.2500 per quintal.", id="d1")]
    answer, model = await generate_answer("Tomato bele yeshtu?", docs, llm=None)
    assert model == "extractive_fallback"
    assert answer.startswith("ಪಡೆಯಲಾದ ದಾಖಲೆಗಳ ಆಧಾರದಲ್ಲಿ:")


@pytest.mark.asyncio
async def test_speculative_draft_engine_injects_kannada_prompt_guidance():
    llm = StubLLM()
    engine = SpeculativeDraftEngine(drafter_llm=llm, verifier_llm=llm, n_subsets=1)
    docs = [SimpleNamespace(text="Tomato modal price is Rs.2500 per quintal.", id="d1")]

    answer, best_idx = await engine.generate_and_select(
        documents=docs,
        query="Tomato bele yaavaga Kolar mandi alli?",
        response_language="kn",
        route="live_price_api",
    )

    assert best_idx == 0
    assert answer == "ಉತ್ತರ ಸಿದ್ಧವಾಗಿದೆ."
    prompt = llm.messages[0].content
    assert "Respond fully in natural, respectful Kannada" in prompt
    assert "ಬೆಲೆ" in prompt
