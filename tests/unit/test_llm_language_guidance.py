"""Tests for automatic language guidance inside the LLM mixin."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agents.base.llm import LLMMixin
from src.agents.base.models import AgentConfig


class StubLLM:
    """Capture generated messages so tests can inspect injected prompts."""

    def __init__(self):
        self.messages = None

    async def generate(self, messages, temperature: float = 0.0, max_tokens: int = 0):
        del temperature, max_tokens
        self.messages = messages
        return SimpleNamespace(content="ok")


class DummyAgent(LLMMixin):
    """Minimal agent shell for exercising the mixin in isolation."""

    def __init__(self, name: str):
        self.config = AgentConfig(
            name=name,
            description="dummy",
            temperature=0.3,
            max_tokens=128,
        )
        self.llm = StubLLM()

    @property
    def name(self) -> str:
        return self.config.name


@pytest.mark.asyncio
async def test_generate_with_llm_injects_kannada_language_guidance():
    agent = DummyAgent("agronomy_agent")

    await agent.generate_with_llm(
        [
            {"role": "system", "content": "Base prompt"},
            {"role": "user", "content": "ಟೊಮೆಟೊ ಬೆಲೆ ಎಷ್ಟು?"},
        ]
    )

    system_messages = [
        message.content for message in agent.llm.messages if message.role == "system"
    ]
    assert any(
        "Respond fully in natural, respectful Kannada" in message for message in system_messages
    )


@pytest.mark.asyncio
async def test_generate_with_llm_skips_language_guidance_for_supervisor():
    agent = DummyAgent("supervisor")

    await agent.generate_with_llm(
        [
            {"role": "system", "content": "Routing prompt"},
            {"role": "user", "content": "ಟೊಮೆಟೊ ಬೆಲೆ ಎಷ್ಟು?"},
        ]
    )

    system_messages = [
        message.content for message in agent.llm.messages if message.role == "system"
    ]
    assert not any(
        "Respond fully in natural, respectful Kannada" in message for message in system_messages
    )


@pytest.mark.asyncio
async def test_generate_with_llm_injects_kannada_domain_context_for_custom_agent():
    agent = DummyAgent("buyer_matching")

    await agent.generate_with_llm(
        [
            {"role": "system", "content": "Base prompt"},
            {"role": "user", "content": "ಟೊಮೆಟೊಗೆ ಒಳ್ಳೆಯ ಖರೀದಿದಾರರನ್ನು ಹುಡುಕಿ"},
        ]
    )

    system_messages = [
        message.content for message in agent.llm.messages if message.role == "system"
    ]
    assert any("Buyer Matching Guidance" in message for message in system_messages)


@pytest.mark.asyncio
async def test_generate_with_llm_respects_kannada_profile_preference():
    agent = DummyAgent("crop_listing")

    await agent.generate_with_llm(
        [
            {"role": "system", "content": "Base prompt"},
            {"role": "user", "content": "Please help me create a listing"},
        ],
        context={"user_profile": {"language": "kn", "language_pref": "kn"}},
    )

    system_messages = [
        message.content for message in agent.llm.messages if message.role == "system"
    ]
    assert any("Listing & Selling Flow" in message for message in system_messages)


@pytest.mark.asyncio
async def test_generate_with_llm_injects_advanced_kannada_runtime_context():
    agent = DummyAgent("buyer_matching")

    await agent.generate_with_llm(
        [
            {"role": "system", "content": "Base prompt"},
            {"role": "user", "content": "Please help me find a buyer"},
        ],
        context={
            "user_profile": {"language": "kn", "district": "Kalaburagi"},
            "dialect_lexicon": [
                {
                    "dialect_tag": "HYDERABAD_KARNATAKA",
                    "slang": "scene",
                    "normalized_kannada": "paristhiti",
                    "english_gloss": "situation",
                    "example_user_sentence": "market scene yenide",
                    "example_ai_reply": "market paristhiti hegide anta keluttiddira",
                }
            ],
            "kannada_context_info": [
                {
                    "type": "buyer_matching",
                    "crop": "tomato",
                    "region": "Kalaburagi",
                    "details": "Bulk buyers ask for early morning arrival windows.",
                }
            ],
        },
    )

    system_messages = [
        message.content for message in agent.llm.messages if message.role == "system"
    ]
    assert any(
        "Likely dialect bucket: HYDERABAD_KARNATAKA" in message for message in system_messages
    )
    assert any("[DIALECT_LEXICON: HYDERABAD_KARNATAKA]" in message for message in system_messages)
    assert any("[CONTEXT_KANNADA_INFO]" in message for message in system_messages)
