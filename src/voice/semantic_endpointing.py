"""Semantic endpointing helpers layered on top of acoustic VAD."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Protocol

from loguru import logger

from src.orchestrator.llm_provider import LLMMessage

FILLER_PATTERNS = {
    "default": {
        "uh",
        "uhh",
        "umm",
        "hmm",
        "wait",
        "one second",
        "hold on",
    },
    "hi": {"एक मिनट", "रुको", "ठहरो", "सोचने दो"},
    "kn": {"ಒಂದು ನಿಮಿಷ", "ಸ್ವಲ್ಪ", "ಅಯ್ಯೋ", "ಹಂ"},
    "te": {"ఒక్క నిమిషం", "ఆగండి", "హమ్"},
    "ta": {"ஒரு நிமிடம்", "சற்று", "ஹும்"},
}

INCOMPLETE_ENDINGS = {
    "default": {"and", "but", "or", "because", "if", "then"},
    "hi": {"और", "लेकिन", "क्योंकि", "अगर"},
    "kn": {"ಮತ್ತು", "ಆದರೆ", "ಯಾಕೆಂದರೆ", "ಅದ್ರೆ"},
    "te": {"మరియు", "కాని", "ఎందుకంటే", "అయితే"},
    "ta": {"மற்றும்", "ஆனா", "ஏனெனில்", "என்றால்"},
}

SENTENCE_END_RE = re.compile(r"[.!?।]$")


class SupportsGenerate(Protocol):
    """Small protocol for the existing provider-layer generate call."""

    async def generate(self, messages: list[LLMMessage], **kwargs): ...


@dataclass(slots=True)
class SemanticEndpointDecision:
    """Decision produced by the semantic endpointing layer."""

    transcript: str
    detected_language: str
    should_flush: bool
    reason: str
    semantic_hold_ms: int = 0
    used_llm: bool = False
    timed_out: bool = False


def _normalized_tokens(text: str) -> list[str]:
    return [token for token in re.split(r"\s+", text.lower().strip()) if token]


def is_likely_incomplete(text: str, language: str) -> bool:
    """Fast heuristic check for fillers, hesitation phrases, and clipped endings."""
    normalized = text.strip().lower()
    if not normalized:
        return False

    fillers = FILLER_PATTERNS["default"] | FILLER_PATTERNS.get(language, set())
    endings = INCOMPLETE_ENDINGS["default"] | INCOMPLETE_ENDINGS.get(language, set())
    tokens = _normalized_tokens(normalized)
    last_token = tokens[-1] if tokens else ""

    if normalized in fillers:
        return True
    if any(normalized.endswith(phrase) for phrase in fillers):
        return True
    if last_token in endings:
        return True
    if not SENTENCE_END_RE.search(normalized) and len(tokens) <= 2 and last_token in fillers:
        return True
    return False


async def evaluate_semantic_flush(
    *,
    transcript: str,
    language: str,
    llm_provider: SupportsGenerate | None,
    enabled: bool,
    timeout_ms: int,
    max_hold_ms: int,
) -> SemanticEndpointDecision:
    """Decide whether an acoustically-complete segment should flush downstream."""
    cleaned = transcript.strip()
    if not enabled or not cleaned:
        return SemanticEndpointDecision(
            transcript=cleaned,
            detected_language=language,
            should_flush=True,
            reason="disabled_or_empty",
        )

    if is_likely_incomplete(cleaned, language):
        return SemanticEndpointDecision(
            transcript=cleaned,
            detected_language=language,
            should_flush=False,
            reason="heuristic_hold",
            semantic_hold_ms=min(max_hold_ms, timeout_ms * 2),
        )

    if llm_provider is None:
        return SemanticEndpointDecision(
            transcript=cleaned,
            detected_language=language,
            should_flush=True,
            reason="heuristic_flush",
        )

    prompt = (
        "You are deciding whether a speech segment is complete enough to send to an assistant.\n"
        "Reply with exactly one word: COMPLETE or HOLD.\n"
        f"Language: {language}\n"
        f"Transcript: {cleaned}"
    )

    try:
        response = await asyncio.wait_for(
            llm_provider.generate(
                [LLMMessage(role="user", content=prompt)],
                temperature=0.0,
                max_tokens=2,
            ),
            timeout=timeout_ms / 1000,
        )
    except asyncio.TimeoutError:
        return SemanticEndpointDecision(
            transcript=cleaned,
            detected_language=language,
            should_flush=True,
            reason="llm_timeout_flush",
            timed_out=True,
        )
    except Exception as exc:  # noqa: BLE001 - fallback to acoustic flush on provider failure
        logger.warning("Semantic endpointing LLM check failed: {}", exc)
        return SemanticEndpointDecision(
            transcript=cleaned,
            detected_language=language,
            should_flush=True,
            reason="llm_error_flush",
        )

    verdict = response.content.strip().upper()
    if verdict.startswith("HOLD"):
        return SemanticEndpointDecision(
            transcript=cleaned,
            detected_language=language,
            should_flush=False,
            reason="llm_hold",
            semantic_hold_ms=min(max_hold_ms, timeout_ms * 2),
            used_llm=True,
        )

    return SemanticEndpointDecision(
        transcript=cleaned,
        detected_language=language,
        should_flush=True,
        reason="llm_flush",
        used_llm=True,
    )
