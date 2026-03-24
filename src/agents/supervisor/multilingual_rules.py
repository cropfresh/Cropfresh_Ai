"""Shared multilingual intent routing for supervisor rule fallbacks."""

from __future__ import annotations

from src.agents.supervisor.models import RoutingDecision
from src.shared.language import detect_response_language
from src.voice.entity_extractor._intents import VoiceIntent
from src.voice.entity_extractor._keywords import INTENT_KEYWORDS

INTENT_AGENT_MAP = {
    VoiceIntent.CREATE_LISTING: "crop_listing_agent",
    VoiceIntent.MY_LISTINGS: "crop_listing_agent",
    VoiceIntent.CHECK_PRICE: "commerce_agent",
    VoiceIntent.TRACK_ORDER: "platform_agent",
    VoiceIntent.FIND_BUYER: "buyer_matching_agent",
    VoiceIntent.CHECK_WEATHER: "agronomy_agent",
    VoiceIntent.GET_ADVISORY: "agronomy_agent",
    VoiceIntent.REGISTER: "platform_agent",
    VoiceIntent.DISPUTE_STATUS: "platform_agent",
    VoiceIntent.QUALITY_CHECK: "quality_assessment_agent",
    VoiceIntent.WEEKLY_DEMAND: "adcl_agent",
    VoiceIntent.HELP: "general_agent",
    VoiceIntent.GREETING: "general_agent",
}


def route_multilingual_intent(query: str) -> RoutingDecision | None:
    """Reuse voice intent keywords for script-aware supervisor fallback routing."""
    query_lower = query.lower()
    language = detect_response_language(query)
    if language == "en":
        return None

    best_intent = None
    best_score = 0.0

    for intent, keywords_by_language in INTENT_KEYWORDS.items():
        agent_name = INTENT_AGENT_MAP.get(intent)
        if not agent_name:
            continue

        keywords = list(keywords_by_language.get(language, []))
        if language != "en":
            keywords.extend(keywords_by_language.get("en", []))

        matches = [keyword for keyword in keywords if keyword.lower() in query_lower]
        if not _is_viable_match(matches):
            continue

        score = len(matches) / max(len(keywords), 1)
        if score > best_score:
            best_intent = intent
            best_score = score

    if not best_intent:
        return None

    return RoutingDecision(
        agent_name=INTENT_AGENT_MAP[best_intent],
        confidence=min(0.62 + best_score, 0.86),
        reasoning=f"Rule-based: multilingual {best_intent.value}",
    )


def _is_viable_match(matches: list[str]) -> bool:
    """Ignore overly weak single-token hits like only 'kg'."""
    if not matches:
        return False
    if len(matches) >= 2:
        return True

    match = matches[0].strip()
    if len(match) >= 4 or " " in match:
        return True
    return any(ord(char) > 127 for char in match)
