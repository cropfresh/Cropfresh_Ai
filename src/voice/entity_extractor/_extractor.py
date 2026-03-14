"""
VoiceEntityExtractor – main extraction class.

Depends only on the sibling data modules; has no large data literals of its own.
"""

import json
import re
from typing import Optional

from loguru import logger

from src.voice.entity_extractor._crops import COMMODITY_ALIASES, CROP_NAMES
from src.voice.entity_extractor._intents import ExtractionResult, VoiceIntent
from src.voice.entity_extractor._keywords import INTENT_KEYWORDS
from src.voice.entity_extractor._language import detect_language_from_text
from src.voice.entity_extractor._patterns import (
    LOCATION_PATTERNS,
    PRICE_PATTERNS,
    QUANTITY_PATTERNS,
    UNIT_MAP,
)


class VoiceEntityExtractor:
    """
    Extract intent and entities from transcribed farmer voice queries.

    Supports 10 Indian languages:
        hi (Hindi)  · kn (Kannada) · ta (Tamil) · te (Telugu)
        mr (Marathi)· bn (Bengali)  · gu (Gujarati) · pa (Punjabi)
        ml (Malayalam) · en (English)

    Example::

        extractor = VoiceEntityExtractor()
        result = await extractor.extract("मेरे पास 200 किलो टमाटर है", "hi")
        # result.intent  → CREATE_LISTING
        # result.entities → {"crop": "tomato", "quantity": 200, "unit": "kg"}
    """

    # ── Expose data dicts as class attributes (backwards-compat) ─────────────
    INTENT_KEYWORDS  = INTENT_KEYWORDS
    CROP_NAMES       = CROP_NAMES
    COMMODITY_ALIASES = COMMODITY_ALIASES
    QUANTITY_PATTERNS = QUANTITY_PATTERNS
    PRICE_PATTERNS   = PRICE_PATTERNS
    LOCATION_PATTERNS = LOCATION_PATTERNS
    UNIT_MAP         = UNIT_MAP

    # Unified crop lookup (CROP_NAMES ∪ COMMODITY_ALIASES)
    _ALL_CROPS: dict[str, str] = {**CROP_NAMES, **COMMODITY_ALIASES}

    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider
        logger.info("VoiceEntityExtractor initialised (10-language support)")

    # ── Public API ────────────────────────────────────────────────────────────

    async def extract(
        self,
        text: str,
        language: str,
        use_llm: bool = True,
        context_intent: str = "",
    ) -> ExtractionResult:
        """Extract intent and entities from text.

        Args:
            context_intent: Active pending intent from session (e.g. 'create_listing').
                           Used to bias extraction for short follow-up answers.
        """
        if not text:
            return ExtractionResult(
                intent=VoiceIntent.UNKNOWN, entities={},
                confidence=0.0, original_text=text, language=language,
            )

        text_lower = text.lower().strip()
        result = self._extract_with_patterns(text_lower, language, context_intent=context_intent)

        if result.confidence < 0.7 and use_llm and self.llm_provider:
            llm_result = await self._extract_with_llm(text, language)
            if llm_result.confidence > result.confidence:
                return llm_result

        return result

    @staticmethod
    def detect_language_from_text(text: str) -> str:
        """Detect language from unicode script ranges. Same as module-level function."""
        return detect_language_from_text(text)

    def get_supported_intents(self) -> list[str]:
        return [i.value for i in VoiceIntent if i != VoiceIntent.UNKNOWN]

    # ── Pattern-based extraction ──────────────────────────────────────────────

    def _extract_with_patterns(self, text: str, language: str, context_intent: str = "") -> ExtractionResult:
        intent, intent_confidence = self._detect_intent(text, language)

        # When a pending multi-turn flow is active and the detected intent
        # is low-confidence or ambiguous, bias towards the pending intent
        # so that short follow-up answers ("20 kgs", "20 rupees") are
        # interpreted in the right context.
        context_vi = None
        if context_intent:
            try:
                context_vi = VoiceIntent(context_intent)
            except ValueError:
                pass

        if context_vi and intent_confidence < 0.5:
            # Low-confidence detection → prefer the pending context intent
            intent = context_vi
            intent_confidence = 0.6  # moderate confidence from context

        entities: dict = {}
        if intent == VoiceIntent.CREATE_LISTING:
            entities = self._extract_listing_entities(text, language)
        elif intent == VoiceIntent.CHECK_PRICE:
            entities = self._extract_price_entities(text, language)
        elif intent == VoiceIntent.TRACK_ORDER:
            entities = self._extract_order_entities(text, language)
        elif intent == VoiceIntent.FIND_BUYER:
            entities = self._extract_find_buyer_entities(text, language)
        elif intent == VoiceIntent.CHECK_WEATHER:
            entities = self._extract_location_entities(text, language)
        elif intent == VoiceIntent.GET_ADVISORY:
            entities = self._extract_advisory_entities(text, language)
        elif intent == VoiceIntent.REGISTER:
            entities = self._extract_register_entities(text, language)
        elif intent == VoiceIntent.DISPUTE_STATUS:
            entities = self._extract_order_entities(text, language)
        elif intent == VoiceIntent.QUALITY_CHECK:
            entities = self._extract_quality_entities(text, language)
        elif intent == VoiceIntent.WEEKLY_DEMAND:
            entities = self._extract_location_entities(text, language)

        # If context intent is active, also try to extract entities for that
        # intent type to catch values the primary extraction might miss
        if context_vi and context_vi != intent:
            context_entities = {}
            if context_vi == VoiceIntent.CREATE_LISTING:
                context_entities = self._extract_listing_entities(text, language)
            elif context_vi == VoiceIntent.FIND_BUYER:
                context_entities = self._extract_find_buyer_entities(text, language)
            elif context_vi == VoiceIntent.REGISTER:
                context_entities = self._extract_register_entities(text, language)
            # Merge context entities (don't overwrite already-found ones)
            for k, v in context_entities.items():
                if k not in entities and v not in [None, ""]:
                    entities[k] = v

        confidence = min(intent_confidence + (0.1 if entities else 0), 0.95)
        return ExtractionResult(
            intent=intent, entities=entities,
            confidence=confidence, original_text=text, language=language,
        )

    def _detect_intent(self, text: str, language: str) -> tuple[VoiceIntent, float]:
        scores: dict[VoiceIntent, float] = {}
        for intent, lang_keywords in self.INTENT_KEYWORDS.items():
            keywords = lang_keywords.get(language, []) + lang_keywords.get("en", [])
            matches = sum(1 for kw in keywords if kw.lower() in text.lower())
            if matches > 0 and keywords:
                scores[intent] = matches / len(keywords)

        if not scores:
            return VoiceIntent.UNKNOWN, 0.0

        best = max(scores, key=scores.get)
        return best, min(scores[best] * 2, 0.9)

    # ── Entity helpers ────────────────────────────────────────────────────────

    def _lookup_crop(self, text: str) -> Optional[str]:
        """Search the unified crop dictionary against lowercased text."""
        text_l = text.lower()
        for name, english in self._ALL_CROPS.items():
            if name.lower() in text_l:
                return english
        return None

    def _extract_quantity(self, text: str, language: str) -> dict:
        pattern = self.QUANTITY_PATTERNS.get(language, self.QUANTITY_PATTERNS["en"])
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return {
                "quantity": float(m.group(1)),
                "unit": self.UNIT_MAP.get(m.group(2).lower(), m.group(2).lower()),
            }
        return {}

    def _extract_location(self, text: str, language: str) -> dict:
        pattern = self.LOCATION_PATTERNS.get(language, self.LOCATION_PATTERNS["en"])
        m = re.search(pattern, text, re.IGNORECASE)
        return {"location": m.group(1)} if m else {}

    def _extract_listing_entities(self, text: str, language: str) -> dict:
        entities: dict = {}
        crop = self._lookup_crop(text)
        if crop:
            entities["crop"] = crop
        entities.update(self._extract_quantity(text, language))
        price_pat = self.PRICE_PATTERNS.get(language, self.PRICE_PATTERNS["en"])
        pm = re.search(price_pat, text, re.IGNORECASE)
        if pm:
            entities["asking_price"] = float(pm.group(1))
        return entities

    def _extract_price_entities(self, text: str, language: str) -> dict:
        entities: dict = {}
        crop = self._lookup_crop(text)
        if crop:
            entities["crop"] = crop
        entities.update(self._extract_location(text, language))
        return entities

    def _extract_order_entities(self, text: str, language: str) -> dict:
        m = re.search(r"(?:order|ऑर्डर|ਆਰਡਰ|ऑर्डर)\s*#?\s*(\w+)", text, re.IGNORECASE)
        return {"order_id": m.group(1)} if m else {}

    def _extract_find_buyer_entities(self, text: str, language: str) -> dict:
        entities: dict = {}
        crop = self._lookup_crop(text)
        if crop:
            entities["commodity"] = crop
        qty = self._extract_quantity(text, language)
        if qty:
            entities["quantity_kg"] = qty.get("quantity")
            entities["unit"] = qty.get("unit")
        return entities

    def _extract_location_entities(self, text: str, language: str) -> dict:
        return self._extract_location(text, language)

    def _extract_advisory_entities(self, text: str, language: str) -> dict:
        entities: dict = {}
        crop = self._lookup_crop(text)
        if crop:
            entities["crop"] = crop
        entities.update(self._extract_location(text, language))
        return entities

    def _extract_register_entities(self, text: str, language: str) -> dict:
        entities: dict = {}
        pm = re.search(r"\b(\d{10})\b", text)
        if pm:
            entities["phone"] = pm.group(1)
        entities.update(self._extract_location(text, language))
        if "location" in entities:
            entities["district"] = entities.pop("location")
        return entities

    def _extract_quality_entities(self, text: str, language: str) -> dict:
        entities: dict = {}
        crop = self._lookup_crop(text)
        if crop:
            entities["commodity"] = crop
        lm = re.search(r"(?:listing|lst)[- ]?(\w+)", text, re.IGNORECASE)
        if lm:
            entities["listing_id"] = lm.group(1)
        return entities

    # ── LLM-based extraction ──────────────────────────────────────────────────

    async def _extract_with_llm(self, text: str, language: str) -> ExtractionResult:
        if not self.llm_provider:
            return ExtractionResult(
                intent=VoiceIntent.UNKNOWN, entities={},
                confidence=0.0, original_text=text, language=language,
            )

        intents_list = " | ".join(i.value for i in VoiceIntent)
        prompt = f"""Extract intent and entities from this farmer's voice query.

Query (language: {language}): "{text}"

Respond in JSON:
{{
    "intent": "{intents_list}",
    "entities": {{
        "crop": "English crop name",
        "quantity": number,
        "unit": "kg | quintal | ton",
        "location": "location name"
    }},
    "confidence": 0.0-1.0
}}

Include only entities clearly mentioned. Return only JSON."""

        try:
            from src.orchestrator.llm_provider import LLMMessage
            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.llm_provider.generate(messages, max_tokens=200)
            response_text = response.content if hasattr(response, 'content') else str(response)
            m = re.search(r'\{[\s\S]*\}', response_text)
            if m:
                data = json.loads(m.group())
                try:
                    intent = VoiceIntent(data.get("intent", "unknown"))
                except ValueError:
                    intent = VoiceIntent.UNKNOWN
                return ExtractionResult(
                    intent=intent,
                    entities=data.get("entities", {}),
                    confidence=data.get("confidence", 0.7),
                    original_text=text,
                    language=language,
                )
        except Exception as exc:
            logger.warning(f"LLM extraction failed: {exc}")

        return ExtractionResult(
            intent=VoiceIntent.UNKNOWN, entities={},
            confidence=0.0, original_text=text, language=language,
        )
