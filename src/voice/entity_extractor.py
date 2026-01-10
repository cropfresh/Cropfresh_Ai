"""
Entity Extraction Module for CropFresh Voice Agent

Extracts intent and entities from transcribed voice input.
Uses LLM for complex extraction with pattern-based fallback.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from loguru import logger


class VoiceIntent(Enum):
    """Supported voice intents"""
    CREATE_LISTING = "create_listing"
    CHECK_PRICE = "check_price"
    TRACK_ORDER = "track_order"
    MY_LISTINGS = "my_listings"
    HELP = "help"
    GREETING = "greeting"
    UNKNOWN = "unknown"


@dataclass
class ExtractionResult:
    """Result from entity extraction"""
    intent: VoiceIntent
    entities: dict[str, Any]
    confidence: float
    original_text: str
    language: str
    
    @property
    def is_actionable(self) -> bool:
        return self.intent != VoiceIntent.UNKNOWN and self.confidence > 0.5


class VoiceEntityExtractor:
    """
    Extract intent and entities from transcribed voice.
    
    Supports:
    - Create listing: "मेरे पास 200 किलो टमाटर है"
    - Check price: "कोलार में टमाटर का भाव क्या है"
    - Track order: "मेरा ऑर्डर कहाँ है"
    - Greetings: "नमस्ते"
    
    Usage:
        extractor = VoiceEntityExtractor(llm_provider)
        result = await extractor.extract("मेरे पास 200 किलो टमाटर है", "hi")
        # result.intent = CREATE_LISTING
        # result.entities = {"crop": "tomato", "quantity": 200, "unit": "kg"}
    """
    
    # Intent keywords by language
    INTENT_KEYWORDS = {
        VoiceIntent.CREATE_LISTING: {
            "hi": ["बेचना", "बेच", "है मेरे पास", "मेरे पास", "किलो", "क्विंटल", "टन"],
            "kn": ["ಮಾರಾಟ", "ನನ್ನ ಬಳಿ", "ಕೆಜಿ", "ಕ್ವಿಂಟಲ್"],
            "en": ["sell", "i have", "kg", "quintal", "ton"],
        },
        VoiceIntent.CHECK_PRICE: {
            "hi": ["भाव", "रेट", "दाम", "कीमत", "मूल्य", "क्या है"],
            "kn": ["ಬೆಲೆ", "ದರ", "ಎಷ್ಟು"],
            "en": ["price", "rate", "cost", "how much"],
        },
        VoiceIntent.TRACK_ORDER: {
            "hi": ["ऑर्डर", "आर्डर", "कहाँ है", "स्थिति", "स्टेटस"],
            "kn": ["ಆರ್ಡರ್", "ಎಲ್ಲಿದೆ", "ಸ್ಥಿತಿ"],
            "en": ["order", "where", "status", "track"],
        },
        VoiceIntent.MY_LISTINGS: {
            "hi": ["मेरी लिस्टिंग", "मेरा सामान", "मेरी सब्जी"],
            "kn": ["ನನ್ನ ಪಟ್ಟಿ", "ನನ್ನ ಸರಕು"],
            "en": ["my listing", "my produce", "my vegetables"],
        },
        VoiceIntent.HELP: {
            "hi": ["मदद", "सहायता", "हेल्प", "कैसे करें"],
            "kn": ["ಸಹಾಯ", "ಹೆಲ್ಪ್"],
            "en": ["help", "how to", "guide"],
        },
        VoiceIntent.GREETING: {
            "hi": ["नमस्ते", "नमस्कार", "हेलो", "राम राम"],
            "kn": ["ನಮಸ್ಕಾರ", "ಹಲೋ"],
            "en": ["hello", "hi", "hey", "good morning"],
        },
    }
    
    # Crop name mapping (multilingual)
    CROP_NAMES = {
        # Hindi -> English
        "टमाटर": "tomato",
        "आलू": "potato",
        "प्याज": "onion",
        "गोभी": "cauliflower",
        "बंद गोभी": "cabbage",
        "बैंगन": "brinjal",
        "भिंडी": "okra",
        "मिर्च": "chili",
        "शिमला मिर्च": "capsicum",
        "गाजर": "carrot",
        "मूली": "radish",
        "पालक": "spinach",
        "धनिया": "coriander",
        "फूलगोभी": "cauliflower",
        "मटर": "peas",
        "लौकी": "bottle_gourd",
        "करेला": "bitter_gourd",
        "खीरा": "cucumber",
        "कद्दू": "pumpkin",
        "सेब": "apple",
        "केला": "banana",
        "आम": "mango",
        "अंगूर": "grapes",
        "संतरा": "orange",
        "अनार": "pomegranate",
        "पपीता": "papaya",
        "अमरूद": "guava",
        "चावल": "rice",
        "गेहूं": "wheat",
        # Kannada -> English
        "ಟೊಮೆಟೊ": "tomato",
        "ಆಲೂಗಡ್ಡೆ": "potato",
        "ಈರುಳ್ಳಿ": "onion",
        "ಹೂಕೋಸು": "cauliflower",
        # English (normalize)
        "tomato": "tomato",
        "tomatoes": "tomato",
        "potato": "potato",
        "potatoes": "potato",
        "onion": "onion",
        "onions": "onion",
    }
    
    # Location patterns
    LOCATION_PATTERNS = {
        "hi": r"(?:में|से|का|की|के)\s+(\w+)",
        "kn": r"(?:ಯಲ್ಲಿ|ಇಂದ)\s+(\w+)",
        "en": r"(?:in|at|from)\s+(\w+)",
    }
    
    # Quantity patterns
    QUANTITY_PATTERNS = {
        "hi": r"(\d+(?:\.\d+)?)\s*(किलो|क्विंटल|टन|kg|quintal)",
        "kn": r"(\d+(?:\.\d+)?)\s*(ಕೆಜಿ|ಕ್ವಿಂಟಲ್|kg|quintal)",
        "en": r"(\d+(?:\.\d+)?)\s*(kg|kilos?|quintals?|tons?)",
    }
    
    # Unit normalization
    UNIT_MAP = {
        "किलो": "kg",
        "kg": "kg",
        "kilo": "kg",
        "kilos": "kg",
        "ಕೆಜಿ": "kg",
        "क्विंटल": "quintal",
        "quintal": "quintal",
        "quintals": "quintal",
        "ಕ್ವಿಂಟಲ್": "quintal",
        "टन": "ton",
        "ton": "ton",
        "tons": "ton",
    }
    
    def __init__(self, llm_provider=None):
        """
        Initialize entity extractor.
        
        Args:
            llm_provider: Optional LLM provider for complex extraction
        """
        self.llm_provider = llm_provider
        logger.info("VoiceEntityExtractor initialized")
    
    async def extract(
        self,
        text: str,
        language: str,
        use_llm: bool = True,
    ) -> ExtractionResult:
        """
        Extract intent and entities from text.
        
        Args:
            text: Transcribed text
            language: Language code
            use_llm: Whether to use LLM for extraction
            
        Returns:
            ExtractionResult with intent and entities
        """
        if not text:
            return ExtractionResult(
                intent=VoiceIntent.UNKNOWN,
                entities={},
                confidence=0.0,
                original_text=text,
                language=language,
            )
        
        # Normalize text
        text_lower = text.lower().strip()
        
        # Try pattern-based extraction first (fast)
        result = self._extract_with_patterns(text_lower, language)
        
        # If low confidence and LLM available, use LLM
        if result.confidence < 0.7 and use_llm and self.llm_provider:
            llm_result = await self._extract_with_llm(text, language)
            if llm_result.confidence > result.confidence:
                return llm_result
        
        return result
    
    def _extract_with_patterns(self, text: str, language: str) -> ExtractionResult:
        """Extract using pattern matching (fast, no LLM)"""
        
        # Detect intent
        intent, intent_confidence = self._detect_intent(text, language)
        
        # Extract entities based on intent
        entities = {}
        
        if intent == VoiceIntent.CREATE_LISTING:
            entities = self._extract_listing_entities(text, language)
        elif intent == VoiceIntent.CHECK_PRICE:
            entities = self._extract_price_entities(text, language)
        elif intent == VoiceIntent.TRACK_ORDER:
            entities = self._extract_order_entities(text, language)
        
        # Calculate overall confidence
        confidence = intent_confidence
        if entities:
            confidence = min(intent_confidence + 0.1, 0.95)
        
        return ExtractionResult(
            intent=intent,
            entities=entities,
            confidence=confidence,
            original_text=text,
            language=language,
        )
    
    def _detect_intent(self, text: str, language: str) -> tuple[VoiceIntent, float]:
        """Detect intent from text"""
        scores = {}
        
        for intent, lang_keywords in self.INTENT_KEYWORDS.items():
            keywords = lang_keywords.get(language, []) + lang_keywords.get("en", [])
            matches = sum(1 for kw in keywords if kw.lower() in text.lower())
            if matches > 0:
                scores[intent] = matches / len(keywords) if keywords else 0
        
        if not scores:
            return VoiceIntent.UNKNOWN, 0.0
        
        # Get highest scoring intent
        best_intent = max(scores, key=scores.get)
        confidence = min(scores[best_intent] * 2, 0.9)  # Scale up
        
        return best_intent, confidence
    
    def _extract_listing_entities(self, text: str, language: str) -> dict:
        """Extract entities for create listing intent"""
        entities = {}
        
        # Extract crop
        for crop_name, crop_english in self.CROP_NAMES.items():
            if crop_name.lower() in text.lower():
                entities["crop"] = crop_english
                break
        
        # Extract quantity
        pattern = self.QUANTITY_PATTERNS.get(language, self.QUANTITY_PATTERNS["en"])
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            entities["quantity"] = float(match.group(1))
            unit_raw = match.group(2).lower()
            entities["unit"] = self.UNIT_MAP.get(unit_raw, unit_raw)
        
        return entities
    
    def _extract_price_entities(self, text: str, language: str) -> dict:
        """Extract entities for check price intent"""
        entities = {}
        
        # Extract crop
        for crop_name, crop_english in self.CROP_NAMES.items():
            if crop_name.lower() in text.lower():
                entities["crop"] = crop_english
                break
        
        # Extract location
        pattern = self.LOCATION_PATTERNS.get(language, self.LOCATION_PATTERNS["en"])
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            entities["location"] = match.group(1)
        
        return entities
    
    def _extract_order_entities(self, text: str, language: str) -> dict:
        """Extract entities for track order intent"""
        entities = {}
        
        # Extract order ID (if mentioned)
        order_match = re.search(r"(?:order|ऑर्डर|आर्डर)\s*#?\s*(\w+)", text, re.IGNORECASE)
        if order_match:
            entities["order_id"] = order_match.group(1)
        
        return entities
    
    async def _extract_with_llm(self, text: str, language: str) -> ExtractionResult:
        """Extract using LLM for complex cases"""
        if not self.llm_provider:
            return ExtractionResult(
                intent=VoiceIntent.UNKNOWN,
                entities={},
                confidence=0.0,
                original_text=text,
                language=language,
            )
        
        prompt = f"""Extract intent and entities from this farmer's voice query.

Query (language: {language}): "{text}"

Respond in JSON format:
{{
    "intent": "create_listing" | "check_price" | "track_order" | "my_listings" | "help" | "greeting" | "unknown",
    "entities": {{
        "crop": "crop name in English",
        "quantity": number,
        "unit": "kg" | "quintal" | "ton",
        "location": "location name"
    }},
    "confidence": 0.0-1.0
}}

Only include entities that are clearly mentioned. Return only JSON."""

        try:
            response = await self.llm_provider.generate(prompt, max_tokens=200)
            
            # Parse JSON response
            import json
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                
                intent_str = data.get("intent", "unknown")
                try:
                    intent = VoiceIntent(intent_str)
                except ValueError:
                    intent = VoiceIntent.UNKNOWN
                
                return ExtractionResult(
                    intent=intent,
                    entities=data.get("entities", {}),
                    confidence=data.get("confidence", 0.7),
                    original_text=text,
                    language=language,
                )
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
        
        return ExtractionResult(
            intent=VoiceIntent.UNKNOWN,
            entities={},
            confidence=0.0,
            original_text=text,
            language=language,
        )
    
    def get_supported_intents(self) -> list[str]:
        """Get list of supported intents"""
        return [intent.value for intent in VoiceIntent if intent != VoiceIntent.UNKNOWN]
