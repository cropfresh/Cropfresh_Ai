"""
VoiceAgent — Multilingual voice assistant for CropFresh.

Processes audio input through STT → intent extraction → handler routing →
TTS and supports 10 Indian languages with multi-turn session management.
"""

from typing import Optional
from uuid import uuid4

from loguru import logger

from src.agents.voice.handlers import (
    handle_check_price,
    handle_create_listing,
    handle_my_listings,
    handle_track_order,
)
from src.agents.voice.handlers_ext import (
    handle_check_weather,
    handle_dispute_status,
    handle_find_buyer,
    handle_get_advisory,
    handle_quality_check,
    handle_register,
    handle_weekly_demand,
)
from src.agents.voice.models import VoiceResponse, VoiceSession
from src.agents.voice.templates import REQUIRED_FIELDS, RESPONSE_TEMPLATES
from src.voice.entity_extractor import ExtractionResult, VoiceIntent


class VoiceAgent:
    """Multilingual voice agent with intent routing and session management."""

    # Expose as class attrs for test access
    RESPONSE_TEMPLATES = RESPONSE_TEMPLATES
    REQUIRED_FIELDS = REQUIRED_FIELDS

    def __init__(self, stt, tts, entity_extractor, **kwargs):
        self.stt = stt
        self.tts = tts
        self.entity_extractor = entity_extractor
        # Optional service integrations
        self.pricing_agent = kwargs.get("pricing_agent")
        self.listing_service = kwargs.get("listing_service")
        self.order_service = kwargs.get("order_service")
        self.matching_agent = kwargs.get("matching_agent")
        self.quality_agent = kwargs.get("quality_agent")
        self.weather_tool = kwargs.get("weather_tool")
        self.agronomy_agent = kwargs.get("agronomy_agent")
        self.adcl_agent = kwargs.get("adcl_agent")
        self.registration_service = kwargs.get("registration_service")
        self.llm_provider = kwargs.get("llm_provider")
        self._sessions: dict[str, VoiceSession] = {}

    async def process_voice(
        self,
        audio: bytes,
        user_id: str,
        session_id: Optional[str] = None,
        language: str = "auto",
    ) -> VoiceResponse:
        """Process audio through the full voice pipeline."""
        session = self._get_or_create_session(user_id, session_id, language)

        # Step 1: STT
        transcription = await self.stt.transcribe(
            audio, language=session.language,
        )
        text = transcription.text
        if session.language == "auto":
            session.language = transcription.language

        logger.info(
            f"Voice input: user='{user_id}' lang='{session.language}' text='{text}'"
        )

        # Step 2: Entity extraction
        pending_intent = session.context.get("pending_intent", "")
        extraction = await self._extract_entities(text, session.language, pending_intent)

        # Step 3: Generate response
        response_text = await self._generate_response(extraction, session)

        # Step 4: TTS
        response_audio = await self._synthesize(response_text, session.language)

        # Step 5: Update history
        session.add_turn(text, response_text)

        return VoiceResponse(
            transcription=text,
            detected_language=session.language,
            intent=extraction.intent.value,
            entities=extraction.entities,
            response_text=response_text,
            response_audio=response_audio,
            session_id=session.session_id,
            confidence=extraction.confidence,
        )

    async def _generate_response(
        self,
        extraction: ExtractionResult,
        session: VoiceSession,
    ) -> str:
        """Route intent to the correct handler."""
        intent = extraction.intent
        entities = extraction.entities
        lang = session.language

        # Get language-specific template
        templates = self.RESPONSE_TEMPLATES.get(intent, {})
        template = templates.get(lang, templates.get("en", ""))

        # Route to handler
        if intent == VoiceIntent.GREETING:
            return template or "Hello! I'm CropFresh."
        if intent == VoiceIntent.HELP:
            return template or "I can help you sell produce and check prices."
        if intent == VoiceIntent.CREATE_LISTING:
            return await handle_create_listing(self, template, entities, session)
        if intent == VoiceIntent.CHECK_PRICE:
            return await handle_check_price(self, template, entities, session)
        if intent == VoiceIntent.TRACK_ORDER:
            return await handle_track_order(self, template, entities, session)
        if intent == VoiceIntent.MY_LISTINGS:
            return await handle_my_listings(self, template, session)
        if intent == VoiceIntent.FIND_BUYER:
            return await handle_find_buyer(self, template, entities, session)
        if intent == VoiceIntent.CHECK_WEATHER:
            return await handle_check_weather(self, template, entities, session)
        if intent == VoiceIntent.GET_ADVISORY:
            return await handle_get_advisory(self, template, entities, session)
        if intent == VoiceIntent.REGISTER:
            return await handle_register(self, template, entities, session)
        if intent == VoiceIntent.DISPUTE_STATUS:
            return await handle_dispute_status(self, template, entities, session)
        if intent == VoiceIntent.QUALITY_CHECK:
            return await handle_quality_check(self, template, entities, session)
        if intent == VoiceIntent.WEEKLY_DEMAND:
            return await handle_weekly_demand(self, template, entities, session)

        # Pending multi-turn
        pending = session.context.get("pending_intent", "")
        if pending:
            try:
                pending_intent = VoiceIntent(pending)
                return await self._generate_response(
                    ExtractionResult(
                        pending_intent, entities,
                        extraction.confidence, extraction.original_text, lang,
                    ),
                    session,
                )
            except ValueError:
                pass

        # Fallback: LLM or unknown
        if self.llm_provider:
            return await self._generate_llm_response(extraction, session)

        return templates.get(lang, "I didn't understand. Please try again.")

    async def _generate_llm_response(
        self,
        extraction: ExtractionResult,
        session: VoiceSession,
    ) -> str:
        """Generate response using LLM for complex queries."""
        if not self.llm_provider:
            return self.RESPONSE_TEMPLATES[VoiceIntent.UNKNOWN].get(session.language, "")

        history_text = ""
        for turn in session.history[-3:]:
            history_text += f"User: {turn['user']}\nBot: {turn['bot']}\n"

        system_prompt = (
            "You are a helpful voice assistant for CropFresh, "
            "an agricultural marketplace. "
            f"Respond in {session.language} language. "
            "Keep response short (1-2 sentences) as it will be spoken."
        )
        user_prompt = (
            f"Previous conversation:\n{history_text}\n\n"
            f"User's query: {extraction.original_text}\n\n"
            f"Generate a helpful response:"
        )

        try:
            from src.orchestrator.llm_provider import LLMMessage
            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ]
            response = await self.llm_provider.generate(messages, max_tokens=100)
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            return self.RESPONSE_TEMPLATES[VoiceIntent.UNKNOWN].get(session.language, "")

    async def _generate_error_response(self, language: str) -> str:
        """Generate error response."""
        errors = {
            "hi": "माफ कीजिए, आवाज सुनाई नहीं दी। कृपया फिर से बोलें।",
            "kn": "ಕ್ಷಮಿಸಿ, ಧ್ವನಿ ಕೇಳಿಸಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೊಮ್ಮೆ ಹೇಳಿ.",
            "en": "Sorry, I couldn't hear you. Please try again.",
        }
        return errors.get(language, errors["en"])

    async def _synthesize(self, text: str, language: str) -> bytes:
        """Synthesize text to audio."""
        result = await self.tts.synthesize(text, language)
        return result.audio

    def _get_or_create_session(
        self,
        user_id: str,
        session_id: Optional[str],
        language: str,
    ) -> VoiceSession:
        """Get existing session or create new one."""
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]

        new_session = VoiceSession(
            session_id=session_id or str(uuid4()),
            user_id=user_id,
            language=language,
        )
        self._sessions[new_session.session_id] = new_session
        return new_session

    async def handle_text_input(
        self,
        text: str,
        user_id: str,
        session_id: Optional[str] = None,
        language: str = "auto",
    ) -> VoiceResponse:
        """Process already-transcribed text, bypassing STT."""
        session = self._get_or_create_session(user_id, session_id, language)

        if session.language == "auto":
            detected_lang = self._detect_language_from_text(text)
            session.language = detected_lang if detected_lang != "unknown" else "hi"

        pending_intent = session.context.get("pending_intent", "")
        extraction = await self._extract_entities(text, session.language, pending_intent)

        response_text = await self._generate_response(extraction, session)
        response_audio = await self._synthesize(response_text, session.language)
        session.add_turn(text, response_text)

        return VoiceResponse(
            transcription=text,
            detected_language=session.language,
            intent=extraction.intent.value,
            entities=extraction.entities,
            response_text=response_text,
            response_audio=response_audio,
            session_id=session.session_id,
            confidence=extraction.confidence,
        )

    def get_session(self, session_id: str) -> Optional[VoiceSession]:
        """Get session by ID."""
        return self._sessions.get(session_id)

    def clear_session(self, session_id: str):
        """Clear a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def get_supported_languages(self) -> list[str]:
        """Get supported languages."""
        return self.stt.get_supported_languages()

    async def _extract_entities(
        self,
        text: str,
        language: str,
        pending_intent: str = "",
    ) -> ExtractionResult:
        """Call the extractor with compatibility for older stub signatures."""
        try:
            return await self.entity_extractor.extract(
                text,
                language,
                context_intent=pending_intent,
            )
        except TypeError as exc:
            if "context_intent" not in str(exc):
                raise
            return await self.entity_extractor.extract(text, language)

    def _detect_language_from_text(self, text: str) -> str:
        """Use extractor language detection when available, otherwise fall back."""
        detector = getattr(self.entity_extractor, "detect_language_from_text", None)
        if callable(detector):
            return detector(text)
        return "unknown"

    # ═══════════════════════════════════════════════════════════════
    # Backward-compatible _handle_* wrappers (used by unit tests)
    # ═══════════════════════════════════════════════════════════════

    async def _handle_create_listing(self, template, entities, session):
        return await handle_create_listing(self, template, entities, session)

    async def _handle_check_price(self, template, entities, session):
        return await handle_check_price(self, template, entities, session)

    async def _handle_track_order(self, template, entities, session):
        return await handle_track_order(self, template, entities, session)

    async def _handle_my_listings(self, template, session):
        return await handle_my_listings(self, template, session)

    async def _handle_find_buyer(self, template, entities, session):
        return await handle_find_buyer(self, template, entities, session)

    async def _handle_check_weather(self, template, entities, session):
        return await handle_check_weather(self, template, entities, session)

    async def _handle_get_advisory(self, template, entities, session):
        return await handle_get_advisory(self, template, entities, session)

    async def _handle_register(self, template, entities, session):
        return await handle_register(self, template, entities, session)

    async def _handle_dispute_status(self, template, entities, session):
        return await handle_dispute_status(self, template, entities, session)

    async def _handle_quality_check(self, template, entities, session):
        return await handle_quality_check(self, template, entities, session)

    async def _handle_weekly_demand(self, template, entities, session):
        return await handle_weekly_demand(self, template, entities, session)

    # Alias used by WebSocket handler (voice_ws.py)
    process_text = handle_text_input
