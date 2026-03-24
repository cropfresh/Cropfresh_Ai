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
from src.memory.state_manager import VoiceSessionState, VoiceTurn
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
        self.state_manager = kwargs.get("state_manager")
        self.orchestrator = kwargs.get("orchestrator")
        self._sessions: dict[str, VoiceSession] = {}

    async def process_voice(
        self,
        audio: bytes,
        user_id: str,
        session_id: Optional[str] = None,
        language: str = "auto",
        speaker_id: str | None = None,
        speaker_label: str | None = None,
        speaker_role: str | None = None,
    ) -> VoiceResponse:
        """Process audio through the full voice pipeline."""
        session = self._get_or_create_session(user_id, session_id, language)
        await self._sync_session_from_state(session)
        self._apply_speaker_context(
            session,
            speaker_id=speaker_id,
            speaker_label=speaker_label,
            speaker_role=speaker_role,
        )
        await self._transition_voice_state(
            session.session_id,
            VoiceSessionState.TRANSCRIBING,
            reason="voice_rest_stt",
        )

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
        await self._transition_voice_state(
            session.session_id,
            VoiceSessionState.THINKING,
            reason="voice_rest_router",
        )
        response_text = await self._generate_response(extraction, session)

        # Step 4: TTS
        await self._transition_voice_state(
            session.session_id,
            VoiceSessionState.SPEAKING,
            reason="voice_rest_tts",
        )
        response_audio = await self._synthesize(response_text, session.language)

        # Step 5: Update history
        session.add_turn(text, response_text)
        await self._persist_shared_session(
            session,
            text,
            response_text,
            speaker_id=speaker_id,
            speaker_label=speaker_label,
            speaker_role=speaker_role,
        )
        await self._transition_voice_state(
            session.session_id,
            VoiceSessionState.IDLE,
            reason="voice_rest_turn_complete",
        )

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

        if self.orchestrator is not None:
            outcome = await self.orchestrator.handle_turn(
                text=extraction.original_text,
                language=lang,
                user_id=session.user_id,
                session_id=session.session_id,
                workflow_context=session.context,
                extraction=extraction,
            )
            if outcome is not None:
                session.context.clear()
                session.context.update(outcome.workflow_updates)
                session.context["voice_persona"] = outcome.persona
                session.context["routed_agent"] = outcome.agent_name
                session.context["last_tools_used"] = ",".join(outcome.tools_used)
                return outcome.response_text

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
        speaker_id: str | None = None,
        speaker_label: str | None = None,
        speaker_role: str | None = None,
    ) -> VoiceResponse:
        """Process already-transcribed text, bypassing STT."""
        session = self._get_or_create_session(user_id, session_id, language)
        await self._sync_session_from_state(session)
        self._apply_speaker_context(
            session,
            speaker_id=speaker_id,
            speaker_label=speaker_label,
            speaker_role=speaker_role,
        )

        if session.language == "auto":
            detected_lang = self._detect_language_from_text(text)
            session.language = detected_lang if detected_lang != "unknown" else "hi"

        pending_intent = session.context.get("pending_intent", "")
        extraction = await self._extract_entities(text, session.language, pending_intent)

        await self._transition_voice_state(
            session.session_id,
            VoiceSessionState.THINKING,
            reason="voice_text_router",
        )
        response_text = await self._generate_response(extraction, session)
        await self._transition_voice_state(
            session.session_id,
            VoiceSessionState.SPEAKING,
            reason="voice_text_tts",
        )
        response_audio = await self._synthesize(response_text, session.language)
        session.add_turn(text, response_text)
        await self._persist_shared_session(
            session,
            text,
            response_text,
            speaker_id=speaker_id,
            speaker_label=speaker_label,
            speaker_role=speaker_role,
        )
        await self._transition_voice_state(
            session.session_id,
            VoiceSessionState.IDLE,
            reason="voice_text_turn_complete",
        )

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

    async def _sync_session_from_state(self, session: VoiceSession) -> None:
        """Hydrate local voice session state from the shared conversation store."""
        if self.state_manager is None:
            return

        context = await self.state_manager.ensure_session(
            session.session_id,
            user_id=session.user_id,
            user_profile={"language": session.language},
        )

        persisted_language = (
            context.user_profile.get("language_pref")
            or context.user_profile.get("language")
            or context.language
        )
        if session.language == "auto" and persisted_language:
            session.language = str(persisted_language)

        for key, value in context.active_workflow.items():
            session.context.setdefault(key, value)

        if context.active_speaker_id:
            session.context.setdefault("active_speaker_id", context.active_speaker_id)
        if context.speaker_profiles:
            session.context.setdefault(
                "known_speakers",
                sorted(context.speaker_profiles.keys()),
            )
            active_profile = context.speaker_profiles.get(
                context.active_speaker_id or ""
            )
            if active_profile is not None:
                if active_profile.label:
                    session.context.setdefault(
                        "active_speaker_label",
                        active_profile.label,
                    )
                if active_profile.role:
                    session.context.setdefault(
                        "active_speaker_role",
                        active_profile.role,
                    )

        if not session.history and context.recent_turns:
            session.history = [
                {"user": turn.user_text, "bot": turn.assistant_text}
                for turn in context.recent_turns[-5:]
            ]

    def _apply_speaker_context(
        self,
        session: VoiceSession,
        *,
        speaker_id: str | None = None,
        speaker_label: str | None = None,
        speaker_role: str | None = None,
    ) -> None:
        """Apply per-turn speaker hints to the in-memory session context."""
        if speaker_id:
            session.context["active_speaker_id"] = speaker_id
        if speaker_label:
            session.context["active_speaker_label"] = speaker_label
        if speaker_role:
            session.context["active_speaker_role"] = speaker_role

        known_speakers = list(session.context.get("known_speakers") or [])
        if speaker_id and speaker_id not in known_speakers:
            known_speakers.append(speaker_id)
        if known_speakers:
            session.context["known_speakers"] = known_speakers

    async def _persist_shared_session(
        self,
        session: VoiceSession,
        user_text: str,
        response_text: str,
        *,
        speaker_id: str | None = None,
        speaker_label: str | None = None,
        speaker_role: str | None = None,
    ) -> None:
        """Persist reconnect-safe voice state after a completed turn."""
        if self.state_manager is None:
            return

        resolved_speaker_id = speaker_id or session.context.get("active_speaker_id")
        resolved_speaker_label = (
            speaker_label or session.context.get("active_speaker_label")
        )
        resolved_speaker_role = (
            speaker_role or session.context.get("active_speaker_role")
        )

        if any(
            value is not None
            for value in (
                resolved_speaker_id,
                resolved_speaker_label,
                resolved_speaker_role,
            )
        ):
            profile = await self.state_manager.update_active_speaker(
                session.session_id,
                speaker_id=resolved_speaker_id,
                speaker_label=resolved_speaker_label,
                speaker_role=resolved_speaker_role,
                speaker_metadata={"transport": "voice_agent"},
            )
            if profile is not None:
                session.context["active_speaker_id"] = profile.speaker_id
                if profile.label:
                    session.context["active_speaker_label"] = profile.label
                if profile.role:
                    session.context["active_speaker_role"] = profile.role
                known_speakers = set(session.context.get("known_speakers") or [])
                known_speakers.add(profile.speaker_id)
                session.context["known_speakers"] = sorted(known_speakers)

        await self.state_manager.ensure_session(
            session.session_id,
            user_id=session.user_id,
            user_profile={
                "language": session.language,
                "language_pref": session.language,
            },
        )
        await self.state_manager.update_user_profile(
            session.session_id,
            {
                "language": session.language,
                "language_pref": session.language,
            },
        )
        await self.state_manager.update_active_workflow(
            session.session_id,
            dict(session.context),
        )
        await self.state_manager.update_voice_runtime(
            session.session_id,
            current_agent=session.context.get("routed_agent"),
            language=session.language,
        )
        await self.state_manager.append_recent_voice_turn(
            session.session_id,
            VoiceTurn(
                turn_id=f"voice-{uuid4().hex[:12]}",
                user_text=user_text,
                assistant_text=response_text,
                language=session.language,
                speaker_id=session.context.get("active_speaker_id"),
                speaker_label=session.context.get("active_speaker_label"),
                speaker_role=session.context.get("active_speaker_role"),
                speaker_metadata={"transport": "voice_agent"},
                timing={
                    "orchestrated": (
                        1.0 if bool(session.context.get("routed_agent")) else 0.0
                    ),
                },
            ),
        )

    async def _transition_voice_state(
        self,
        session_id: str,
        state: VoiceSessionState,
        *,
        reason: str,
    ) -> None:
        """Best-effort shared voice-state transition for REST/text flows."""
        if self.state_manager is None:
            return

        try:
            await self.state_manager.transition_voice_state(
                session_id,
                state,
                source="voice_agent",
                reason=reason,
                actor="voice_agent",
            )
        except ValueError as exc:
            logger.debug(
                "Ignoring invalid voice state transition for {}: {}",
                session_id,
                exc,
            )

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
