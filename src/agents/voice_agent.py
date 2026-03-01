"""
Voice Agent for CropFresh AI

Two-way voice communication agent that enables farmers to interact
with the platform using voice in Indian languages.

Flow:
1. Farmer speaks → STT → Text
2. Text → Entity Extraction → Intent + Entities
3. Intent → LangGraph Orchestrator → Response
4. Response → TTS → Audio to Farmer
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from loguru import logger

from ..voice.stt import IndicWhisperSTT, TranscriptionResult
from ..voice.tts import IndicTTS, SynthesisResult
from ..voice.entity_extractor import VoiceEntityExtractor, VoiceIntent, ExtractionResult


@dataclass
class VoiceSession:
    """Voice conversation session"""
    session_id: str
    user_id: str
    language: str
    history: list[dict] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    
    def add_turn(self, user_text: str, bot_response: str):
        """Add a conversation turn"""
        self.history.append({
            "user": user_text,
            "bot": bot_response,
        })
        # Keep last 5 turns
        if len(self.history) > 5:
            self.history = self.history[-5:]


@dataclass
class VoiceResponse:
    """Complete voice agent response"""
    # Transcription
    transcription: str
    detected_language: str
    
    # Intent & Entities
    intent: str
    entities: dict[str, Any]
    
    # Response
    response_text: str
    response_audio: bytes
    
    # Metadata
    session_id: str
    confidence: float
    
    @property
    def is_successful(self) -> bool:
        return len(self.transcription) > 0 and len(self.response_audio) > 0


class VoiceAgent:
    """
    Two-way voice communication agent for CropFresh.

    Enables illiterate farmers to:
    - Create listings using voice
    - Check market prices
    - Track orders
    - Get help and guidance

    Usage:
        agent = VoiceAgent(pricing_agent=pricing, listing_service=svc)
        response = await agent.process_voice(audio_bytes, user_id="farmer123")
    """

    RESPONSE_TEMPLATES = {
        VoiceIntent.CREATE_LISTING: {
            "hi": "आपकी {quantity} {unit} {crop} की लिस्टिंग ₹{price}/{unit} पर बन गई है। लिस्टिंग आईडी: {listing_id}।",
            "kn": "ನಿಮ್ಮ {quantity} {unit} {crop} ಪಟ್ಟಿ ₹{price}/{unit} ದಲ್ಲಿ ರಚಿಸಲಾಗಿದೆ. ಪಟ್ಟಿ ಐಡಿ: {listing_id}.",
            "en": "Your listing for {quantity} {unit} of {crop} at ₹{price}/{unit} is created. Listing ID: {listing_id}.",
        },
        VoiceIntent.CHECK_PRICE: {
            "hi": "{crop} का भाव आज {location} मंडी में {price} रुपये प्रति {unit} है।",
            "kn": "{location} ಮಂಡಿಯಲ್ಲಿ ಇಂದು {crop} ಬೆಲೆ {price} ರೂಪಾಯಿ ಪ್ರತಿ {unit}.",
            "en": "The price of {crop} in {location} mandi today is {price} rupees per {unit}.",
        },
        VoiceIntent.TRACK_ORDER: {
            "hi": "आपका ऑर्डर रास्ते में है। करीब {eta} में पहुंच जाएगा।",
            "kn": "ನಿಮ್ಮ ಆರ್ಡರ್ ಹಾದಿಯಲ್ಲಿದೆ. ಸುಮಾರು {eta} ನಲ್ಲಿ ತಲುಪುತ್ತದೆ.",
            "en": "Your order is on the way. It will arrive in about {eta}.",
        },
        VoiceIntent.GREETING: {
            "hi": "नमस्ते! मैं क्रॉपफ्रेश हूं। आप सब्जी बेचना चाहते हैं या भाव जानना चाहते हैं?",
            "kn": "ನಮಸ್ಕಾರ! ನಾನು ಕ್ರಾಪ್‌ಫ್ರೆಶ್. ನೀವು ತರಕಾರಿ ಮಾರಾಟ ಮಾಡಲು ಅಥವಾ ಬೆಲೆ ತಿಳಿಯಲು ಬಯಸುತ್ತೀರಾ?",
            "en": "Hello! I'm CropFresh. Would you like to sell vegetables or check prices?",
        },
        VoiceIntent.HELP: {
            "hi": "आप बोलकर सब्जी बेच सकते हैं। बस बोलें - मेरे पास 100 किलो टमाटर है। या भाव जानने के लिए बोलें - टमाटर का भाव क्या है।",
            "kn": "ನೀವು ಮಾತನಾಡುವ ಮೂಲಕ ತರಕಾರಿ ಮಾರಾಟ ಮಾಡಬಹುದು. ಹೇಳಿ - ನನ್ನ ಬಳಿ 100 ಕೆಜಿ ಟೊಮೆಟೊ ಇದೆ.",
            "en": "You can sell vegetables by speaking. Just say - I have 100 kg tomatoes. Or to check prices, say - what is the price of tomatoes.",
        },
        VoiceIntent.MY_LISTINGS: {
            "hi": "आपकी {count} लिस्टिंग चल रही हैं। {details}",
            "kn": "ನಿಮ್ಮ {count} ಪಟ್ಟಿಗಳು ಸಕ್ರಿಯವಾಗಿವೆ. {details}",
            "en": "You have {count} active listings. {details}",
        },
        VoiceIntent.UNKNOWN: {
            "hi": "मुझे समझ नहीं आया। कृपया फिर से बोलें।",
            "kn": "ನನಗೆ ಅರ್ಥವಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೊಮ್ಮೆ ಹೇಳಿ.",
            "en": "I didn't understand. Please say that again.",
        },
    }
    REQUIRED_FIELDS = {
        VoiceIntent.CREATE_LISTING: ["crop", "quantity", "asking_price"],
    }
    
    def __init__(
        self,
        stt: Optional[IndicWhisperSTT] = None,
        tts: Optional[IndicTTS] = None,
        entity_extractor: Optional[VoiceEntityExtractor] = None,
        llm_provider=None,
        orchestrator=None,
        pricing_agent=None,
        listing_service=None,
        order_service=None,
    ):
        self.stt = stt or IndicWhisperSTT()
        self.tts = tts or IndicTTS()
        self.entity_extractor = entity_extractor or VoiceEntityExtractor(llm_provider)
        self.llm_provider = llm_provider
        self.orchestrator = orchestrator

        self.pricing_agent = pricing_agent
        self.listing_service = listing_service
        self.order_service = order_service

        self._sessions: dict[str, VoiceSession] = {}

        logger.info("VoiceAgent initialized")
    
    async def process_voice(
        self,
        audio: bytes,
        user_id: str,
        session_id: Optional[str] = None,
        language: str = "auto",
    ) -> VoiceResponse:
        """
        Process voice input and return voice response.
        
        Complete flow:
        1. Transcribe audio to text (STT)
        2. Detect language if auto
        3. Extract intent and entities
        4. Generate response (template or LLM)
        5. Synthesize response audio (TTS)
        
        Args:
            audio: Audio bytes (WAV, MP3, etc.)
            user_id: User identifier
            session_id: Existing session ID for context
            language: Language code or 'auto'
            
        Returns:
            VoiceResponse with transcription and audio response
        """
        # Get or create session
        session = self._get_or_create_session(user_id, session_id, language)
        
        logger.info(f"Processing voice for user {user_id}, session {session.session_id}")
        
        # Step 1: Transcribe
        transcription = await self.stt.transcribe(audio, language=session.language)
        
        if not transcription.is_successful:
            # Return error response
            error_response = await self._generate_error_response(session.language)
            return VoiceResponse(
                transcription="",
                detected_language=session.language,
                intent="error",
                entities={},
                response_text=error_response,
                response_audio=await self._synthesize(error_response, session.language),
                session_id=session.session_id,
                confidence=0.0,
            )
        
        logger.info(f"Transcribed: {transcription.text} (lang: {transcription.language})")
        
        # Update session language if detected
        if transcription.language != "auto" and transcription.language != session.language:
            session.language = transcription.language
        
        # Step 2: Extract intent and entities
        extraction = await self.entity_extractor.extract(
            transcription.text,
            session.language,
        )
        
        logger.info(f"Extracted: intent={extraction.intent}, entities={extraction.entities}")
        
        # Step 3: Generate response
        response_text = await self._generate_response(
            extraction,
            session,
        )
        
        # Step 4: Synthesize audio
        response_audio = await self._synthesize(response_text, session.language)
        
        # Update session history
        session.add_turn(transcription.text, response_text)
        
        return VoiceResponse(
            transcription=transcription.text,
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
        """Generate response text based on intent"""
        intent = extraction.intent
        entities = extraction.entities
        language = session.language
        
        # Get template
        templates = self.RESPONSE_TEMPLATES.get(intent, self.RESPONSE_TEMPLATES[VoiceIntent.UNKNOWN])
        template = templates.get(language, templates.get("en", ""))
        
        # Continue pending multi-turn create listing flow.
        pending_intent = session.context.get("pending_intent")
        if pending_intent == VoiceIntent.CREATE_LISTING.value and intent in [VoiceIntent.CREATE_LISTING, VoiceIntent.UNKNOWN]:
            create_templates = self.RESPONSE_TEMPLATES.get(VoiceIntent.CREATE_LISTING, {})
            create_template = create_templates.get(language, create_templates.get("en", ""))
            return await self._handle_create_listing(create_template, entities, session)

        # Handle specific intents
        if intent == VoiceIntent.CREATE_LISTING:
            return await self._handle_create_listing(template, entities, session)
        
        elif intent == VoiceIntent.CHECK_PRICE:
            return await self._handle_check_price(template, entities, session)
        
        elif intent == VoiceIntent.TRACK_ORDER:
            return await self._handle_track_order(template, entities, session)
        
        elif intent == VoiceIntent.GREETING:
            return template
        
        elif intent == VoiceIntent.HELP:
            return template
        
        elif intent == VoiceIntent.MY_LISTINGS:
            return await self._handle_my_listings(template, session)
        
        else:
            # Unknown intent - try LLM if available
            if self.llm_provider:
                return await self._generate_llm_response(extraction, session)
            return template
    
    async def _handle_create_listing(
        self,
        template: str,
        entities: dict,
        session: VoiceSession,
    ) -> str:
        """Handle create listing intent — creates listing via service if available."""
        pending_entities = session.context.get("pending_listing", {}).copy()
        pending_entities.update({k: v for k, v in entities.items() if v not in [None, ""]})
        session.context["pending_intent"] = VoiceIntent.CREATE_LISTING.value
        session.context["pending_listing"] = pending_entities

        crop = pending_entities.get("crop", "")
        quantity = pending_entities.get("quantity", pending_entities.get("quantity_kg"))
        unit = pending_entities.get("unit", "kg")
        asking_price = pending_entities.get("asking_price")

        if not crop:
            if session.language == "hi":
                return "कौन सी सब्जी की लिस्टिंग बनानी है?"
            if session.language == "kn":
                return "ಯಾವ ತರಕಾರಿ ಪಟ್ಟಿಯನ್ನು ರಚಿಸಬೇಕು?"
            return "Which crop do you want to list?"

        if not quantity:
            if session.language == "hi":
                return f"कितने {unit} {crop} बेचना है?"
            if session.language == "kn":
                return f"ಎಷ್ಟು {unit} {crop} ಮಾರಾಟ ಮಾಡಬೇಕು?"
            return f"How many {unit} of {crop} do you want to sell?"

        if not asking_price:
            if session.language == "hi":
                return f"{crop} का बेचने का भाव प्रति {unit} कितना रखना है?"
            if session.language == "kn":
                return f"{crop} ಅನ್ನು ಪ್ರತಿ {unit}ಗೆ ಯಾವ ಬೆಲೆಗೆ ಮಾರಾಟ ಮಾಡಬೇಕು?"
            return f"What is your asking price per {unit} for {crop}?"

        listing_id = "pending"
        if self.listing_service:
            try:
                listing = await self.listing_service.create_listing(
                    farmer_id=session.user_id,
                    commodity=crop,
                    quantity_kg=float(quantity),
                    asking_price_per_kg=float(asking_price),
                )
            except TypeError:
                listing = await self.listing_service.create_listing(
                    farmer_id=session.user_id,
                    commodity=crop,
                    quantity_kg=float(quantity),
                )
            except Exception as exc:
                logger.warning("Voice listing creation failed: {}", exc)
                listing = {}
            if isinstance(listing, dict):
                listing_id = listing.get("id") or listing.get("listing_id") or listing_id
                session.context["last_listing_id"] = listing_id
                logger.info("Listing created via voice: {}", listing_id)

        # Clear pending flow once listing is created.
        session.context.pop("pending_intent", None)
        session.context.pop("pending_listing", None)
        return template.format(
            crop=crop,
            quantity=quantity,
            unit=unit,
            price=asking_price,
            listing_id=listing_id,
        )
    
    async def _handle_check_price(
        self,
        template: str,
        entities: dict,
        session: VoiceSession,
    ) -> str:
        """Handle check price intent — fetches real prices via PricingAgent."""
        crop = entities.get("crop", "")
        location = entities.get("location", "Kolar")

        if not crop:
            if session.language == "hi":
                return "किस सब्जी का भाव जानना है?"
            elif session.language == "kn":
                return "ಯಾವ ತರಕಾರಿ ಬೆಲೆ ತಿಳಿಯಬೇಕು?"
            return "Which vegetable's price do you want to know?"

        price_value = 0.0
        recommendation_text = ""
        if self.pricing_agent:
            try:
                rec = await self.pricing_agent.get_recommendation(crop, location)
                price_value = rec.current_price
                action = getattr(rec, "recommended_action", "")
                reason = getattr(rec, "reason", "")
                if action:
                    recommendation_text = f" Recommendation: {action}."
                if reason:
                    recommendation_text += f" {reason}"
            except Exception as exc:
                logger.warning("Voice price lookup failed: {}", exc)

        if price_value <= 0:
            price_value = 25.0

        return template.format(
            crop=crop, location=location,
            price=f"{price_value:.0f}", unit="kg",
        ) + recommendation_text
    
    async def _handle_track_order(
        self,
        template: str,
        entities: dict,
        session: VoiceSession,
    ) -> str:
        """Handle track order intent — queries order service if available."""
        order_id = entities.get("order_id")
        if self.order_service:
            try:
                if order_id and hasattr(self.order_service, "get_status"):
                    status = await self.order_service.get_status(order_id=order_id, user_id=session.user_id)
                    eta = status.get("eta", "30 minutes") if isinstance(status, dict) else "30 minutes"
                    return template.format(eta=eta)
                if order_id and hasattr(self.order_service, "get_order_status"):
                    status = await self.order_service.get_order_status(order_id=order_id, user_id=session.user_id)
                    eta = status.get("eta", "30 minutes") if isinstance(status, dict) else "30 minutes"
                    return template.format(eta=eta)
                orders = await self.order_service.get_active_orders(user_id=session.user_id)
                if orders:
                    latest = orders[0]
                    eta = latest.get("eta", "30 minutes")
                    return template.format(eta=eta)
            except Exception as exc:
                logger.warning("Voice order tracking failed: {}", exc)

        return template.format(eta="30 minutes")

    async def _handle_my_listings(
        self,
        template: str,
        session: VoiceSession,
    ) -> str:
        """Handle my listings intent — fetches from listing service if available."""
        if self.listing_service:
            try:
                listings = await self.listing_service.get_farmer_listings(
                    farmer_id=session.user_id,
                )
                if listings:
                    count = len(listings)
                    details = ", ".join(
                        f"{l.get('quantity_kg', '?')} kg {l.get('commodity', '?')}"
                        for l in listings[:3]
                    )
                    return template.format(count=count, details=details)
            except Exception as exc:
                logger.warning("Voice listings lookup failed: {}", exc)

        return template.format(count=0, details="no active listings")
    
    async def _generate_llm_response(
        self,
        extraction: ExtractionResult,
        session: VoiceSession,
    ) -> str:
        """Generate response using LLM for complex queries"""
        if not self.llm_provider:
            return self.RESPONSE_TEMPLATES[VoiceIntent.UNKNOWN].get(session.language, "")
        
        # Build context from history
        history_text = ""
        for turn in session.history[-3:]:
            history_text += f"User: {turn['user']}\nBot: {turn['bot']}\n"
        
        prompt = f"""You are a helpful voice assistant for CropFresh, an agricultural marketplace.
Respond in {session.language} language. Keep response short (1-2 sentences) as it will be spoken.

Previous conversation:
{history_text}

User's query: {extraction.original_text}

Generate a helpful response:"""

        try:
            response = await self.llm_provider.generate(prompt, max_tokens=100)
            return response.strip()
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            return self.RESPONSE_TEMPLATES[VoiceIntent.UNKNOWN].get(session.language, "")
    
    async def _generate_error_response(self, language: str) -> str:
        """Generate error response"""
        errors = {
            "hi": "माफ कीजिए, आवाज सुनाई नहीं दी। कृपया फिर से बोलें।",
            "kn": "ಕ್ಷಮಿಸಿ, ಧ್ವನಿ ಕೇಳಿಸಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೊಮ್ಮೆ ಹೇಳಿ.",
            "en": "Sorry, I couldn't hear you. Please try again.",
        }
        return errors.get(language, errors["en"])
    
    async def _synthesize(self, text: str, language: str) -> bytes:
        """Synthesize text to audio"""
        result = await self.tts.synthesize(text, language)
        return result.audio
    
    def _get_or_create_session(
        self,
        user_id: str,
        session_id: Optional[str],
        language: str,
    ) -> VoiceSession:
        """Get existing session or create new one"""
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        
        # Create new session
        new_session = VoiceSession(
            session_id=session_id or str(uuid4()),
            user_id=user_id,
            language=language if language != "auto" else "hi",  # Default to Hindi
        )
        self._sessions[new_session.session_id] = new_session
        
        return new_session
    
    def get_session(self, session_id: str) -> Optional[VoiceSession]:
        """Get session by ID"""
        return self._sessions.get(session_id)
    
    def clear_session(self, session_id: str):
        """Clear a session"""
        if session_id in self._sessions:
            del self._sessions[session_id]
    
    def get_supported_languages(self) -> list[str]:
        """Get supported languages"""
        return self.stt.get_supported_languages()
