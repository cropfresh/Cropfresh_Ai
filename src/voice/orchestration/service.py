"""Router-first voice orchestration for Sprint 10."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from loguru import logger

from src.agents.voice.handlers import handle_create_listing, handle_my_listings
from src.agents.voice.templates import RESPONSE_TEMPLATES
from src.voice.entity_extractor import ExtractionResult, VoiceEntityExtractor, VoiceIntent

from .models import VOICE_PERSONAS, VoiceOrchestrationResult, VoiceRoute, VoiceWorkflowSession


class VoiceOrchestrator:
    """Route voice turns to the first Sprint 10 specialist roster."""

    def __init__(
        self,
        *,
        supervisor: Any = None,
        tool_registry: Any = None,
        listing_service: Any = None,
        logistics_agent: Any = None,
        state_manager: Any = None,
        entity_extractor: VoiceEntityExtractor | None = None,
        llm_provider: Any = None,
    ) -> None:
        self.supervisor = supervisor
        self.tool_registry = tool_registry
        self.listing_service = listing_service
        self.logistics_agent = logistics_agent
        self.state_manager = state_manager
        self.entity_extractor = entity_extractor or VoiceEntityExtractor(llm_provider=llm_provider)

    async def handle_turn(
        self,
        *,
        text: str,
        language: str,
        user_id: str,
        session_id: str,
        workflow_context: dict[str, Any] | None = None,
        extraction: ExtractionResult | None = None,
    ) -> VoiceOrchestrationResult | None:
        """Return a routed voice response when this service can handle the turn."""
        session = VoiceWorkflowSession(
            user_id=user_id,
            language=language or "en",
            context=dict(workflow_context or {}),
        )
        extraction = extraction or await self.entity_extractor.extract(
            text,
            session.language,
            context_intent=str(session.context.get("pending_intent") or ""),
        )
        route = self._route_turn(text=text, extraction=extraction, session=session)
        if route is None:
            return None

        if route.intent == VoiceIntent.CHECK_PRICE.value:
            return await self._handle_price_turn(
                route=route,
                extraction=extraction,
                session=session,
            )

        if route.intent in {VoiceIntent.CREATE_LISTING.value, VoiceIntent.MY_LISTINGS.value}:
            return await self._handle_listing_turn(
                route=route,
                extraction=extraction,
                session=session,
            )

        if route.intent == "logistics":
            return await self._handle_logistics_turn(
                route=route,
                text=text,
                extraction=extraction,
                session=session,
                session_id=session_id,
            )

        if route.intent == "fallback":
            return await self._handle_fallback_turn(
                route=route,
                text=text,
                extraction=extraction,
                session=session,
                session_id=session_id,
            )

        return None

    def _route_turn(
        self,
        *,
        text: str,
        extraction: ExtractionResult,
        session: VoiceWorkflowSession,
    ) -> VoiceRoute | None:
        pending_intent = str(session.context.get("pending_intent") or "")
        if extraction.intent == VoiceIntent.CHECK_PRICE or pending_intent == VoiceIntent.CHECK_PRICE.value:
            return VoiceRoute(
                persona=VOICE_PERSONAS[VoiceIntent.CHECK_PRICE.value],
                voice_agent_name="arjun_market_agent",
                downstream_target="price_api",
                intent=VoiceIntent.CHECK_PRICE.value,
                reasoning="Voice intent and workflow context resolve to market pricing.",
            )

        if extraction.intent in {VoiceIntent.CREATE_LISTING, VoiceIntent.MY_LISTINGS} or pending_intent == VoiceIntent.CREATE_LISTING.value:
            intent_value = extraction.intent.value if extraction.intent != VoiceIntent.UNKNOWN else pending_intent
            return VoiceRoute(
                persona=VOICE_PERSONAS[VoiceIntent.CREATE_LISTING.value],
                voice_agent_name="priya_farmer_assistant",
                downstream_target="crop_listing_agent",
                intent=intent_value or VoiceIntent.CREATE_LISTING.value,
                reasoning="Voice intent and workflow context resolve to listing management.",
            )

        routing = self._route_rule_based(text)
        if routing is None:
            return None

        if routing.agent_name in {"commerce_agent", "pricing_agent", "price_prediction_agent"}:
            return VoiceRoute(
                persona=VOICE_PERSONAS[VoiceIntent.CHECK_PRICE.value],
                voice_agent_name="arjun_market_agent",
                downstream_target=routing.agent_name,
                intent=VoiceIntent.CHECK_PRICE.value,
                reasoning=routing.reasoning,
            )

        if routing.agent_name == "crop_listing_agent":
            return VoiceRoute(
                persona=VOICE_PERSONAS[VoiceIntent.CREATE_LISTING.value],
                voice_agent_name="priya_farmer_assistant",
                downstream_target="crop_listing_agent",
                intent=VoiceIntent.CREATE_LISTING.value,
                reasoning=routing.reasoning,
            )

        if routing.agent_name == "logistics_agent":
            return VoiceRoute(
                persona=VOICE_PERSONAS["logistics"],
                voice_agent_name="ravi_logistics_agent",
                downstream_target="logistics_agent",
                intent="logistics",
                reasoning=routing.reasoning,
            )

        if self.supervisor is None:
            return None

        return VoiceRoute(
            persona=VOICE_PERSONAS["fallback"],
            voice_agent_name="admin_supervisor",
            downstream_target=routing.agent_name or "general_agent",
            intent="fallback",
            reasoning=routing.reasoning,
        )

    async def _handle_price_turn(
        self,
        *,
        route: VoiceRoute,
        extraction: ExtractionResult,
        session: VoiceWorkflowSession,
    ) -> VoiceOrchestrationResult:
        context = session.context
        pending = dict(context.get("pending_price_query") or {})
        pending.update({key: value for key, value in extraction.entities.items() if value not in (None, "")})
        context["pending_intent"] = VoiceIntent.CHECK_PRICE.value
        context["pending_price_query"] = pending

        crop = str(pending.get("crop") or "").strip()
        if not crop:
            return VoiceOrchestrationResult(
                response_text=self._price_crop_prompt(session.language),
                persona=route.persona,
                agent_name=route.voice_agent_name,
                routed_intent=route.intent,
                workflow_updates=context,
                metadata={"downstream_target": route.downstream_target},
            )

        location = str(
            pending.get("location")
            or pending.get("district")
            or context.get("last_market_location")
            or "Kolar"
        ).strip()

        price_value = 25.0
        unit = "kg"
        source = "fallback"
        price_date = None
        tools_used: list[str] = []

        if self.tool_registry is not None:
            tool_result = await self.tool_registry.execute(
                "price_api",
                commodity=crop,
                district=location,
                market=location,
            )
            if tool_result.success:
                tools_used.append("price_api")
                price_value, unit, source, location, price_date = self._extract_price_result(
                    tool_result.result,
                    default_location=location,
                )
            else:
                logger.warning("Voice price tool failed: {}", tool_result.error)

        template = self._resolve_template(VoiceIntent.CHECK_PRICE, session.language)
        response_text = template.format(
            crop=crop,
            location=location,
            price=f"{price_value:.0f}",
            unit=unit,
        )

        context.pop("pending_intent", None)
        context.pop("pending_price_query", None)
        context["last_market_location"] = location
        context["live_market_context"] = {
            "commodity": crop,
            "location": location,
            "price_per_unit": f"{price_value:.2f}",
            "unit": unit,
            "source": source,
            "price_date": price_date,
        }

        return VoiceOrchestrationResult(
            response_text=response_text,
            persona=route.persona,
            agent_name=route.voice_agent_name,
            routed_intent=route.intent,
            tools_used=tools_used,
            workflow_updates=context,
            metadata={
                "downstream_target": route.downstream_target,
                "source": source,
                "price_date": price_date,
            },
        )

    async def _handle_listing_turn(
        self,
        *,
        route: VoiceRoute,
        extraction: ExtractionResult,
        session: VoiceWorkflowSession,
    ) -> VoiceOrchestrationResult:
        template = self._resolve_template(
            VoiceIntent(route.intent),
            session.language,
        )
        tools_used: list[str] = []

        agent_adapter = SimpleNamespace(listing_service=self.listing_service)
        if route.intent == VoiceIntent.MY_LISTINGS.value:
            response_text = await handle_my_listings(agent_adapter, template, session)
            if self.listing_service is not None:
                tools_used.append("listing_get_farmer_listings")
        else:
            before_listing_id = session.context.get("last_listing_id")
            response_text = await handle_create_listing(
                agent_adapter,
                template,
                extraction.entities,
                session,
            )
            after_listing_id = session.context.get("last_listing_id")
            if after_listing_id and after_listing_id != before_listing_id:
                tools_used.append("listing_create")

        return VoiceOrchestrationResult(
            response_text=response_text,
            persona=route.persona,
            agent_name=route.voice_agent_name,
            routed_intent=route.intent,
            tools_used=tools_used,
            workflow_updates=session.context,
            metadata={"downstream_target": route.downstream_target},
        )

    async def _handle_logistics_turn(
        self,
        *,
        route: VoiceRoute,
        text: str,
        extraction: ExtractionResult,
        session: VoiceWorkflowSession,
        session_id: str,
    ) -> VoiceOrchestrationResult:
        agent = self.logistics_agent or self._resolve_supervisor_agent("logistics_agent")
        if agent is None:
            return VoiceOrchestrationResult(
                response_text=(
                    "I can help with delivery planning if you share pickup, destination, and quantity."
                ),
                persona=route.persona,
                agent_name=route.voice_agent_name,
                routed_intent=route.intent,
                workflow_updates=session.context,
                metadata={"downstream_target": route.downstream_target},
            )

        response = await agent.process(
            text,
            context=await self._build_agent_context(
                session_id=session_id,
                language=session.language,
                user_id=session.user_id,
                entities=extraction.entities,
                workflow_context=session.context,
            ),
        )
        return VoiceOrchestrationResult(
            response_text=response.content,
            persona=route.persona,
            agent_name=route.voice_agent_name,
            routed_intent=route.intent,
            tools_used=list(response.tools_used),
            workflow_updates=session.context,
            metadata={
                "downstream_target": route.downstream_target,
                "confidence": response.confidence,
            },
        )

    async def _handle_fallback_turn(
        self,
        *,
        route: VoiceRoute,
        text: str,
        extraction: ExtractionResult,
        session: VoiceWorkflowSession,
        session_id: str,
    ) -> VoiceOrchestrationResult | None:
        if self.supervisor is None:
            return None

        agent = self._resolve_supervisor_agent(route.downstream_target)
        if agent is None:
            agent = getattr(self.supervisor, "_fallback_agent", None)
        if agent is None:
            return None

        response = await agent.process(
            text,
            context=await self._build_agent_context(
                session_id=session_id,
                language=session.language,
                user_id=session.user_id,
                entities=extraction.entities,
                workflow_context=session.context,
            ),
        )
        return VoiceOrchestrationResult(
            response_text=response.content,
            persona=route.persona,
            agent_name=route.voice_agent_name,
            routed_intent=route.intent,
            tools_used=list(response.tools_used),
            workflow_updates=session.context,
            metadata={
                "downstream_target": route.downstream_target,
                "confidence": response.confidence,
            },
        )

    async def _build_agent_context(
        self,
        *,
        session_id: str,
        language: str,
        user_id: str,
        entities: dict[str, Any],
        workflow_context: dict[str, Any],
    ) -> dict[str, Any]:
        context = {
            "farmer_id": user_id,
            "user_profile": {"language": language, "language_pref": language},
            "entities": dict(entities),
            "workflow_context": dict(workflow_context),
            "speaker_context": {
                "active_speaker_id": workflow_context.get("active_speaker_id"),
                "known_speakers": list(workflow_context.get("known_speakers") or []),
            },
        }
        if self.state_manager is None:
            return context

        persisted = await self.state_manager.get_context(session_id)
        if persisted is None:
            return context

        context["conversation_summary"] = self.state_manager.get_conversation_summary(persisted)
        context["user_profile"] = {
            **persisted.user_profile,
            **context["user_profile"],
        }
        context["entities"] = {
            **persisted.entities,
            **context["entities"],
        }
        context["speaker_context"] = {
            "active_speaker_id": persisted.active_speaker_id,
            "known_speakers": [
                profile.model_dump()
                for profile in persisted.speaker_profiles.values()
            ],
        }
        return context

    def _route_rule_based(self, text: str) -> Any | None:
        if self.supervisor is not None and hasattr(self.supervisor, "_route_rule_based"):
            return self.supervisor._route_rule_based(text)
        return None

    def _resolve_supervisor_agent(self, name: str) -> Any | None:
        if self.supervisor is None:
            return None
        return getattr(self.supervisor, "_agents", {}).get(name)

    def _resolve_template(self, intent: VoiceIntent, language: str) -> str:
        templates = RESPONSE_TEMPLATES.get(intent, {})
        return templates.get(language, templates.get("en", ""))

    def _extract_price_result(
        self,
        payload: Any,
        *,
        default_location: str,
    ) -> tuple[float, str, str, str, str | None]:
        if not isinstance(payload, dict):
            return 25.0, "kg", "fallback", default_location, None

        canonical_rates = payload.get("canonical_rates") or []
        first = canonical_rates[0] if canonical_rates else {}
        price_value = first.get("modal_price") or first.get("price_value") or first.get("max_price") or first.get("min_price") or 25.0
        unit = str(first.get("unit") or "kg")
        location = str(first.get("location_label") or first.get("market") or default_location)
        source = str(first.get("source") or "fallback")
        price_date = first.get("price_date")
        if isinstance(price_value, str):
            try:
                price_value = float(price_value)
            except ValueError:
                price_value = 25.0
        return float(price_value), unit, source, location, str(price_date) if price_date else None

    def _price_crop_prompt(self, language: str) -> str:
        prompts = {
            "en": "Which crop price do you want to check?",
            "hi": "किस फसल का भाव जानना है?",
            "kn": "ಯಾವ ಬೆಳೆ ಬೆಲೆ ತಿಳಿಯಬೇಕು?",
        }
        return prompts.get(language, prompts["en"])
