"""
Additional intent handlers for the VoiceAgent.

Covers: find_buyer, check_weather, get_advisory, register,
dispute_status, quality_check, weekly_demand.
"""

from loguru import logger

from src.voice.entity_extractor import VoiceIntent


async def handle_find_buyer(agent, template, entities, session):
    """Handle find_buyer — multi-turn until commodity+quantity collected."""
    pending = session.context.get("pending_find_buyer", {}).copy()
    pending.update({k: v for k, v in entities.items() if v not in [None, ""]})
    session.context["pending_intent"] = VoiceIntent.FIND_BUYER.value
    session.context["pending_find_buyer"] = pending

    commodity = pending.get("commodity", "")
    quantity_kg = pending.get("quantity_kg")

    if not commodity:
        if session.language == "hi":
            return "किस फसल के लिए खरीदार चाहिए?"
        if session.language == "kn":
            return "ಯಾವ ಬೆಳೆಗೆ ಖರೀದಿದಾರ ಬೇಕು?"
        return "Which crop do you want to find a buyer for?"

    if not quantity_kg:
        if session.language == "hi":
            return f"कितने किलो {commodity} बेचना है?"
        if session.language == "kn":
            return f"ಎಷ್ಟು ಕೆಜಿ {commodity} ಮಾರಾಟ ಮಾಡಬೇಕು?"
        return f"How many kg of {commodity} do you want to sell?"

    session.context.pop("pending_intent", None)
    session.context.pop("pending_find_buyer", None)

    if not agent.matching_agent:
        if session.language == "hi":
            return "खरीदार खोजने की सेवा अभी उपलब्ध नहीं है। कल फिर कोशिश करें।"
        if session.language == "kn":
            return "ಖರೀದಿದಾರ ಸೇವೆ ಲಭ್ಯವಿಲ್ಲ. ನಾಳೆ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ."
        return "Buyer matching service is not available right now. Try again tomorrow."

    try:
        matches = await agent.matching_agent.find_matches(
            listing_id=session.context.get("last_listing_id", f"voice-{session.user_id}"),
        )
    except Exception as exc:
        logger.warning("Voice buyer matching failed: {}", exc)
        matches = []

    if not matches:
        if session.language == "hi":
            return f"{commodity} के लिए अभी कोई खरीदार नहीं मिला। कल फिर कोशिश करें।"
        if session.language == "kn":
            return f"{commodity} ಗೆ ಈಗ ಯಾವ ಖರೀದಿದಾರ ಸಿಗಲಿಲ್ಲ. ನಾಳೆ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ."
        return f"No buyers found for {commodity} right now. Try again tomorrow."

    top = matches[0]
    return template.format(
        crop=commodity, count=len(matches),
        buyer_name=getattr(top, "buyer_name", "Unknown"),
        buyer_district=getattr(top, "buyer_type", "local"),
        price=getattr(top, "price_fit", 0), qty=quantity_kg,
    )


async def handle_check_weather(agent, template, entities, session):
    """Handle check_weather — fetches forecast via weather_tool."""
    location = entities.get("location", session.context.get("location", "Kolar"))

    if agent.weather_tool:
        try:
            forecast = await agent.weather_tool.get_forecast(location=location)
            return template.format(
                location=location,
                condition=forecast.get("condition", "Clear"),
                temp=forecast.get("temperature", 28),
                advisory=forecast.get("advisory", ""),
            )
        except Exception as exc:
            logger.warning("Voice weather lookup failed: {}", exc)

    if session.language == "hi":
        return f"{location} के लिए मौसम सेवा अभी उपलब्ध नहीं है।"
    if session.language == "kn":
        return f"{location} ಗಾಗಿ ಹವಾಮಾನ ಸೇವೆ ಲಭ್ಯವಿಲ್ಲ."
    return f"Weather service is not available right now for {location}."


async def handle_get_advisory(agent, template, entities, session):
    """Handle get_advisory — queries agronomy agent for crop guidance."""
    crop = entities.get("crop", "")

    if not crop:
        if session.language == "hi":
            return "किस फसल के बारे में सलाह चाहिए?"
        if session.language == "kn":
            return "ಯಾವ ಬೆಳೆಯ ಬಗ್ಗೆ ಸಲಹೆ ಬೇಕು?"
        return "Which crop do you need advice for?"

    if agent.agronomy_agent:
        try:
            response = await agent.agronomy_agent.process(
                f"Give brief farming advice for {crop}",
                context={"language": session.language},
            )
            advisory_text = getattr(response, "content", str(response))
            return template.format(crop=crop, advisory=advisory_text[:200])
        except Exception as exc:
            logger.warning("Voice advisory lookup failed: {}", exc)

    return template.format(crop=crop, advisory="No advisory available at this time.")


async def handle_register(agent, template, entities, session):
    """Handle register — multi-turn collection of name, phone, district."""
    pending = session.context.get("pending_register", {}).copy()
    pending.update({k: v for k, v in entities.items() if v not in [None, ""]})
    session.context["pending_intent"] = VoiceIntent.REGISTER.value
    session.context["pending_register"] = pending

    name = pending.get("name", "")
    phone = pending.get("phone", "")
    district = pending.get("district", "")

    if not name:
        if session.language == "hi":
            return "आपका नाम क्या है?"
        if session.language == "kn":
            return "ನಿಮ್ಮ ಹೆಸರೇನು?"
        return "What is your name?"

    if not phone:
        if session.language == "hi":
            return "आपका मोबाइल नंबर क्या है?"
        if session.language == "kn":
            return "ನಿಮ್ಮ ಮೊಬೈಲ್ ಸಂಖ್ಯೆ ಏನು?"
        return "What is your mobile number?"

    if not district:
        if session.language == "hi":
            return "आप किस जिले में हैं?"
        if session.language == "kn":
            return "ನೀವು ಯಾವ ಜಿಲ್ಲೆಯಲ್ಲಿದ್ದೀರಿ?"
        return "Which district are you in?"

    session.context.pop("pending_intent", None)
    session.context.pop("pending_register", None)

    farmer_id = "pending"
    if agent.registration_service:
        try:
            result = await agent.registration_service.register_farmer(
                name=name, phone=phone, district=district,
            )
            farmer_id = result.get("farmer_id", farmer_id) if isinstance(result, dict) else farmer_id
        except Exception as exc:
            logger.warning("Voice registration failed: {}", exc)

    return template.format(name=name, farmer_id=farmer_id)


async def handle_dispute_status(agent, template, entities, session):
    """Handle dispute_status — queries order service for dispute info."""
    order_id = entities.get("order_id", "")

    if agent.order_service:
        try:
            if hasattr(agent.order_service, "get_dispute_status"):
                dispute = await agent.order_service.get_dispute_status(
                    order_id=order_id, user_id=session.user_id,
                )
                return template.format(
                    dispute_id=dispute.get("dispute_id", order_id or "N/A"),
                    status=dispute.get("status", "Under Review"),
                    notes=dispute.get("notes", ""),
                )
        except Exception as exc:
            logger.warning("Voice dispute lookup failed: {}", exc)

    if session.language == "hi":
        return "विवाद की जानकारी अभी उपलब्ध नहीं है।"
    if session.language == "kn":
        return "ವಿವಾದ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ."
    return "Dispute status is not available right now. Please try again later."


async def handle_quality_check(agent, template, entities, session):
    """Handle quality_check — requests assessment from quality agent."""
    commodity = entities.get("commodity", "")
    listing_id = entities.get("listing_id", session.context.get("last_listing_id", ""))

    if not commodity:
        if session.language == "hi":
            return "किस फसल की गुणवत्ता जाँचनी है?"
        if session.language == "kn":
            return "ಯಾವ ಬೆಳೆಯ ಗುಣಮಟ್ಟ ಪರೀಕ್ಷಿಸಬೇಕು?"
        return "Which crop's quality do you want to check?"

    if not agent.quality_agent:
        if session.language == "hi":
            return "गुणवत्ता जाँच सेवा अभी उपलब्ध नहीं है।"
        if session.language == "kn":
            return "ಗುಣಮಟ್ಟ ಸೇವೆ ಲಭ್ಯವಿಲ್ಲ."
        return "Quality check service is not available right now."

    try:
        result = await agent.quality_agent.execute({
            "commodity": commodity,
            "listing_id": listing_id or f"voice-{session.user_id}",
            "description": f"Voice quality check requested for {commodity}",
        })
        grade = result.get("grade", "B")
        confidence = int(result.get("confidence", 0.7) * 100)
        hitl = result.get("hitl_required", False)
        notes = "Human review required." if hitl else result.get("message", "")
        return template.format(
            commodity=commodity, grade=grade, confidence=confidence, notes=notes,
        )
    except Exception as exc:
        logger.warning("Voice quality check failed: {}", exc)
        return template.format(commodity=commodity, grade="B", confidence=60, notes="")


async def handle_weekly_demand(agent, template, entities, session):
    """Handle weekly_demand — fetches list via adcl_agent."""
    location = entities.get("location", session.context.get("location", "Karnataka"))

    if agent.adcl_agent:
        try:
            if hasattr(agent.adcl_agent, "get_weekly_list"):
                demand = await agent.adcl_agent.get_weekly_list(location=location)
                demand_list = demand if isinstance(demand, str) else ", ".join(str(d) for d in demand)
                return template.format(location=location, demand_list=demand_list)
        except Exception as exc:
            logger.warning("Voice weekly demand lookup failed: {}", exc)

    if session.language == "hi":
        return f"{location} के लिए साप्ताहिक मांग सेवा अभी उपलब्ध नहीं है।"
    if session.language == "kn":
        return f"{location} ಸಾಪ್ತಾಹಿಕ ಬೇಡಿಕೆ ಸೇವೆ ಲಭ್ಯವಿಲ್ಲ."
    return f"Weekly demand service is not available right now for {location}."
