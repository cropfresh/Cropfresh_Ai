"""Additional intent handlers for the VoiceAgent."""

from loguru import logger

from src.agents.voice.contextual_handlers import (
    handle_check_weather,
    handle_get_advisory,
    handle_weekly_demand,
)
from src.voice.entity_extractor import VoiceIntent


async def handle_find_buyer(agent, template, entities, session):
    """Handle find_buyer until commodity and quantity are collected."""
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
        crop=commodity,
        count=len(matches),
        buyer_name=getattr(top, "buyer_name", "Unknown"),
        buyer_district=getattr(top, "buyer_type", "local"),
        price=getattr(top, "price_fit", 0),
        qty=quantity_kg,
    )


async def handle_register(agent, template, entities, session):
    """Handle register via a small multi-turn collection flow."""
    pending = session.context.get("pending_register", {}).copy()
    pending.update({k: v for k, v in entities.items() if v not in [None, ""]})
    session.context["pending_intent"] = VoiceIntent.REGISTER.value
    session.context["pending_register"] = pending
    name = pending.get("name", "")
    phone = pending.get("phone", "")
    district = pending.get("district", "")

    if not name:
        return "What is your name?"
    if not phone:
        return "What is your mobile number?"
    if not district:
        return "Which district are you in?"

    session.context.pop("pending_intent", None)
    session.context.pop("pending_register", None)
    farmer_id = "pending"
    if agent.registration_service:
        try:
            result = await agent.registration_service.register_farmer(
                name=name,
                phone=phone,
                district=district,
            )
            if isinstance(result, dict):
                farmer_id = result.get("farmer_id", farmer_id)
        except Exception as exc:
            logger.warning("Voice registration failed: {}", exc)
    return template.format(name=name, farmer_id=farmer_id)


async def handle_dispute_status(agent, template, entities, session):
    """Handle dispute_status via the shared order service."""
    order_id = entities.get("order_id", "")
    if agent.order_service:
        try:
            if hasattr(agent.order_service, "get_dispute_status"):
                dispute = await agent.order_service.get_dispute_status(
                    order_id=order_id,
                    user_id=session.user_id,
                )
                return template.format(
                    dispute_id=dispute.get("dispute_id", order_id or "N/A"),
                    status=dispute.get("status", "Under Review"),
                    notes=dispute.get("notes", ""),
                )
        except Exception as exc:
            logger.warning("Voice dispute lookup failed: {}", exc)
    return "Dispute status is not available right now. Please try again later."


async def handle_quality_check(agent, template, entities, session):
    """Handle quality_check via the shared quality agent."""
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
            commodity=commodity,
            grade=grade,
            confidence=confidence,
            notes=notes,
        )
    except Exception as exc:
        logger.warning("Voice quality check failed: {}", exc)
        return template.format(commodity=commodity, grade="B", confidence=60, notes="")
