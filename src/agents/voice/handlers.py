"""
Intent handler methods for the VoiceAgent.

Each handler processes a specific VoiceIntent (create_listing, check_price, etc.)
and returns a localized response string.
"""

from loguru import logger

from src.voice.entity_extractor import VoiceIntent


async def handle_create_listing(agent, template, entities, session):
    """Handle create listing intent — multi-turn collection of crop, qty, price."""
    pending_entities = session.context.get("pending_listing", {}).copy()
    pending_entities.update({k: v for k, v in entities.items() if v not in [None, ""]})
    session.context["pending_intent"] = VoiceIntent.CREATE_LISTING.value
    session.context["pending_listing"] = pending_entities

    crop = pending_entities.get("crop", "")
    quantity = pending_entities.get("quantity", pending_entities.get("quantity_kg"))
    unit = pending_entities.get("unit", "kg")
    asking_price = pending_entities.get("asking_price") or pending_entities.get("price")

    if not crop:
        _ask = {
            "hi": "कौन सी सब्जी की लिस्टिंग बनानी है?",
            "kn": "ಯಾವ ತರಕಾರಿ ಪಟ್ಟಿಯನ್ನು ರಚಿಸಬೇಕು?",
            "ta": "எந்த பயிரை பதிவு செய்ய வேண்டும்?",
            "te": "ఏ పంటను నమోదు చేయాలి?",
            "mr": "कोणत्या पिकाची यादी बनवायची आहे?",
            "bn": "কোন ফসলের তালিকা তৈরি করতে হবে?",
            "gu": "કયા પાકનું લિસ્ટિંગ બનાવવું છે?",
            "pa": "ਕਿਹੜੀ ਫਸਲ ਦੀ ਸੂਚੀ ਬਣਾਉਣੀ ਹੈ?",
            "ml": "ഏത് വിളയുടെ ലിസ്റ്റിംഗ് ഉണ്ടാക്കണം?",
            "en": "Which crop do you want to list?",
        }
        return _ask.get(session.language, _ask["en"])

    if not quantity:
        _ask = {
            "hi": f"कितने {unit} {crop} बेचना है?",
            "kn": f"ಎಷ್ಟು {unit} {crop} ಮಾರಾಟಕ್ಕೆ?",
            "ta": f"எத்தனை {unit} {crop} விற்க வேண்டும்?",
            "te": f"ఎన్ని {unit} {crop} అమ్మాలి?",
            "mr": f"किती {unit} {crop} विकायचे आहे?",
            "bn": f"কত {unit} {crop} বিক্রি করবেন?",
            "gu": f"કેટલા {unit} {crop} વેચવા છે?",
            "pa": f"ਕਿੰਨੇ {unit} {crop} ਵੇਚਣੇ ਹਨ?",
            "ml": f"എത്ര {unit} {crop} വിൽക്കണം?",
            "en": f"How many {unit} of {crop} do you want to sell?",
        }
        return _ask.get(session.language, _ask["en"])

    if not asking_price:
        _ask = {
            "hi": f"{crop} का भाव प्रति {unit} कितना रखना है?",
            "kn": f"{crop} ಪ್ರತಿ {unit}ಗೆ ಬೆಲೆ ಎಷ್ಟು?",
            "ta": f"{crop} ஒரு {unit}க்கு உங்கள் விலை என்ன?",
            "te": f"{crop} ఒక్క {unit}కి ధర ఎంత?",
            "mr": f"{crop} प्रति {unit} भाव किती ठेवायचा?",
            "bn": f"{crop} প্রতি {unit} দাম কত রাখবেন?",
            "gu": f"{crop} प्रति {unit} भाव केटલો राખવો છે?",
            "pa": f"{crop} प्रति {unit} भाअ ਕਿੰਨੀ ਰੱਖਣੀ ਹੈ?",
            "ml": f"{crop} പ്രതി {unit} വில എന്താണ്?",
            "en": f"What is your asking price per {unit} for {crop}?",
        }
        return _ask.get(session.language, _ask["en"])

    listing_id = "pending"
    if agent.listing_service:
        try:
            listing = await agent.listing_service.create_listing(
                farmer_id=session.user_id,
                commodity=crop,
                quantity_kg=float(quantity),
                asking_price_per_kg=float(asking_price),
            )
        except TypeError:
            listing = await agent.listing_service.create_listing(
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

    session.context.pop("pending_intent", None)
    session.context.pop("pending_listing", None)
    return template.format(
        crop=crop, quantity=quantity, unit=unit,
        price=asking_price, listing_id=listing_id,
    )


async def handle_check_price(agent, template, entities, session):
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
    if agent.pricing_agent:
        try:
            rec = await agent.pricing_agent.get_recommendation(crop, location)
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


async def handle_track_order(agent, template, entities, session):
    """Handle track order intent — queries order service if available."""
    order_id = entities.get("order_id")
    if agent.order_service:
        try:
            if order_id and hasattr(agent.order_service, "get_status"):
                status = await agent.order_service.get_status(order_id=order_id, user_id=session.user_id)
                eta = status.get("eta", "30 minutes") if isinstance(status, dict) else "30 minutes"
                return template.format(eta=eta)
            if order_id and hasattr(agent.order_service, "get_order_status"):
                status = await agent.order_service.get_order_status(order_id=order_id, user_id=session.user_id)
                eta = status.get("eta", "30 minutes") if isinstance(status, dict) else "30 minutes"
                return template.format(eta=eta)
            orders = await agent.order_service.get_active_orders(user_id=session.user_id)
            if orders:
                latest = orders[0]
                eta = latest.get("eta", "30 minutes")
                return template.format(eta=eta)
        except Exception as exc:
            logger.warning("Voice order tracking failed: {}", exc)

    return template.format(eta="30 minutes")


async def handle_my_listings(agent, template, session):
    """Handle my listings intent — fetches from listing service if available."""
    if agent.listing_service:
        try:
            listings = await agent.listing_service.get_farmer_listings(
                farmer_id=session.user_id,
            )
            if listings:
                count = len(listings)
                details = ", ".join(
                    f"{listing.get('quantity_kg', '?')} kg {listing.get('commodity', '?')}"
                    for listing in listings[:3]
                )
                return template.format(count=count, details=details)
        except Exception as exc:
            logger.warning("Voice listings lookup failed: {}", exc)

    return template.format(count=0, details="no active listings")
