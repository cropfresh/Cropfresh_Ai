# Task 4: Wire Voice Agent TODO Stubs to Real Agent Calls

> **Priority:** 🔴 P0 | **Phase:** 1 | **Effort:** 2–3 days | **Status:** 🟡 In Progress (2026-03-01)  
> **Files:** `src/agents/voice_agent.py`, `src/voice/entity_extractor.py`  
> **Score Target:** 9/10 — All intents route to real services, multi-turn conversations work

---

## 📌 Problem Statement

Voice agent handlers currently return template strings instead of calling real agents. Need to wire STT → Intent Detection → Real Agent Call → TTS for all 10+ intents.

---

## 🏗️ Implementation Spec

### 1. Intent → Agent Routing Map
```python
INTENT_AGENT_MAP = {
    # Existing (wire to real services)
    'create_listing': ('listing_service', 'create'),
    'check_price':    ('pricing_agent', 'get_recommendation'),
    'track_order':    ('order_service', 'get_status'),
    'my_listings':    ('listing_service', 'get_farmer_listings'),
    
    # New intents
    'find_buyer':     ('matching_agent', 'find_matches'),
    'check_weather':  ('weather_tool', 'get_forecast'),
    'get_advisory':   ('agronomy_agent', 'get_advice'),
    'register':       ('registration_service', 'register_farmer'),
    'dispute_status': ('order_service', 'get_dispute_status'),
    'quality_check':  ('quality_agent', 'request_assessment'),
    'weekly_demand':  ('adcl_agent', 'get_weekly_list'),
}
```

### 2. Multi-Turn Conversation State Machine
```python
class ConversationFlow:
    """
    Multi-turn dialog for complex intents (e.g., create_listing).
    
    Example flow:
    Turn 1: "I want to sell tomatoes" → extract: commodity=tomato
    Turn 2: "How many kg?" → "500 kg" → extract: quantity=500
    Turn 3: "What's your asking price?" → "₹25" → extract: price=25
    Turn 4: Confirm → Create listing → Return ID
    """
    
    REQUIRED_FIELDS = {
        'create_listing': ['commodity', 'quantity_kg', 'asking_price'],
        'find_buyer': ['commodity', 'quantity_kg'],
        'register': ['name', 'phone', 'district'],
    }
    
    def get_next_question(self, intent: str, collected: dict) -> Optional[str]:
        """Return the next question to ask, or None if all fields collected."""
        required = self.REQUIRED_FIELDS.get(intent, [])
        for field in required:
            if field not in collected:
                return self.QUESTION_TEMPLATES[field]
        return None  # All collected
```

### 3. New Intent Handlers
```python
async def _handle_find_buyer(self, entities: dict, session: VoiceSession) -> str:
    """Find matching buyers for farmer's produce."""
    if not self.matching_agent:
        return "Buyer matching service not available right now."
    
    matches = await self.matching_agent.find_matches(
        commodity=entities.get('commodity'),
        quantity_kg=entities.get('quantity'),
        farmer_location=session.context.get('location'),
    )
    
    if not matches:
        return f"No buyers found for {entities.get('commodity')} right now. Try again tomorrow."
    
    top = matches[0]
    return (
        f"Found {len(matches)} interested buyers. "
        f"Best match: {top.buyer_name} in {top.buyer_district}, "
        f"offering ₹{top.price_per_kg}/kg for {top.quantity_kg} kg. "
        f"Shall I connect you?"
    )
```

### 4. Language-Aware Response Templates
```python
RESPONSE_TEMPLATES = {
    'create_listing_success': {
        'en': "Your listing for {qty} kg of {crop} at ₹{price}/kg has been created. Listing ID: {id}",
        'kn': "ನಿಮ್ಮ {qty} ಕೆಜಿ {crop} ₹{price}/ಕೆಜಿ ಪಟ್ಟಿ ರಚಿಸಲಾಗಿದೆ. ಪಟ್ಟಿ ID: {id}",
        'hi': "आपकी {qty} किलो {crop} की लिस्टिंग ₹{price}/किलो पर बनाई गई है। लिस्टिंग ID: {id}",
    },
    'price_check': {
        'en': "{crop} is currently ₹{price}/kg in {market}. {recommendation}",
        'kn': "{crop} ಈಗ {market} ನಲ್ಲಿ ₹{price}/ಕೆಜಿ ಇದೆ. {recommendation}",
        'hi': "{crop} अभी {market} में ₹{price}/किलो है। {recommendation}",
    },
}
```

---

## ✅ Acceptance Criteria (9/10 Score)

| # | Criterion | Weight |
|---|-----------|--------|
| 1 | All 10+ intents route to real agents/services | 30% |
| 2 | Multi-turn conversation for create_listing works | 20% |
| 3 | Templates in 3+ languages (en, kn, hi) | 15% |
| 4 | Graceful fallback when service unavailable | 15% |
| 5 | Session context preserved across turns | 10% |
| 6 | Manual test: voice "tomato ka daam" returns real price | 10% |

---

## 🔄 Progress Update (2026-03-01)

### Completed in this increment
- Added multi-turn stateful `create_listing` flow in `src/agents/voice_agent.py`:
  - Collects missing fields across turns (`crop`, `quantity`, `asking_price`)
  - Preserves pending intent/context in session and completes listing when all fields are present
- Upgraded service-backed handlers:
  - `check_price` now uses `pricing_agent.get_recommendation(...)` and appends recommendation reasoning
  - `track_order` now attempts order-status methods before default fallback
- Extended extraction in `src/voice/entity_extractor.py`:
  - Added asking-price pattern extraction for listing completion
- Added tests:
  - `tests/unit/test_voice_agent.py` covers multi-turn listing flow and real service calls
  - Validation command: `uv run pytest tests/unit/test_voice_agent.py` → **3 passed**
- Added and validated ChatGPT-style static streaming UI:
  - `static/voice_realtime.html`, `static/assets/css/voice-realtime.css`, `static/assets/js/voice-realtime.js`
  - Live checks confirmed static page serving and SSE stream token delivery

### Remaining for full Task 4 closure
- Complete wiring for additional intents listed in spec (`find_buyer`, `quality_check`, `weekly_demand`, etc.)
- Expand multi-turn flows beyond create listing where required
- Add integration-level voice path validation for full STT → intent → service → TTS cycle
