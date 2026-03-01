# Task 19: Integration Tests — End-to-End Flows

> **Priority:** 🟠 P1 | **Phase:** 5 | **Effort:** 3–4 days  
> **Files:** `tests/e2e/` (3 new test files)  
> **Score Target:** 9/10 — Full lifecycle tests proving business flows work

---

## 📌 Problem Statement

No end-to-end tests validate the complete business flows. Need to verify: register → list → match → order → deliver → settle.

---

## 🏗️ Implementation Spec

### Test Flows

#### Flow 1: Farmer Listing Flow (`test_listing_flow.py`)
```python
async def test_farmer_listing_e2e():
    """
    Complete listing flow:
    1. Register farmer (phone + OTP)
    2. Create listing (tomato, 500kg, ₹20/kg)
    3. Auto-price suggestion applied
    4. Quality assessment triggered (mock)
    5. Listing appears in search results
    6. Matched with buyer
    """
    # Step 1: Register
    farmer = await registration_service.register("+919876543210", "farmer")
    assert farmer.id
    
    # Step 2: Create listing
    listing = await listing_service.create_listing({
        "farmer_id": farmer.id,
        "commodity": "Tomato",
        "quantity_kg": 500,
    })
    assert listing.asking_price_per_kg > 0  # Auto-suggested
    assert listing.status == "active"
    
    # Step 3: Verify searchable
    results = await listing_service.search_listings(commodity="Tomato")
    assert any(r.id == listing.id for r in results)
    
    # Step 4: Match with buyer
    matches = await matching_engine.find_matches(listing.id)
    assert len(matches) >= 1
```

#### Flow 2: Order Flow (`test_order_flow.py`)
```python
async def test_order_lifecycle_e2e():
    """
    Complete order flow:
    1. Create order from matched listing + buyer
    2. AISP calculated with full breakdown
    3. Status: confirmed → pickup_scheduled → in_transit → delivered → settled
    4. Escrow: pending → held → released
    5. Farmer payout calculated
    """
```

#### Flow 3: Voice Flow (`test_voice_flow.py`)
```python
async def test_voice_price_check_e2e():
    """
    Voice round-trip:
    1. Audio input (mock WAV) → STT
    2. Transcription → entity extraction
    3. Intent detected: check_price
    4. PricingAgent called → real price returned
    5. Response → TTS → audio output
    6. Verify audio bytes > 0
    """
```

### Test Infrastructure
```python
# conftest.py additions
@pytest.fixture
async def test_db():
    """In-memory SQLite or test PostgreSQL with rollback."""
    async with create_test_db() as db:
        yield db

@pytest.fixture
def mock_agents():
    """Pre-configured mock agents for E2E testing."""
    return {
        'pricing': PricingAgent(use_mock=True),
        'matching': MatchingEngine(db=mock_db),
        'listing': ListingService(db=mock_db),
        'order': OrderService(db=mock_db),
    }
```

---

## ✅ Acceptance Criteria

| # | Criterion | Weight |
|---|-----------|--------|
| 1 | Listing flow E2E passes (register → list → search → match) | 30% |
| 2 | Order flow E2E passes (match → order → deliver → settle) | 30% |
| 3 | Voice flow E2E passes (audio → STT → agent → TTS → audio) | 25% |
| 4 | Tests use proper fixtures and test DB | 15% |
