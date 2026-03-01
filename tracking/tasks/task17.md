# Task 17: Unit Tests for All Agents (Coverage 35% → 60%)

> **Priority:** 🟠 P1 | **Phase:** 5 | **Effort:** 4–5 days  
> **Files:** `tests/unit/` (8 new test files)  
> **Score Target:** 9/10 — Comprehensive tests with edge cases and mocking

---

## 📌 Problem Statement

Only 6 test files exist in `tests/unit/`. Many new agents have zero tests. Need to increase coverage from ~35% to 60% with meaningful test cases.

---

## 🏗️ Implementation Spec

### Test Files to Create
| Test File | Agent Under Test | Test Count |
|-----------|-----------------|------------|
| `test_matchmaking_agent.py` | BuyerMatchingAgent | 10+ |
| `test_crop_listing_agent.py` | CropListingAgent + ListingService | 12+ |
| `test_price_prediction_agent.py` | PricePredictionAgent | 8+ |
| `test_digital_twin.py` | DigitalTwinEngine | 8+ |
| `test_logistics_router.py` | LogisticsRouter | 10+ |
| `test_adcl_agent.py` | ADCLAgent | 6+ |
| `test_order_service.py` | OrderService | 12+ |
| `test_listing_service.py` | ListingService | 10+ |

### Testing Patterns
```python
# 1. Use fixtures for common setup
@pytest.fixture
def pricing_agent():
    return PricingAgent(use_mock=True)

# 2. Mock external dependencies
@patch('src.db.postgres_client.AuroraPostgresClient')
async def test_listing_creation(mock_db):
    mock_db.create_listing.return_value = "uuid-123"
    service = ListingService(db=mock_db)
    result = await service.create_listing(valid_data)
    assert result.id == "uuid-123"

# 3. Test edge cases
def test_aisp_zero_quantity_raises():
    agent = PricingAgent()
    with pytest.raises(ValueError):
        agent.calculate_aisp(farmer_price=20, quantity_kg=0)

# 4. Parametrize for coverage
@pytest.mark.parametrize("grade,min_grade,expected", [
    ('A+', 'A', 0.9),
    ('A', 'A', 1.0),
    ('B', 'A', 0.0),
    ('C', 'C', 1.0),
])
def test_quality_match_scoring(grade, min_grade, expected):
    engine = MatchingEngine()
    assert engine.calculate_quality_match(grade, min_grade) == expected

# 5. Test state machines
@pytest.mark.parametrize("from_status,to_status,valid", [
    ('confirmed', 'pickup_scheduled', True),
    ('confirmed', 'delivered', False),  # Invalid transition
    ('in_transit', 'disputed', True),
    ('settled', 'confirmed', False),
])
def test_order_state_transitions(from_status, to_status, valid):
    ...
```

### Coverage Targets
```
src/agents/pricing_agent.py        → 90%
src/agents/buyer_matching/agent.py → 85%
src/agents/quality_assessment/     → 80%
src/agents/voice_agent.py          → 70%
src/api/services/                  → 85%
src/agents/logistics_router.py     → 80%
Overall project                    → 60%+
```

---

## ✅ Acceptance Criteria

| # | Criterion | Weight |
|---|-----------|--------|
| 1 | 8 new test files with 70+ total test cases | 30% |
| 2 | All tests pass: `uv run pytest tests/unit/ -v` | 25% |
| 3 | Edge cases tested (zero quantity, invalid transitions, etc.) | 20% |
| 4 | Mocking used for DB and external API calls | 15% |
| 5 | Coverage report shows ≥60% overall | 10% |
