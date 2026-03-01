# Sprint 05 — Core Agent Completion
> **Dates:** 2026-03-01 to 2026-03-14 (2 weeks)
> **Theme:** Bring all 5 core AI agents from stub/partial → working implementation
> **Sprint Goal:** Zero `NotImplementedError` in any business-critical agent path

---

## 🎯 Sprint Context

Following the complete business model analysis (PDF: "India's AI-Powered Agri-Intelligence Marketplace"), we identified that:
1. The Matchmaking Engine is a skeleton stub
2. The Pricing Agent (DPLE) is missing deadhead factor, risk buffer (2%), and mandi cap check
3. The Voice Agent has `# TODO` stubs on all key intents (listing, price check, order tracking)
4. The Quality Assessment Agent raises `NotImplementedError`
5. The Buyer Matching Agent delegates to a non-existent matchmaking module

This sprint closes all Phase 1 gaps.

---

## ✅ Sprint Tasks

### P0 — Must Complete

| # | Task | File | Owner | Done? |
|---|------|------|-------|-------|
| 1 | Fix `calculate_aisp()` — utilization-based deadhead, risk buffer(2%), mandi cap (`×1.05`) | `src/agents/pricing_agent.py` | AI | [x] |
| 2 | Implement buyer matching engine — multi-factor score + reverse matching + cache | `src/agents/buyer_matching/agent.py` | AI | [x] |
| 3 | Implement `QualityAssessmentAgent` — HITL trigger, grade workflow, `GradeResult` model | `src/agents/quality_assessment/agent.py` | AI | [x] |
| 4 | Wire `VoiceAgent` TODOs — `create_listing`, `check_price`, `track_order`, `my_listings` → real agents | `src/agents/voice_agent.py` | AI | [ ] |
| 5 | Wire `BuyerMatchingAgent` into supervisor/chat bootstrap | `src/agents/supervisor_agent.py`, `src/api/routes/chat.py` | AI | [x] |

### P1 — High Priority

| # | Task | File | Owner | Done? |
|---|------|------|-------|-------|
| 6 | Write unit tests for `pricing_agent.py` (deadhead, risk buffer, mandi cap, AISP tiers, trend/seasonality) | `tests/unit/test_pricing_agent.py` | AI | [x] |
| 7 | Write unit tests for buyer matching (`matching + reverse + routing`) | `tests/unit/test_buyer_matching.py`, `tests/unit/test_supervisor_routing.py` | AI | [x] |
| 8 | Write unit tests for `quality_assessment/agent.py` (HITL trigger, grade downgrade logic) | `tests/unit/test_quality_assessment.py` | AI | [x] |
| 9 | Create `config/supabase_schema.sql` — farmers, listings, orders, disputes, haulers, buyers | `config/supabase_schema.sql` | AI | [ ] |
| 10 | Add `GET_AISP`, `CROP_ADVISORY` intents to Voice Agent | `src/agents/voice_agent.py` | AI | [ ] |

### P2 — Nice to Have

| # | Task | File | Done? |
|---|------|-------|-------|
| 11 | Punjabi (`pa`) + Telugu (`te`) response templates in VoiceAgent | `src/agents/voice_agent.py` | [ ] |
| 12 | Register eNAM API access (enam.gov.in/register) | external | [ ] |
| 13 | Test Pipecat WebSocket with live audio on Windows | `src/voice/pipecat_bot.py` | [ ] |

---

## 📐 Key Designs This Sprint

### AISP Formula (Corrected)
```
AISP = Farmer_Payout
     + Logistics
     + Deadhead_Surcharge (utilization-based)
     + Handling
     + Platform_Fee (5–8% dynamic tiered)
     + Risk_Buffer (2% of subtotal)

ALSO: AISP must NOT exceed (Mandi_Modal × 1.05)
```

### Matchmaking Algorithm (v1)
```python
# Step 1: Score each listing-buyer pair with weighted factors
score = (
    0.30 * proximity +
    0.25 * quality_match +
    0.20 * price_fit +
    0.15 * demand_signal +
    0.10 * reliability
)

# Step 2: Rank descending and return top candidates
matches = sorted(candidates, key=lambda x: x.match_score, reverse=True)

# Step 3: Cache result for 5 min (redis or local fallback)
cache.setex(key, 300, serialized_matches)

# Step 4: Reverse matching supports buyer->farmer lookup
buyer_matches = find_farmers_for_buyer(...)
```

### Quality Assessment HITL Logic
```
AI confidence ≥ 0.7 with non-premium grade → auto-accept
AI confidence < 0.7 → HITL_required = True
Grade A+ → Always HITL_required = True (premium verification policy)
Farmer grade-upgrade request → HITL_required = True
```

---

## 🏁 Sprint Definition of Done

- [ ] `uv run pytest tests/unit/ -v` passes with ≥ 5 new test files
- [x] No `NotImplementedError` in `src/agents/` core paths
- [ ] Voice `create_listing` intent returns a confirmation with listing ID stub
- [x] `calculate_aisp(farmer_price=13.5, qty=1000, distance=60)` returns `total_aisp` that includes risk buffer
- [x] Matchmaking test: valid ranked match candidates returned and reverse matching works
- [ ] Test coverage: ≥ 45% (up from 35%)

---

## 📖 Sprint Outcome (fill at end)

> _To be completed by 2026-03-14_

### What Shipped
-

### What Slipped
-

### Key Learnings
-

### Metrics After Sprint
| Metric | Before | After |
|--------|--------|-------|
| Test coverage | 35% | — |
| Voice agent TODO stubs | ~8 | — |
| Agents with NotImplementedError | 3 | — |
