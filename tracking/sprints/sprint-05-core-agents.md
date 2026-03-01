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
| 1 | Fix `calculate_aisp()` — add deadhead(10%), risk buffer(2%), mandi cap check | `src/agents/pricing_agent.py` | AI | [ ] |
| 2 | Implement `MatchmakingAgent` — GPS clustering + buyer preference matrix + margin optimization | `src/agents/matchmaking_agent.py` | AI | [ ] |
| 3 | Implement `QualityAssessmentAgent` — HITL trigger, grade workflow, `GradeResult` model | `src/agents/quality_assessment/agent.py` | AI | [ ] |
| 4 | Wire `VoiceAgent` TODOs — `create_listing`, `check_price`, `track_order`, `my_listings` → real agents | `src/agents/voice_agent.py` | AI | [ ] |
| 5 | Wire `BuyerMatchingAgent` → `MatchmakingAgent` | `src/agents/buyer_matching/agent.py` | AI | [ ] |

### P1 — High Priority

| # | Task | File | Owner | Done? |
|---|------|------|-------|-------|
| 6 | Write unit tests for `pricing_agent.py` (deadhead, risk buffer, mandi cap, AISP tiers) | `tests/unit/test_pricing_agent.py` | AI | [ ] |
| 7 | Write unit tests for `matchmaking_agent.py` (clustering, margin ranking, vehicle selection) | `tests/unit/test_matchmaking_agent.py` | AI | [ ] |
| 8 | Write unit tests for `quality_assessment/agent.py` (HITL trigger, grade downgrade logic) | `tests/unit/test_quality_assessment.py` | AI | [ ] |
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
AISP = Farmer_Ask
     + Logistics (DPLE rate × qty × deadhead_factor=1.10)
     + Platform_Margin (4–8% dynamic tiered)
     + Risk_Buffer (2% of subtotal)

ALSO: AISP must NOT exceed Mandi_Landed_Price (cap check)
```

### Matchmaking Algorithm (v1)
```python
# Step 1: Cluster nearby farmers (radius < 5km)
farmer_clusters = cluster_by_gps(listings, radius_km=5)

# Step 2: Score each cluster against each buyer demand
scores = score_matches(clusters, buyer_demands, weights={
    "grade_match": 0.4,
    "price_discount_vs_mandi": 0.3,
    "logistics_efficiency": 0.2,
    "freshness_hours": 0.1
})

# Step 3: Return top matches ranked by platform margin
matches = sorted(scores, key=lambda x: x.platform_margin, reverse=True)
```

### Quality Assessment HITL Logic
```
AI confidence ≥ 95% → Auto Grade → Digital Twin stub
AI confidence < 95% → HITL_required = True
Farmer claims Grade A → Always HITL_required = True (first 3 months)
Agent overrides AI → Ground truth label → Weekly retraining
```

---

## 🏁 Sprint Definition of Done

- [ ] `uv run pytest tests/unit/ -v` passes with ≥ 5 new test files
- [ ] No `NotImplementedError` in `src/agents/` core paths
- [ ] Voice `create_listing` intent returns a confirmation with listing ID stub
- [ ] `calculate_aisp(farmer_price=13.5, qty=1000, distance=60)` returns `total_aisp` that includes risk buffer
- [ ] Matchmaking test: given 2 farmers + 1 buyer → returns 1 valid match
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
