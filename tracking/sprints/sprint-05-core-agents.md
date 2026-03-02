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

| #   | Task                                                                                            | File                                                       | Owner | Done? |
| --- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | ----- | ----- |
| 1   | Fix `calculate_aisp()` — utilization-based deadhead, risk buffer(2%), mandi cap (`×1.05`)       | `src/agents/pricing_agent.py`                              | AI    | [x]   |
| 2   | Implement buyer matching engine — multi-factor score + reverse matching + cache                 | `src/agents/buyer_matching/agent.py`                       | AI    | [x]   |
| 3   | Implement `QualityAssessmentAgent` — HITL trigger, grade workflow, `GradeResult` model          | `src/agents/quality_assessment/agent.py`                   | AI    | [x]   |
| 4   | Wire `VoiceAgent` TODOs — all 10+ intents → real agents, multi-turn flows, 3-language templates | `src/agents/voice_agent.py`                                | AI    | [x]   |
| 5   | Wire `BuyerMatchingAgent` into supervisor/chat bootstrap                                        | `src/agents/supervisor_agent.py`, `src/api/routes/chat.py` | AI    | [x]   |

### P1 — High Priority

| #   | Task                                                                                                                | File                                                                         | Owner | Done? |
| --- | ------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ----- | ----- |
| 6   | Write unit tests for `pricing_agent.py` (deadhead, risk buffer, mandi cap, AISP tiers, trend/seasonality)           | `tests/unit/test_pricing_agent.py`                                           | AI    | [x]   |
| 7   | Write unit tests for buyer matching (`matching + reverse + routing`)                                                | `tests/unit/test_buyer_matching.py`, `tests/unit/test_supervisor_routing.py` | AI    | [x]   |
| 8   | Write unit tests for `quality_assessment/agent.py` (HITL trigger, grade downgrade logic)                            | `tests/unit/test_quality_assessment.py`                                      | AI    | [x]   |
| 9   | Extend DB schema — 10 business tables, PostGIS GIST indexes, migration runner, 13 CRUD methods                      | `src/db/migrations/`, `src/db/postgres_client.py`                            | AI    | [x]   |
| 10  | Add 7 new intents to Voice Agent (find_buyer, weather, advisory, register, dispute, quality, weekly_demand)         | `src/agents/voice_agent.py`                                                  | AI    | [x]   |
| 11  | Fix & implement `PricePredictionAgent` — rule-based, trend, seasonal calendar, recommendation, LLM fallback         | `src/agents/price_prediction/agent.py`                                       | AI    | [x]   |
| 12  | Build Order Management Service — state machine, escrow, AISP, dispute + Digital Twin diff trigger, 8 REST endpoints | `src/api/services/order_service.py`, `src/api/routers/orders.py`             | AI    | [x]   |
| 13  | Build Registration & Auth Service — OTP flow, stdlib JWT, farmer/buyer profiles, voice compat, 6 REST endpoints     | `src/api/services/registration_service.py`, `src/api/routers/auth.py`        | AI    | [x]   |

### P2 — Nice to Have

| #   | Task                                                            | File                        | Done? |
| --- | --------------------------------------------------------------- | --------------------------- | ----- |
| 11  | Punjabi (`pa`) + Telugu (`te`) response templates in VoiceAgent | `src/agents/voice_agent.py` | [ ]   |
| 12  | Register eNAM API access (enam.gov.in/register)                 | external                    | [ ]   |
| 13  | Test Pipecat WebSocket with live audio on Windows               | `src/voice/pipecat_bot.py`  | [ ]   |

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

- [x] `uv run pytest tests/unit/ -v` passes with ≥ 5 new test files (**382 tests across 15 test files**)
- [x] No `NotImplementedError` in `src/agents/` core paths
- [x] Voice `create_listing` intent creates real DB record (AC6 of Task 7)
- [x] `calculate_aisp(farmer_price=13.5, qty=1000, distance=60)` returns `total_aisp` that includes risk buffer
- [x] Matchmaking test: valid ranked match candidates returned and reverse matching works
- [x] Test coverage: ≥ 45% (**~57%** — 382 tests across 15 test files, up from 35%)
- [x] Order lifecycle: `confirmed → pickup_scheduled → in_transit → delivered → settled` enforced in state machine
- [x] Escrow: `held → released` on settlement; `held → refunded` on cancellation/dispute
- [x] Registration: OTP → JWT → ProfileResponse round-trip working, phone normalised to `+91XXXXXXXXXX`

---

## 📖 Sprint Outcome

> **Sprint Status: 🟢 Major milestones completed — 2026-03-01**

### What Shipped

- **Task 1** — Pricing Agent DPLE: deadhead utilization, 2% risk buffer, mandi cap, trend + seasonality
- **Task 2** — Buyer Matching Engine: 5-factor scoring, reverse matching, 5-min cache, supervisor wiring
- **Task 3** — Quality Assessment Agent: HITL threshold policy, grade pipeline, digital twin linkage
- **Task 4** — Voice Agent full wiring: 10+ intents, multi-turn flows, 3-language templates
- **Task 5** — Price Prediction Agent: rule-based + numpy trend + Karnataka seasonal calendar + LLM fallback
- **Task 6** — Database Schema Extension: 10 business tables, PostGIS, migration runner, 13 CRUD methods
- **Task 7** — Crop Listing Service: auto-price, shelf-life expiry, QR code, 7 REST endpoints, NL agent
- **Task 8** — Order Management Service: 11-status state machine, escrow flow (held→released/refunded), AISP ratio breakdown (80/10/6/4%), Digital Twin dispute diff trigger, 4 new DB CRUD methods, 8 REST endpoints, 73 tests
- **Task 9** — Registration & Auth Service: OTP flow (in-memory, 10-min expiry), stdlib HS256 JWT (no pyjwt), 31-district Karnataka language map, GPS→district centroid lookup, Aadhaar SHA-256 hashing, voice `register_farmer()` compat, 5 DB methods, 6 REST endpoints, 64 tests
- **Task 10** — Digital Twin Engine: `src/agents/digital_twin/` — create_departure_twin(), compare_arrival(), generate_diff_report(); SSIM → perceptual hash → rule-based similarity; 6-rule liability matrix; QualityAssessmentAgent.compare_twin() + create_departure_twin(); OrderService.\_trigger_twin_diff() dual-path; get_digital_twin() + update_dispute_diff_report() in postgres_client; 42 tests
- **Task 11** — DPLE Logistics Routing Engine: `src/agents/logistics_router/` package — HDBSCAN farm clustering (haversine); OR-Tools TSP + greedy fallback; 4-vehicle model (2W EV/3W Auto/Tempo/Cold Chain, full cost model); deadhead return factor; `cost_per_kg < ₹2.5` proven for 5-farm 30km cluster; 17 unit tests pass
- **Static UI** — ChatGPT-style streaming chat surface, buyer matching + quality check dashboard cards

### What Slipped

- Punjabi/Telugu voice templates (P2 — nice to have)
- eNAM API registration (external dependency)
- Pipecat WebSocket live audio test (Windows env constraint)

### Key Learnings

- Circular FK (listings ↔ digital_twins) requires ALTER TABLE post-creation pattern
- Seasonal multiplier must be isolated in tests to validate base prediction accuracy
- Intent priority in NL parsing matters — "cancel my listing" must check `cancel` before `my listing`

### Metrics After Sprint

| Metric                          | Before Sprint 05  | After Sprint 05                              |
| ------------------------------- | ----------------- | -------------------------------------------- |
| Test count                      | ~50               | **399**                                      |
| Test files                      | 5                 | **16**                                       |
| Voice agent TODO stubs          | ~8                | **0**                                        |
| Agents with NotImplementedError | 3                 | **0**                                        |
| DB tables (business)            | 4                 | **14**                                       |
| REST endpoints (listings)       | 0                 | **7**                                        |
| REST endpoints (orders)         | 0                 | **8**                                        |
| Total REST endpoints            | 0                 | **15**                                       |
| CRUD methods (postgres_client)  | 5                 | **22**                                       |
| Logistics cost/kg               | — (unimplemented) | **₹1.33/kg** (3-farm cluster, 3W Auto, 30km) |
| DPLE routing engine             | stub              | **✅ complete** (17 tests)                   |
