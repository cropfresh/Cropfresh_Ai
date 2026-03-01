# F005: Buyer Matching

## Overview
Graph-based buyer-seller matching using Neo4j.

## Acceptance Criteria
- [x] Match based on crop type, quantity, location, grade
- [x] Relevance scoring (multi-factor weighted score)
- [ ] Notification to matched buyers

## Priority: P1 | Status: In Progress

## Progress Notes (2026-03-01)
- Completed Task 2 matching core in `src/agents/buyer_matching/agent.py`:
  - 5-factor weighted scoring
  - haversine proximity scoring
  - quality and price fit enforcement
  - demand signal + reliability factors
  - reverse matching (`buyer -> farmers`)
  - 5-minute cache (redis + local fallback)
- Added supervisor intent routing support in `src/agents/supervisor_agent.py`.
- Registered buyer matching agent in chat bootstrap (`src/api/routes/chat.py`, `src/api/routers/chat.py`).
- Validation:
  - `uv run pytest tests/unit/test_buyer_matching.py tests/unit/test_supervisor_routing.py`
  - **28 passed**
