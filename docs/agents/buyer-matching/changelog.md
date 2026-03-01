# buyer-matching Agent — Changelog

## [Unreleased]
- Completed Task 2 Buyer Matching core implementation:
  - Added `MatchingEngine` with 5 weighted signals:
    - proximity (30%), quality (25%), price fit (20%), demand signal (15%), reliability (10%)
  - Added haversine-based proximity scoring with non-linear decay.
  - Added quality-grade enforcement (`below minimum => 0.0` quality score).
  - Added reverse matching flow (`find_farmers_for_buyer`).
  - Added caching for match results (redis + local fallback, TTL 5 minutes).
- Integrated with routing and chat bootstrap:
  - `supervisor_agent.py` intent updates
  - `api/routes/chat.py` + `api/routers/chat.py` registration
- Added and updated unit tests:
  - `tests/unit/test_buyer_matching.py`
  - `tests/unit/test_supervisor_routing.py`
  - Validation run: `uv run pytest tests/unit/test_buyer_matching.py tests/unit/test_supervisor_routing.py` → 28 passed
