# buyer-matching Agent Specification

## Purpose
Match farmer listings with buyers using multi-factor scoring and geospatial optimization, and support reverse buyer-to-farmer discovery for scheduled procurement.

## Inputs
- Listing-side inputs:
  - `listing_id`, `farmer_id`, `commodity`, `quantity_kg`, `asking_price_per_kg`, `grade`, pickup coordinates
- Buyer-side inputs:
  - `buyer_id`, buyer coordinates, preferred/min grade, max price, commodity demand, order history
- Matching controls:
  - `top_n` / `max_results`
  - `min_score`
  - reverse matching filters (`commodity`, `quantity_needed_kg`, `max_price_per_kg`)

## Outputs
- Ranked `MatchCandidate` list with transparent factors:
  - `match_score`
  - `proximity_km`
  - `quality_match`
  - `price_fit`
  - `demand_signal`
  - `reliability`
  - delivery/logistics estimates
- `MatchResult` metadata:
  - `total_candidates_evaluated`
  - `cache_hit`
  - timestamp

## Constraints
- Weighted score formula must stay consistent:
  - proximity 30%, quality 25%, price fit 20%, demand signal 15%, reliability 10%
- Grade constraint:
  - listing below buyer minimum grade must produce quality score `0.0`
- Distance constraint:
  - long-distance listings decay strongly by proximity score
- Caching:
  - target TTL = 5 minutes for recent repeated lookups

## Dependencies
- `src/agents/supervisor_agent.py` for matching intent routing.
- `src/api/routes/chat.py` and `src/api/routers/chat.py` for runtime agent registration.
- Redis (optional) for cache backend; local in-memory fallback is supported.
- Unit tests:
  - `tests/unit/test_buyer_matching.py`
  - `tests/unit/test_supervisor_routing.py`
