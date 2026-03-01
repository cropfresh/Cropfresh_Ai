# F003: Price Prediction

## Overview
AI-powered crop price prediction using APMC data, weather, and historical trends.

## Acceptance Criteria
- [x] Real-time APMC price fetching (Agmarknet + mock fallback)
- [ ] 7-day price forecast
- [ ] Weather impact analysis
- [x] Historical trend analysis (7d/30d averages + volatility + recommendation)

## Priority: P0 | Status: In Progress

## Progress Notes (2026-03-01)
- Completed Task 1 DPLE baseline in `src/agents/pricing_agent.py`:
  - business-aligned AISP formula
  - utilization-based deadhead surcharge
  - fixed 2% risk buffer
  - mandi cap at `mandi_modal × 1.05`
  - seasonal multiplier method for major Karnataka crops
- Added historical price support in `src/tools/agmarknet.py` via `get_historical_prices()`.
- Unit validation: `uv run pytest tests/unit/test_pricing_agent.py` → 11 passing.
