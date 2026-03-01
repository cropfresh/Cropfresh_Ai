# price-prediction Agent — Changelog

## [Unreleased]
- Completed Task 1 (DPLE pricing baseline) in `src/agents/pricing_agent.py`:
  - AISP formula aligned to business model
  - utilization-based deadhead surcharge
  - fixed 2% risk buffer
  - mandi cap enforced at `mandi_modal × 1.05`
- Added trend and seasonality support:
  - `get_price_trend()` with 7/30-day averages, volatility, and recommendation
  - `get_seasonal_adjustment()` for major Karnataka crops
- Added historical price fetch support in `src/tools/agmarknet.py`:
  - `get_historical_prices()` + mock-history fallback
- Added/updated unit tests in `tests/unit/test_pricing_agent.py`:
  - `uv run pytest tests/unit/test_pricing_agent.py` → 11 passing
