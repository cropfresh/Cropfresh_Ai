# price-prediction Agent Specification

## Purpose
Provide business-aligned farm-gate and buyer-side pricing intelligence for CropFresh:
- Calculate transparent AISP (All-Inclusive Sourcing Price) with logistics and risk components.
- Generate actionable sell/hold recommendations from current and historical mandi prices.
- Apply seasonal multipliers for key Karnataka crops.

## Inputs
- `commodity: str`
- `location/district: str`
- `quantity_kg: float`
- `farmer_price_per_kg: float`
- Optional pricing modifiers:
  - `distance_km: float`
  - `handling_per_kg: float`
  - `route_utilization_pct: float`
  - `cold_chain: bool`
  - `mandi_modal_per_kg: float`
- Trend analysis inputs:
  - `days: int` (default 30)
  - historical market prices from Agmarknet

## Outputs
- `AISPCalculation` breakdown:
  - farmer payout, logistics, deadhead surcharge, handling, platform fee, risk buffer
  - final `total_aisp`, `aisp_per_kg`, `mandi_cap_applied`
- Recommendation payload (`PriceRecommendation`) with:
  - current market prices, action (`sell`/`hold`), confidence, and reason
- Trend payload (`get_price_trend`) with:
  - `trend`, `volatility_index`, `7d_avg`, `30d_avg`, `recommendation`
- Seasonal multiplier (`get_seasonal_adjustment`) as `float`

## Constraints
- AISP must include fixed risk buffer of **2%**.
- Mandi cap rule: `AISP per kg <= mandi_modal_per_kg * 1.05` when mandi modal is available.
- Route utilization must be within `0–100`.
- Trend output quality depends on availability and quality of historical price data.
- Mock mode should provide deterministic fallback for local development and tests.

## Dependencies
- `src/tools/agmarknet.py` for current and historical mandi prices.
- `src/orchestrator/llm_provider.py` for optional LLM-assisted natural-language analysis.
- Unit tests in `tests/unit/test_pricing_agent.py`.
