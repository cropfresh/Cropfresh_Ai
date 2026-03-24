# ADCL Backtest Artifact - 2026-03-17

## Purpose

This artifact captures the execution-ready backtest plan for the district-first ADCL service that landed on 2026-03-17. The canonical service, persistence, REST route, listing enrichment, and voice integration are now implemented. A live 8-week score run still requires a district order snapshot and production-like source credentials.

## Implemented Evaluation Inputs

- Canonical weekly report contract with district, crops, freshness, source health, and evidence
- `force_live` path for bypassing cached reports
- Composite persistence key `(week_start, district)` for repeatable weekly report storage
- 20-query golden set at `src/evaluation/datasets/adcl_golden_queries.json`
- ADCL-focused unit and API verification

## Live Backtest Runbook

1. Apply `src/db/migrations/004_adcl_reports_district_persistence.sql`.
2. Load at least 8 weeks of district-tagged `orders`, `listings`, `buyers`, `farmers`, and `price_history`.
3. Enable live connectors:
   - rate hub
   - IMD client
   - eNAM credentials with `ENABLE_ENAM=true`
4. Generate reports per district and week boundary with `force_live=true`.
5. Join the generated report crops back to realized sell-through and order outcomes.
6. Record the metrics below into this artifact and the sprint file.

## Metrics To Record

| Metric | Definition | Status |
|--------|------------|--------|
| Precision@3 | Top-3 recommended crops that realized strong demand | Pending live run |
| Recommendation coverage | `green_count / crop_count` per district-week | Available in report metadata |
| Freshness SLA | Age of source timestamps at report generation | Available in report freshness |
| Bad recommendation rate | Green-labelled crops with weak realized demand | Pending live run |
| Source degradation rate | Share of reports with degraded or gated sources | Available in source health |

## Current Verification Snapshot

- `uv run pytest tests/unit/test_adcl_agent.py tests/api/test_adcl_routes.py tests/unit/test_voice_agent.py tests/unit/test_listing_service.py tests/unit/test_api_config.py -q`
- Result on 2026-03-17: `79 passed`
- `uv run python -c "from src.api.main import app; print(app.title)"` succeeded after config parsing hardening

## Known Gaps

- No real district order snapshot is bundled in the repo, so historical realized-demand scoring was not executed here.
- eNAM stays gated until credentials are available.
- Final production sign-off still needs one run against the target Aurora environment.
