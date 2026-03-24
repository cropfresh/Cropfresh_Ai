# Sprint 06 - ADCL Productionization & Full Hardening

> **Period:** 2026-03-25 -> 2026-04-07
> **Theme:** Make ADCL development-complete as a district-first live service with persistence, app wiring, evaluation, and operational hardening
> **Sprint Status:** In Progress (implementation started early on 2026-03-17)

---

## Goals (Measurable)

1. One canonical `ADCLService` contract is reused by REST, wrapper, listings, and voice.
2. Weekly district reports are generated from live marketplace data plus approved external sources and persisted once per `(week_start, district)`.
3. `GET /api/v1/adcl/weekly` returns the canonical ADCL payload with evidence, freshness, and source-health metadata.
4. No production ADCL runtime path uses hardcoded mock orders or mock live-source clients.
5. A backtest over the last 8 historical weeks and a 20-query golden set are produced and linked from sprint artifacts.

---

## Scope (Stories / Tasks)

### Canonical Service Contract
- [x] `src/agents/adcl/service.py` (or equivalent) - Introduce `ADCLService` with:
  - `generate_weekly_report(district, force_live=False, farmer_id=None, language=None)`
  - `get_weekly_demand(district, force_live=False)`
  - `is_recommended_crop(commodity, district, week_start=None)`
- [x] Standardize one payload across all callers:
  - `commodity`
  - `green_label`
  - `recommendation`
  - `demand_trend`
  - `price_trend`
  - `seasonal_fit`
  - `sow_season_fit`
  - `buyer_count`
  - `total_demand_kg`
  - `predicted_price_per_kg`
  - `evidence`
  - `freshness`
  - `source_health`
- [x] Keep thin compatibility shims only where needed so wrapper, listings, and voice can migrate without diverging.
- [x] Tests: unit coverage for the canonical contract and adapter compatibility.

### Database and Persistence
- [x] `src/db/postgres/client.py` - Add `get_recent_orders(district, days)`, `insert_adcl_report(report)`, and `get_latest_adcl_report(district)`.
- [x] Add migration to change `adcl_reports` uniqueness from `week_start` to `(week_start, district)`.
- [x] Add `district`, `source_health`, `metadata`, and freshness fields to `adcl_reports`.
- [x] Remove runtime mock-order fallback from ADCL production paths; keep mocks test-only.
- [ ] Tests: integration coverage for orders, price history, and `adcl_reports` persistence in a real DB environment.

### Live Data and Ranking Inputs
- [x] Use marketplace internals as the primary demand truth: confirmed orders, buyer counts, repeat buyers, cancellations, listing velocity, and district activity.
- [x] Reuse the shared official-first rate hub for price context instead of creating a parallel price-fetch path.
- [x] Use IMD/Agromet live weather and advisory inputs with mocks disabled in production paths.
- [x] Wire eNAM as a gated source behind credentials or a feature flag and expose its health and freshness in ADCL output.
- [ ] Keep Google AMED and farmer-specific satellite personalization out of this sprint.

### API, App, and Caller Wiring
- [x] Add `GET /api/v1/adcl/weekly?district=...&force_live=...` returning the canonical weekly report.
- [x] Expose `db`, `adcl_service`, `listing_service`, and shared voice services on app startup so routers do not instantiate partial dependencies.
- [x] Update listing enrichment to call `is_recommended_crop(...)`.
- [x] Update voice weekly-demand flow and ADCL wrapper formatting to use canonical fields instead of ad hoc `crop` / `label` / `summary` names.
- [x] Tests: API coverage for `/api/v1/adcl/weekly`, listing badge flow, and voice weekly-demand flow.

### Hardening and Evaluation
- [x] Add APScheduler jobs for weekly district report generation and daily source-health refresh.
- [x] Add structured logs and metrics for generation count, latency, source failures, freshness lag, and recommendation coverage.
- [ ] Produce a backtest or evaluation artifact for the last 8 historical weeks.
- [x] Produce a 20-query golden set for user-facing ADCL behavior.

### Documentation and Handoff
- [x] Update `docs/api/overview.md` and `docs/api/endpoints-reference.md` once the ADCL endpoint ships.
- [x] Update `tracking/PROJECT_STATUS.md`, `WORKFLOW_STATUS.md`, and daily logs as implementation lands.
- [ ] Record any scope changes or slips in this sprint file instead of rewriting history.

---

## Acceptance / Done Criteria

- [ ] Weekly report generates from live DB plus live sources and persists once per `(week_start, district)`.
- [x] Listing creation marks an in-demand crop from the current weekly report.
- [x] Voice weekly-demand for a district returns the same canonical ADCL contract used by the REST endpoint.
- [ ] Source failures degrade evidence and source health, but the service still returns a usable report from remaining sources.
- [x] `force_live=true` bypasses cache and exposes fresh source timestamps.
- [ ] Historical backtest and golden-set artifacts are generated and linked from sprint notes.
- [x] No production ADCL path uses hardcoded mock orders or mock live-source clients.

---

## Out of Scope

- No Google AMED integration in this sprint.
- No farmer-specific satellite personalization in this sprint.
- No new farmer agronomy-profile schema.
- No broad marketplace refactor outside the ADCL integration surfaces.
- No mobile redesign beyond exposing ADCL through existing app surfaces.

---

## Risks / Open Questions

- eNAM credentials or API access may still be pending when connector work starts.
- IMD or advisory feeds may require normalization work before they are useful in ranking.
- Some districts may have sparse recent order history; fallback behavior must remain explicit and evidence-backed.
- Source outages must degrade gracefully without silently switching to mocks.

---

## Assumptions

- Sprint scope is district-first, with only light farmer-aware overlays from existing profile fields.
- Official-plus-gated source policy is required for this sprint.
- Live-source surfaces were verified for planning on 2026-03-17:
  - `https://www.data.gov.in/`
  - `https://enam.gov.in/web/`
  - `https://agromet.imd.gov.in/`
  - `https://mausam.imd.gov.in/responsive/agricultureweather.php`
  - `https://mausam.imd.gov.in/Bengaluru/data_request.php`

---

## Sprint Outcome (fill at end of sprint)

**What Shipped:**
- Canonical `ADCLService` plus shared factory, compatibility shims, and district-scoped weekly report contract
- Aurora ADCL repo methods, composite-key migration, and listing persistence updates
- `GET /api/v1/adcl/weekly`, shared app-state wiring, voice/listings integration, and APScheduler jobs
- ADCL evaluation starter assets: golden set and live backtest runbook

**What Slipped to Next Sprint:**
- Live 8-week backtest execution against a real Aurora district snapshot
- Production verification of gated eNAM credentials once access is available

**Key Learnings:**
- ADCL works best as a shared service contract, not as separate wrapper-specific payloads
- Reusing the shared rate hub removed a parallel live-price path and reduced drift risk
- Startup wiring needed to be centralized so REST, listings, and voice stop creating partial agents

**Agent Eval Scores This Sprint:**

| Agent | Precision@3 | Sell-Through Lift | Freshness SLA | Notes |
|-------|-------------|-------------------|---------------|-------|
| ADCL | | | | |

---

## Related Files

- `tracking/PROJECT_STATUS.md`
- `ROADMAP.md`
- `tracking/tasks/backlog.md`
- `docs/decisions/ADR-012-adcl-district-first-service-contract.md`
- `docs/decisions/ADR-013-adcl-source-precedence-and-evidence.md`
- `src/agents/adcl/`
- `src/api/main.py`
- `src/api/routers/listings.py`
- `src/api/rest/voice.py`
- `src/db/postgres/client.py`
- `tests/unit/`
- `tests/integration/`
- `src/evaluation/datasets/adcl_golden_queries.json`
- `src/evaluation/reports/adcl_backtest_2026-03-17.md`
