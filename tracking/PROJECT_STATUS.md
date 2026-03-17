# CropFresh AI - Project Status

> **Last Updated:** 2026-03-17
> **Phase:** Phase 2 - Business Services
> **Sprint:** Sprint 05 (current)
> **Next Sprint:** Sprint 06 - ADCL Productionization & Full Hardening (implementation started early)

---

## Component Status

| Component | Status | Progress | Sprint |
|-----------|--------|----------|--------|
| Project Structure | Complete | 100% | Sprint 01 |
| Multi-Agent System (15 agents) | Complete | 100% | Sprint 01 |
| RAG Pipeline (RAPTOR + Hybrid) | Complete | 100% | Sprint 01 |
| Memory System (Redis + in-memory) | Complete | 100% | Sprint 01 |
| Voice Agent v1 (10 languages) | Complete | 95% | Sprint 01-04 |
| VoiceAgent Multi-turn Flows | Complete | 90% | Sprint 04 |
| Tool Registry + Shared Tools | Complete | 100% | Sprint 01-05 |
| Agmarknet APMC Scraper | Complete | 100% | Sprint 04 |
| WebSocket Voice Streaming | Complete | 85% | Sprint 04 |
| Pipecat Voice Pipeline | In Progress | 60% | Sprint 04 |
| Documentation System | In Progress | 90% | Sprint 05 |
| Multi-Source Rate Hub | In Progress | 85% | Sprint 05 |
| AgriEmbeddingWrapper (L1) | Not Started | 0% | Sprint 05 |
| Agentic RAG Orchestrator | In Progress | 35% | Sprint 05 |
| Adaptive Query Router | Not Started | 0% | Sprint 05 |
| RAGAS Evaluation Baseline | Not Started | 0% | Sprint 05 |
| ADCL Service (district-first) | In Progress | 80% | Sprint 06 |
| Supabase/Auth Hardening | Not Started | 0% | Sprint 07 |
| Vision Agent (YOLOv12) | Not Started | 0% | Phase 3 |
| Flutter Mobile App | Not Started | 0% | Phase 4 |

---

## Recent Accomplishments

| Date | Accomplishment |
|------|----------------|
| 2026-03-17 | Canonical ADCL service, persistence, API route, voice/listing wiring, scheduler, and focused tests implemented |
| 2026-03-17 | Sprint 06 ADCL productionization sprint file, ADRs, and tracking handoff created |
| 2026-03-17 | Multi-source Karnataka rate hub added across API, tools, planner, scheduler, tests, and docs |
| 2026-03-11 | Full documentation system created (15+ docs) |
| 2026-03-09 | Multilingual memory + language state for voice |
| 2026-03-09 | Agmarknet potato scraper verified |
| 2026-03-03 | Voice WebSocket streaming with VAD |
| 2026-02-27 | RAG 2027 research + 4 ADRs |
| 2026-02-27 | Scrapling-based APMC scraper with stealth |
| 2026-02-26 | Pipecat voice pipeline (STT + TTS services) |

---

## Current Priorities

1. **Sprint 06:** Apply ADCL migration and validate the live path against Aurora district data
2. **Sprint 06:** Run the ADCL golden-set review plus historical backtest using a real order snapshot
3. **Sprint 06:** Verify gated eNAM behavior when credentials land and confirm IMD live advisories end to end
4. **Sprint 05:** Adaptive Query Router + AgriEmbeddingWrapper
5. **Sprint 05:** RAGAS evaluation baseline (20 golden queries)
6. **Sprint 07:** Supabase/auth hardening follow-up after ADCL productionization
7. **Ongoing:** Voice pipeline optimization (target: <2s latency)

---

## Blockers

| Blocker | Impact | Mitigation |
|---------|--------|-----------|
| eNAM API registration pending | Limits gated-source verification for the rate hub and ADCL | Use internal demand plus official-first public sources; keep eNAM behind a flag |
| No 8-week district order snapshot in repo | Blocks a real historical ADCL backtest from being executed locally | Use the new backtest runbook and run once Aurora data is available |
| BGE-M3 requires 1GB RAM | Slow on low-memory machines | MiniLM-L6-v2 fallback available |
| Repo-wide Ruff and mypy backlog | CI is noisy and full-repo checks fail outside feature slices | Track cleanup separately from feature delivery; keep feature-level verification explicit |

---

## Architecture Decisions (ADRs)

| ADR | Decision | Status |
|-----|----------|--------|
| ADR-007 | Replace fixed RAG pipeline with agentic orchestrator | Approved |
| ADR-008 | 8-strategy adaptive query router | Approved |
| ADR-009 | Two-layer agri embedding strategy | Approved |
| ADR-010 | Browser-augmented RAG via Scrapling | Approved |
| ADR-011 | Multi-source Karnataka rate hub with official-first precedence | Approved |
| ADR-012 | ADCL district-first canonical service contract | Approved |
| ADR-013 | ADCL marketplace-first source precedence with explicit evidence | Approved |

---

## Next Session Start Here

1. Read `tracking/sprints/sprint-06-adcl-productionization.md`
2. Read `tracking/daily/2026-03-17.md`
3. Review `src/agents/adcl/service.py`, `src/api/routes/adcl.py`, and `src/api/runtime/services.py`
4. Use `src/evaluation/reports/adcl_backtest_2026-03-17.md` when moving from fixture validation to live district validation

---

## Key Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Agent routing accuracy | >90% | ~85% |
| Voice response latency | <2s | ~3-4s |
| API P95 latency | <500ms | ~300ms (cached) |
| Cost per query | <Rs0.50 | ~Rs0.44 |
| Test coverage | >70% | ~40% |
