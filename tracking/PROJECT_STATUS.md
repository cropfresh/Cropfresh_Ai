# CropFresh AI - Project Status

> **Last Updated:** 2026-03-18
> **Phase:** Phase 2 - Business Services
> **Sprint:** Sprint 06 (current)
> **Next Sprint:** Sprint 08 - LiveKit Voice Bridge Foundation

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
| Duplex WebSocket Voice Path | In Progress | 80% | Sprint 07 carryover |
| Pipecat Voice Pipeline | In Progress | 60% | Sprint 07 (experimental) |
| LiveKit Voice Bridge Foundation | In Progress | 15% | Sprint 08 |
| Semantic VAD + Session Recovery | Not Started | 0% | Sprint 09 |
| Voice Orchestration State + Tools | Not Started | 0% | Sprint 10 |
| Voice Load Hardening + Observability | Not Started | 0% | Sprint 11 |
| LiveKit Scale + Deployment | Not Started | 0% | Sprint 12 |
| Tool Registry + Shared Tools | Complete | 100% | Sprint 01-05 |
| Agmarknet APMC Scraper | Complete | 100% | Sprint 04 |
| Documentation System | In Progress | 95% | Sprint 05-07 |
| Multi-Source Rate Hub | In Progress | 85% | Sprint 05 |
| AgriEmbeddingWrapper (L1) | Not Started | 0% | Sprint 05 |
| Agentic RAG Orchestrator | In Progress | 35% | Sprint 05 |
| Adaptive Query Router | Not Started | 0% | Sprint 05 |
| RAGAS Evaluation Baseline | Not Started | 0% | Sprint 05 |
| ADCL Service (district-first) | In Progress | 80% | Sprint 06 |
| Supabase/Auth Hardening | Not Started | 0% | Backlog after Sprint 12 |
| Vision Agent (YOLOv12) | Not Started | 0% | Phase 3 |
| Flutter Mobile App | Not Started | 0% | Phase 4 |

---

## Recent Accomplishments

| Date | Accomplishment |
|------|----------------|
| 2026-03-18 | Sprint 08 first implementation slice landed: `services/voice-gateway/`, `services/vad-service/`, premium bootstrap fallback flow, Dockerfiles, compose entries, and focused tests |
| 2026-03-18 | Sprint 09-12 voice-program sprint files created and linked into roadmap, backlog, workflow, and handoff docs |
| 2026-03-18 | Docs-first Sprint 08 handoff created with a new sprint file, ADR-015, planned bridge architecture doc, daily log, and synced tracking pointers |
| 2026-03-17 | Full markdown sweep started for the Sprint 07 voice duplex handoff, including provider-policy cleanup and websocket doc corrections |
| 2026-03-17 | Sprint 07 voice duplex productionization sprint file created with realistic latency targets and Bedrock-removal scope |
| 2026-03-17 | Canonical ADCL service, persistence, API route, voice/listing wiring, scheduler, and focused tests implemented |
| 2026-03-17 | Sprint 06 ADCL productionization sprint file, ADRs, and tracking handoff created |
| 2026-03-17 | Multi-source Karnataka rate hub added across API, tools, planner, scheduler, tests, and docs |
| 2026-03-11 | Full documentation system created (15+ docs) |
| 2026-03-09 | Multilingual memory plus language state for voice |
| 2026-03-03 | Voice WebSocket streaming with VAD |
| 2026-02-27 | RAG 2027 research plus 4 ADRs |
| 2026-02-26 | Pipecat voice pipeline (STT plus TTS services) |

---

## Current Priorities

1. **Sprint 08 source of truth:** Continue from the new gateway, VAD, and premium bootstrap scaffolds already in repo instead of reopening the bridge architecture.
2. **Sprint 08:** Wire the gateway ring buffer and RMS gate into the actual downstream relay path instead of leaving them as isolated utilities.
3. **Sprint 08:** Move from bootstrap-only bridge mode toward a real gateway -> VAD -> duplex relay loop while preserving `/api/v1/voice/ws/duplex` as the truthful live fallback.
4. **Sprint 08:** Keep the current Groq plus Edge/local Indic provider path for Phase 1 and defer provider swaps until after bridge relay scaffolding lands.
5. **Voice program guardrails:** Use the new Sprint 09-12 docs to keep semantic endpointing, orchestration, hardening, and deployment work out of Sprint 08.
6. **Sprint 06 carryover:** Run the ADCL golden-set review plus historical backtest using a real district order snapshot.
7. **Sprint 06 carryover:** Verify gated eNAM behavior when credentials land and confirm IMD live advisories end to end.
8. **Sprint 05 carryover:** Adaptive Query Router, AgriEmbeddingWrapper, and RAGAS baseline remain backlog debt after the voice bridge handoff.

---

## Blockers

| Blocker | Impact | Mitigation |
|---------|--------|-----------|
| No fixed multilingual voice benchmark set in repo | Makes latency and naturalness comparisons noisy across sessions | Keep benchmark-set creation in the Sprint 08 voice bridge plan before trusting latency comparisons |
| Pipecat slice still has failing focused tests | Experimental voice path is not trustworthy enough for production claims | Keep Pipecat explicitly experimental and out of the Sprint 08 critical path |
| No 8-week district order snapshot in repo | Blocks a real ADCL backtest from being executed locally | Use the existing runbook and run once Aurora data is available |
| eNAM API registration pending | Limits gated-source verification for ADCL and rate-hub flows | Keep eNAM behind a flag and continue with official-first public sources |
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
| ADR-014 | Duplex websocket as the canonical production voice path | Planned for Sprint 07 |
| ADR-015 | LiveKit bridge hybrid cutover instead of immediate websocket replacement | Approved |

---

## Next Session Start Here

1. Read `tracking/sprints/sprint-08-livekit-voice-bridge-foundation.md`.
2. Skim `tracking/sprints/sprint-09-semantic-vad-continuity-and-session-recovery.md`, `tracking/sprints/sprint-10-voice-orchestration-state-and-tools.md`, `tracking/sprints/sprint-11-voice-load-hardening-and-observability.md`, and `tracking/sprints/sprint-12-livekit-scale-security-and-deployment.md` to keep future boundaries fixed.
3. Read `docs/decisions/ADR-015-livekit-bridge-hybrid-cutover.md`.
4. Read `docs/features/livekit-voice-bridge.md` and `tracking/daily/2026-03-18.md`.
5. Cross-read `docs/api/websocket-voice.md` and `docs/features/voice-pipeline.md` for current runtime truth.
6. Review `services/voice-gateway/src/server.ts`, `services/voice-gateway/src/services/session-bootstrap.ts`, and `services/vad-service/app/run_service.py`.
7. Review `static/assets/js/duplex/bootstrap.js`, `static/assets/js/duplex/socket.js`, and `static/premium_voice.html`.
8. Cross-read `src/api/websocket/voice_pkg/router.py`, `src/api/websocket/voice_pkg/duplex.py`, and `src/voice/duplex_pipeline.py`.

---

## Key Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Agent routing accuracy | >90% | ~85% |
| Voice first-audio latency | <1.2s P95 | Not instrumented end to end yet |
| Voice full response latency | <2s P95 | ~3-4s |
| API P95 latency | <500ms | ~300ms (cached) |
| Cost per query | <Rs0.50 | ~Rs0.44 |
| Test coverage | >70% | ~40% |
