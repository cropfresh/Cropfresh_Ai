# CropFresh AI - Project Status

> **Last Updated:** 2026-03-24
> **Phase:** Phase 2 - Business Services
> **Sprint:** Sprint 10 (in progress)
> **Next Sprint:** Sprint 11 - Voice Load Hardening and Observability

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
| Duplex WebSocket Voice Path | In Progress | 85% | Sprint 07 carryover |
| Pipecat Voice Pipeline | In Progress | 60% | Sprint 07 (experimental) |
| LiveKit Voice Bridge Foundation | Complete | 100% | Sprint 08 |
| Semantic VAD + Session Recovery | Complete | 100% | Sprint 09 |
| Voice Orchestration State + Tools | In Progress | 65% | Sprint 10 |
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
| 2026-03-24 | Sprint 10 speaker-aware slice landed: grouped-turn speaker profiles, per-turn speaker metadata, REST speaker propagation, duplex `speaker_hint` plus `speaker_ack`, and focused grouped-speaker tests (`14 passed`) |
| 2026-03-24 | Sprint 10 first slice landed: canonical voice-state transitions/events, router-first voice orchestration for price/listing/logistics/fallback, shared workflow memory reuse across REST plus duplex paths, and focused Sprint 10 tests (`9 passed`) |
| 2026-03-24 | Sprint 09 continuity/recovery slice landed: comfort-noise fills, relay debug metadata, retry/backoff policy, client dead-peer recovery, and focused stale-session/heartbeat-timeout coverage |
| 2026-03-24 | Sprint 09 benchmark/eval slice landed: fixed multilingual utterance set, artifact runner, rubric, and heuristic baseline matched `8/8` |
| 2026-03-24 | Sprint 09 semantic relay slice landed: joint acoustic+semantic endpointing, timeout-safe hold/flush logic, and continuity metrics |
| 2026-03-18 | Sprint 08 closeout landed: gateway relay frame/flush/reset contract, VAD FastAPI analyze/reset surface, Voice Hub bootstrap fallback flow, and focused closeout tests |
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

1. **Sprint 10 source of truth:** Start from `tracking/sprints/sprint-10-voice-orchestration-state-and-tools.md`.
2. **Sprint 10:** Expand the canonical voice-state machine across the remaining bridge-facing services and keep the FastAPI duplex path as the truthful fallback runtime.
3. **Sprint 10:** Broaden router-first orchestration beyond the first price/listing/logistics/fallback slice without collapsing back into one monolithic voice handler.
4. **Sprint 10:** Reuse the shared Redis conversation contract for workflow memory, routed-agent context, and reconnect-safe turn history on every voice entry path.
5. **Sprint 10:** Extend the new speaker-aware hint/profile contract carefully while keeping full ML diarization and embeddings deferred until they can be benchmarked.
6. **Voice program guardrails:** Keep load hardening, observability, and deployment work inside Sprint 11-12 boundaries.
7. **Sprint 06 carryover:** Run the ADCL golden-set review plus historical backtest using a real district order snapshot.
8. **Sprint 05 carryover:** Adaptive Query Router, AgriEmbeddingWrapper, and RAGAS baseline remain backlog debt after the current voice slice.

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

1. Read `tracking/sprints/sprint-10-voice-orchestration-state-and-tools.md`.
2. Read `tracking/PROJECT_STATUS.md` and `tracking/daily/2026-03-24.md`.
3. Review `src/memory/state_pkg/manager.py` and `src/memory/state_pkg/models.py`.
4. Review `src/voice/orchestration/service.py` and `src/voice/orchestration/models.py`.
5. Cross-read `src/agents/voice/agent.py`, `src/api/rest/voice_runtime.py`, and `src/api/runtime/services.py`.
6. Cross-read `src/api/websocket/voice_pkg/router.py`, `src/api/websocket/voice_pkg/duplex.py`, and `src/api/rest/voice.py`.
7. Run `uv run pytest tests/unit/test_voice_state_machine.py tests/unit/test_voice_speaker_state.py tests/unit/test_voice_orchestrator.py tests/unit/test_voice_agent_stateful_runtime.py tests/unit/test_voice_duplex_orchestration.py tests/api/test_voice_rest_speaker_context.py`.

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

---

## 2026-03-24 CI/CD Recovery Addendum

- Repo-wide CI is green again on the intended commands: `uv run ruff check src/ ai/`, `uv run pytest tests/ -v --tb=short`, `npm run build`, and `npm test` in `services/voice-gateway`.
- GitHub Actions now treats AWS App Runner as the only live deploy target, and the AWS deploy workflow is gated on a successful `CI — Lint + Test` run on `main`.
- The scheduled scraper workflow now runs explicit one-shot job IDs through `python -m src.scrapers` and can connect to Aurora when `PG_*` secrets are populated in GitHub Actions.
- The legacy realtime websocket smoke tests remain in the repo, but they are now opt-in E2E checks instead of mandatory CI tests because they require a manually running live server.
- Remaining external dependency: remote AWS deploy and scheduled scraper persistence still depend on valid GitHub secrets for Aurora and AWS; local verification on 2026-03-24 ran without `PG_HOST`, so persistence could only be smoke-tested in no-DB mode.
