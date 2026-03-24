# Sprint 10 - Voice Orchestration, State, and Tools

> **Period:** 2026-05-20 -> 2026-06-02
> **Theme:** Add production-grade voice state management, multi-agent routing, memory, and tool-calling on top of the stabilized bridge path
> **Sprint Status:** In Progress

---

## Goals (Measurable)

1. Voice turns move through one explicit distributed state machine instead of ad hoc per-service state.
2. Router plus domain-agent voice orchestration is active for the CropFresh personas and major task classes.
3. Redis-backed conversation memory and context injection are stable across the bridge path.
4. Mid-conversation tool usage is wired for voice-facing price, listing, logistics, and marketplace actions.
5. Speaker-aware handling exists for the first multi-speaker or group-call scenarios.

---

## Entry Assumptions

- Sprint 09 delivers a stable bridge path with reconnect-safe state and benchmark artifacts.
- The current FastAPI voice path still remains the compatibility fallback while orchestration service boundaries are expanded.

---

## Scope (Stories / Tasks)

### Distributed Voice State Machine

- [x] Implement the canonical voice state model:
  - `IDLE`
  - `LISTENING`
  - `VAD_TRIGGERED`
  - `TRANSCRIBING`
  - `THINKING`
  - `SPEAKING`
  - `BARGE_IN`
- [x] Broadcast state transitions over Redis pub/sub for gateway, orchestration, and UI consumers.
- [x] Keep state correlation tied to one stable session id across services.
- [x] Tests: state transition and interruption coverage.

### Multi-Agent Voice Routing

- [x] Add or extract the voice orchestration service boundary for router plus specialist agents.
- [x] Define the initial voice-facing agent roster:
  - Priya - farmer assistant
  - Arjun - market agent
  - Ravi - logistics agent
  - Admin - supervisor escalation
- [x] Route voice turns by intent and context instead of one monolithic response path.
- [x] Keep the current voice agent as a compatibility fallback during rollout.

### Memory and Context Injection

- [x] Persist the last 10 turns plus language, user profile, and active workflow context in Redis.
- [x] Inject user profile, session history, and live market context per turn.
- [x] Ensure reconnect recovery uses the same memory contract instead of a parallel voice-only format.

### Tool Calling and Speaker Awareness

- [x] Wire voice-safe tool invocation for price lookup, listing actions, logistics workflow, and relevant shared services.
- [x] Add initial speaker diarization integration for grouped voice turns via speaker hints and stable grouped-turn speaker profiles.
- [x] Store speaker metadata or embeddings only where needed for continuity and debugging.
- [x] Tests: tool-routing, memory usage, and grouped-speaker behavior.

---

## First Implementation Slice

1. Lock the Redis-backed state schema and state-transition events.
2. Add a router-first orchestration path for price and listing intents.
3. Reuse the shared memory contract for voice turns.
4. Add focused tests for state transitions and tool invocation.

## Implementation Progress - 2026-03-24

- Landed the first Sprint 10 slice directly on the shared FastAPI voice runtime so REST fallback and duplex websocket paths now use one canonical voice-state contract.
- Added `VoiceSessionState` and `VoiceStateEvent` plus state-transition validation, in-memory event history, and Redis pub/sub broadcasting in `src/memory/state_pkg/manager.py`.
- Added `src/voice/orchestration/service.py` and `src/voice/orchestration/models.py` for router-first voice orchestration across price, listing, logistics, and supervisor fallback flows using the Priya, Arjun, Ravi, and Admin personas.
- Wired `src/agents/voice/agent.py`, `src/api/runtime/services.py`, `src/api/rest/voice_runtime.py`, and `src/api/websocket/voice_pkg/router.py` so shared workflow memory, reconnect-safe turn history, and routed-agent context persist across new turns and duplex recovery.
- Added `process_duplex_text_response(...)` so orchestrated turns can bypass the LLM stream without breaking websocket timing, history, or interruption semantics.
- Added focused test coverage for state transitions, orchestrator routing, shared voice-session hydration, and duplex orchestration.
- Added the grouped-speaker follow-on slice: shared speaker profiles in session state, per-turn speaker metadata, REST speaker propagation, duplex `speaker_hint` plus `speaker_ack`, and grouped-turn persistence tests.

### Verification - 2026-03-24

- `uv run pytest tests/unit/test_voice_state_machine.py tests/unit/test_voice_orchestrator.py tests/unit/test_voice_agent_stateful_runtime.py tests/unit/test_voice_duplex_orchestration.py` -> passed (`9 passed`)
- `uv run pytest tests/unit/test_voice_state_machine.py tests/unit/test_voice_speaker_state.py tests/unit/test_voice_orchestrator.py tests/unit/test_voice_agent_stateful_runtime.py tests/unit/test_voice_duplex_orchestration.py tests/api/test_voice_rest_speaker_context.py` -> passed (`14 passed`)

---

## Acceptance / Done Criteria

- [x] One canonical state machine is visible across the main bridge services.
- [x] Router and specialist-agent voice handling is active for at least the main price, listing, logistics, and fallback flows.
- [x] Redis conversation memory and context injection are reused consistently across reconnects and new turns.
- [x] Voice tool calls work without breaking interruption handling.
- [x] Sprint notes capture speaker-awareness limits and any rollout constraints.

---

## Out of Scope

- No heavy load-hardening or k6 target work yet; that belongs to Sprint 11.
- No cluster deployment or production cutover work yet; that belongs to Sprint 12.
- No Flutter or Next.js client build in this sprint.

---

## Risks / Open Questions

- Distributed state can drift if correlation ids or Redis event contracts are not tightly defined.
- Multi-agent routing may add latency if prompt and tool boundaries are too broad.
- Speaker diarization can be expensive and should stay scoped to the smallest useful slice first.

## Current Limits

- Full ML diarization is still deferred; the current Sprint 10 slice relies on stable speaker hints/labels and grouped-turn speaker profiles instead of model-based speaker separation.
- The new orchestration layer currently runs inside the existing FastAPI runtime, not the planned bridge services.
- The routed-tool slice is intentionally narrow: price lookup, listing flows, logistics delegation, and supervisor fallback are in scope now; broader marketplace tools still need follow-on work.

---

## Related Files

- `tracking/sprints/sprint-09-semantic-vad-continuity-and-session-recovery.md`
- `tracking/sprints/sprint-11-voice-load-hardening-and-observability.md`
- `docs/features/livekit-voice-bridge.md`
- `services/voice-gateway/`
- `services/vad-service/`
- `src/agents/`
