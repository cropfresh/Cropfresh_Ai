# Sprint 10 - Voice Orchestration, State, and Tools

> **Period:** 2026-05-20 -> 2026-06-02
> **Theme:** Add production-grade voice state management, multi-agent routing, memory, and tool-calling on top of the stabilized bridge path
> **Sprint Status:** Not Started

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

- [ ] Implement the canonical voice state model:
  - `IDLE`
  - `LISTENING`
  - `VAD_TRIGGERED`
  - `TRANSCRIBING`
  - `THINKING`
  - `SPEAKING`
  - `BARGE_IN`
- [ ] Broadcast state transitions over Redis pub/sub for gateway, orchestration, and UI consumers.
- [ ] Keep state correlation tied to one stable session id across services.
- [ ] Tests: state transition and interruption coverage.

### Multi-Agent Voice Routing

- [ ] Add or extract the voice orchestration service boundary for router plus specialist agents.
- [ ] Define the initial voice-facing agent roster:
  - Priya - farmer assistant
  - Arjun - market agent
  - Ravi - logistics agent
  - Admin - supervisor escalation
- [ ] Route voice turns by intent and context instead of one monolithic response path.
- [ ] Keep the current voice agent as a compatibility fallback during rollout.

### Memory and Context Injection

- [ ] Persist the last 10 turns plus language, user profile, and active workflow context in Redis.
- [ ] Inject user profile, session history, and live market context per turn.
- [ ] Ensure reconnect recovery uses the same memory contract instead of a parallel voice-only format.

### Tool Calling and Speaker Awareness

- [ ] Wire voice-safe tool invocation for price lookup, listing actions, logistics workflow, and relevant shared services.
- [ ] Add initial speaker diarization integration for grouped voice turns.
- [ ] Store speaker metadata or embeddings only where needed for continuity and debugging.
- [ ] Tests: tool-routing, memory usage, and grouped-speaker behavior.

---

## First Implementation Slice

1. Lock the Redis-backed state schema and state-transition events.
2. Add a router-first orchestration path for price and listing intents.
3. Reuse the shared memory contract for voice turns.
4. Add focused tests for state transitions and tool invocation.

---

## Acceptance / Done Criteria

- [ ] One canonical state machine is visible across the main bridge services.
- [ ] Router and specialist-agent voice handling is active for at least the main price, listing, logistics, and fallback flows.
- [ ] Redis conversation memory and context injection are reused consistently across reconnects and new turns.
- [ ] Voice tool calls work without breaking interruption handling.
- [ ] Sprint notes capture speaker-awareness limits and any rollout constraints.

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

---

## Related Files

- `tracking/sprints/sprint-09-semantic-vad-continuity-and-session-recovery.md`
- `tracking/sprints/sprint-11-voice-load-hardening-and-observability.md`
- `docs/features/livekit-voice-bridge.md`
- `services/voice-gateway/`
- `services/vad-service/`
- `src/agents/`

