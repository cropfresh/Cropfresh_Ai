# Sprint 09 - Semantic VAD, Continuity, and Session Recovery

> **Period:** 2026-05-06 -> 2026-05-19
> **Theme:** Turn the Sprint 08 bridge scaffold into a more natural and interruption-safe live voice path with semantic endpointing, continuity controls, and reconnect recovery
> **Sprint Status:** Not Started

---

## Goals (Measurable)

1. Acoustic plus semantic endpointing reduces false cutoffs on thinking pauses and code-mixed farmer speech.
2. Barge-in interruption reacts within `150ms` and avoids abrupt cutoff artifacts.
3. Session reconnect can resume mid-conversation using persisted session context and the last 10 turns in Redis.
4. Stream continuity remains intact during brief processing gaps through comfort-noise and ring-buffer controls.
5. A fixed multilingual benchmark set exists for `kn`, `hi`, `te`, and `ta` so later latency and quality claims are comparable.

---

## Entry Assumptions

- Sprint 08 delivers the initial `services/voice-gateway/` and `services/vad-service/` scaffolding.
- `/api/v1/voice/ws/duplex` still remains the truthful downstream runtime and fallback path.
- The current Groq plus Edge/local Indic provider path remains active during this sprint.

---

## Scope (Stories / Tasks)

### Semantic Endpointing

- [ ] Extend `services/vad-service/` with a semantic completeness helper using a low-latency LLM micro-call.
- [ ] Flush speech to downstream STT only when acoustic and semantic signals agree, with an explicit timeout safety path.
- [ ] Tune handling for filler pauses such as "umm", "one second", and multilingual hesitation phrases.
- [ ] Tests: semantic endpointing coverage for thinking pauses, clipped endings, and code-mixed utterances.

### Stream Continuity and Barge-In

- [ ] Add comfort-noise frames during short processing stalls so media playback does not sound dead.
- [ ] Add ring-buffer watermarks and continuity metrics for burst absorption.
- [ ] Implement graceful barge-in behavior that finishes the current audible fragment cleanly before stopping when possible.
- [ ] Emit interruption timing and continuity metadata for debugging.

### Reconnect and Session Recovery

- [ ] Persist reconnect-safe session context in Redis, including the last 10 turns and voice state metadata.
- [ ] Add reconnect tokens and retry/backoff handling for bridge clients.
- [ ] Support ICE restart and network-change recovery where bridge transport is active.
- [ ] Add heartbeat and dead-peer detection behavior for stalled sessions.
- [ ] Tests: reconnect, stale-session expiry, and fallback-to-fresh-session coverage.

### Benchmarks and Evaluation

- [ ] Add the fixed multilingual utterance set for `kn`, `hi`, `te`, and `ta`.
- [ ] Log per-turn benchmark artifacts for endpointing quality, first audio, and interruption recovery.
- [ ] Add a small manual-review rubric for naturalness and interruption handling.

---

## First Implementation Slice

1. Add semantic completeness evaluation behind a feature flag in `services/vad-service/`.
2. Wire the joint acoustic-plus-semantic end-of-segment decision into the bridge path.
3. Add Redis-backed session recovery metadata before deeper ICE/network handling.
4. Add focused tests for pause handling and reconnect recovery.

---

## Acceptance / Done Criteria

- [ ] Semantic endpointing is available behind a documented flag and demonstrably reduces false turn endings on the benchmark set.
- [ ] Barge-in interruption is measurable and stays within the `150ms` target on the supported test path.
- [ ] Redis-backed reconnect recovery restores recent context instead of forcing a blank session.
- [ ] Continuity and interruption metrics are visible in logs or scrapeable metrics.
- [ ] Sprint notes link to benchmark artifacts and known edge cases.

---

## Out of Scope

- No multi-agent tool-routing split yet; that belongs to Sprint 10.
- No speaker diarization or wake-word pipeline yet.
- No provider swap away from the current Phase 1 path.
- No Kubernetes or multi-node deployment work yet.

---

## Risks / Open Questions

- Semantic endpointing can add latency if the micro-call is not tightly bounded.
- Joint acoustic/semantic logic can become too conservative and delay short replies if thresholds are not tuned carefully.
- Redis session persistence needs clear expiry rules to avoid stale reconnection state.

---

## Related Files

- `tracking/sprints/sprint-08-livekit-voice-bridge-foundation.md`
- `tracking/sprints/sprint-10-voice-orchestration-state-and-tools.md`
- `docs/features/livekit-voice-bridge.md`
- `docs/decisions/ADR-015-livekit-bridge-hybrid-cutover.md`
- `services/vad-service/`
- `services/voice-gateway/`

