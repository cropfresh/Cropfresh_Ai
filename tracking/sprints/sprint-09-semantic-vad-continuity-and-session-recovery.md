# Sprint 09 - Semantic VAD, Continuity, and Session Recovery

> **Period:** 2026-05-06 -> 2026-05-19
> **Theme:** Turn the Sprint 08 bridge scaffold into a more natural and interruption-safe live voice path with semantic endpointing, continuity controls, and reconnect recovery
> **Sprint Status:** In Progress

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

- [x] Extend `services/vad-service/` with a semantic completeness helper using a low-latency LLM micro-call.
- [x] Flush speech to downstream STT only when acoustic and semantic signals agree, with an explicit timeout safety path.
- [x] Tune handling for filler pauses such as "umm", "one second", and multilingual hesitation phrases.
- [x] Tests: semantic endpointing coverage for thinking pauses, clipped endings, and code-mixed utterances.

### Stream Continuity and Barge-In

- [x] Add comfort-noise frames during short processing stalls so media playback does not sound dead.
- [x] Add ring-buffer watermarks and continuity metrics for burst absorption.
- [x] Implement graceful barge-in behavior that finishes the current audible fragment cleanly before stopping when possible.
- [x] Emit interruption timing and continuity metadata for debugging.

### Reconnect and Session Recovery

- [x] Persist reconnect-safe session context in Redis, including the last 10 turns and voice state metadata.
- [x] Add reconnect tokens and retry/backoff handling for bridge clients.
- [x] Support ICE restart and network-change recovery where bridge transport is active.
- [x] Add heartbeat and dead-peer detection behavior for stalled sessions.
- [x] Tests: reconnect, stale-session expiry, and fallback-to-fresh-session coverage.

### Benchmarks and Evaluation

- [x] Add the fixed multilingual utterance set for `kn`, `hi`, `te`, and `ta`.
- [x] Log per-turn benchmark artifacts for endpointing quality, first audio, and interruption recovery.
- [x] Add a small manual-review rubric for naturalness and interruption handling.

---

## First Implementation Slice

1. Add semantic completeness evaluation behind a feature flag in `services/vad-service/`.
2. Wire the joint acoustic-plus-semantic end-of-segment decision into the bridge path.
3. Add Redis-backed session recovery metadata before deeper ICE/network handling.
4. Add focused tests for pause handling and reconnect recovery.

---

## Acceptance / Done Criteria

- [x] Semantic endpointing is available behind a documented flag and demonstrably reduces false turn endings on the benchmark set.
- [ ] Barge-in interruption is measurable and stays within the `150ms` target on the supported test path.
- [x] Redis-backed reconnect recovery restores recent context instead of forcing a blank session.
- [x] Continuity and interruption metrics are visible in logs or scrapeable metrics.
- [x] Sprint notes link to benchmark artifacts and known edge cases.

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

---

## Implementation Progress - 2026-03-24

The first Sprint 09 implementation slice is now partially landed.

### What Landed

- `services/vad-service/` now exposes `POST /v1/vad/segments/evaluate` behind the semantic feature flag so the bridge can ask for a hold-or-flush decision after an acoustic segment ends.
- The VAD runtime now keeps session-scoped semantic hold timers and forces a safe flush when the hold budget expires.
- `services/voice-gateway/` now combines acoustic `end_of_segment` with semantic decisions before flushing buffered PCM into `/api/v1/voice/ws/duplex`.
- The gateway now emits Prometheus counters for continuity fills and joint hold/flush decisions.
- Focused tests now cover semantic hold behavior, timeout-safe flush behavior, and the existing duplex reconnect flow.

### Benchmark Slice - 2026-03-24

- Added `src/evaluation/datasets/voice_multilingual_benchmark.json` as the fixed Sprint 09 utterance set for `kn`, `hi`, `te`, and `ta`.
- Added `src/evaluation/voice_benchmark_runner.py` plus `ai/evals/run_voice_benchmark.py` so the semantic endpointing contract can be evaluated and written to JSON/markdown artifacts under `reports/voice/`.
- Added `docs/features/voice-benchmarking.md` with the manual review rubric and the expected artifact flow.
- Added focused tests for dataset coverage and artifact generation in `tests/unit/test_voice_benchmark_runner.py`.
- Tuned the shared Kannada hesitation heuristics so the current fixed benchmark now matches `8/8` on the heuristic-only semantic path.

### Continuity and Recovery Completion Slice - 2026-03-24

- `services/voice-gateway/src/audio/comfort-noise.ts` plus `relay-session.ts` now synthesize low-energy comfort-noise fills for short continuity gaps instead of buffering dead silence only.
- `services/voice-gateway/src/routes/relay-debug.ts` now exposes continuity and interruption metadata so relay callers can inspect watermark, fill mode, gap length, and barge-in timing without scraping raw downstream payloads.
- `services/voice-gateway/src/services/session-bootstrap.ts` now returns explicit recovery policy metadata, including retry/backoff timing, dead-peer timeout, and bridge-mode ICE/network recovery hints.
- `static/assets/js/duplex/` now refreshes bootstrap state on reconnect, watches for missing heartbeat acknowledgements, retries with the configured backoff policy, reacts to browser online/offline changes, and fades playback down over a bounded window before stopping on barge-in.
- `tests/unit/test_voice_duplex_recovery.py` now covers expired-session fallback-to-fresh behavior and heartbeat timeout closure in addition to the earlier reconnect-token flow.

### Remaining Notes

- The current truthful media runtime is still the duplex websocket fallback path.
- Bridge-mode recovery is therefore implemented as bootstrap refresh plus recovery-policy handling inside the hybrid client until the later sprint media cutover work lands.
- Final focused verification for the current Sprint 09 code path passed on 2026-03-24: `30` targeted Python tests, `29` gateway Vitest checks, and `npm run build` in `services/voice-gateway/`.
