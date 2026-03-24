# Sprint 08 - LiveKit Voice Bridge Foundation

> **Period:** 2026-03-18 -> 2026-03-18
> **Theme:** Stand up a feature-flagged LiveKit bridge path that coexists with the current duplex websocket stack and removes re-planning for the next implementation session
> **Sprint Status:** Complete

---

## Goals (Measurable)

1. A new voice gateway can bootstrap browser voice sessions with LiveKit room metadata and a safe fallback back to `/api/v1/voice/ws/duplex`.
2. A standalone Python VAD service exposes Silero-based acoustic segmentation over FastAPI plus gRPC using the agreed dual-threshold settings.
3. The current FastAPI duplex websocket path remains the truthful production-facing runtime while Sprint 08 adds bridge-mode infrastructure around it.
4. Static voice pages can exercise bridge mode without replacing the current websocket demo path.
5. Sprint, ADR, feature-doc, daily-log, and status/backlog pointers all agree so the next implementation session can start without re-planning.

---

## Current Reality to Carry Into the Sprint

- The canonical realtime path in the repo today is `/api/v1/voice/ws/duplex`.
- Current voice transport is JSON text frames carrying base64 audio, not a LiveKit production path.
- The current repo provider stack for active voice work is still Groq Whisper plus Edge/local Indic TTS.
- `mobile/` is still only a placeholder and there is no Next.js app in the repo yet.
- Sprint 07 remains useful context for duplex hardening, latency instrumentation, and provider-policy cleanup, but Sprint 08 is the new implementation-facing handoff for the next session.

---

## Scope (Stories / Tasks)

### Voice Gateway (Node.js / TypeScript)

- [x] Create `services/voice-gateway/` as a strict-mode TypeScript service.
- [x] Add bootstrap and operational routes:
  - `POST /sessions/bootstrap`
  - `GET /health`
  - `GET /ready`
  - `GET /metrics`
- [x] Bootstrap LiveKit rooms/tokens while keeping gateway behavior behind a feature flag.
- [x] Add a 5-second PCM ring buffer and RMS pre-gate before downstream relay.
- [x] Fallback to the existing `/api/v1/voice/ws/duplex` path when LiveKit bootstrap, relay, or media setup fails.
- [x] Tests: unit coverage for bootstrap behavior, readiness, fallback selection, and ring-buffer wraparound.

### Python VAD Service

- [x] Create `services/vad-service/` as a FastAPI plus gRPC service.
- [x] Expose `/health` and `/ready` plus a streaming gRPC endpoint for frame-by-frame acoustic segmentation.
- [x] Use Silero ONNX with:
  - speech onset threshold `0.5`
  - speech offset threshold `0.35`
  - minimum speech duration `250ms`
  - trailing silence padding `300ms`
  - streaming chunk size `512` samples at `16kHz`
- [x] Emit segment-ready events only; semantic endpointing is explicitly deferred to a later sprint.
- [x] Tests: unit coverage `>= 80%` for segmentation rules, model bootstrap failure, and silence/click-noise filtering.

### Bridge Integration with Existing Voice Runtime

- [x] Keep `src/api/websocket/voice_pkg/router.py`, `src/api/websocket/voice_pkg/duplex.py`, and `src/voice/duplex_pipeline.py` as the downstream speech engine for Phase 1.
- [x] Define the internal relay contract from the gateway into the existing duplex websocket runtime instead of splitting STT/TTS/agent orchestration in Sprint 08.
- [x] Preserve the current Groq plus Edge/local provider path inside that downstream runtime; do not switch Phase 1 to Deepgram/Cartesia.
- [x] Emit bridge-visible timing for bootstrap, VAD segment flush, and downstream first-audio timing where feasible without breaking the current websocket contract.
- [x] Tests: focused integration coverage for bridge-mode relay and forced websocket fallback.

### Minimal Static Client Surface

- [x] Extend the current static voice surfaces only:
  - `static/premium_voice.html`
  - `static/voice_agent.html`
  - their JS helpers as needed
- [x] Add a small bridge bootstrap step plus a visible fallback indicator.
- [x] Keep the existing websocket demos working if bridge mode is disabled or unavailable.
- [x] Do not create a new Next.js or Flutter client in this sprint.

### Documentation and Tracking

- [x] Keep `WORKFLOW_STATUS.md`, `tracking/PROJECT_STATUS.md`, and daily logs aligned as implementation lands.
- [x] Record follow-up ADRs only if the bridge contract or service boundary changes materially during implementation.

---

## First Implementation Slice

1. Create `services/voice-gateway/` with config, `POST /sessions/bootstrap`, `/health`, and `/ready`.
2. Create `services/vad-service/` with `/health`, `/ready`, and the initial gRPC/FastAPI contract surface.
3. Add one minimal static bootstrap path that calls the gateway first and falls back to `/api/v1/voice/ws/duplex`.
4. Add focused tests for bootstrap and fallback before any deeper media transport work.

---

## Implementation Snapshot (2026-03-18)

- Added `services/voice-gateway/` with a strict TypeScript scaffold, `POST /sessions/bootstrap`, `/health`, `/ready`, `/metrics`, and initial ring-buffer plus RMS-gate utilities.
- Added `services/vad-service/` with FastAPI probes, `/v1/vad/config`, `/v1/vad/analyze`, session reset support, the Sprint 08 dual-threshold segmenter, and the initial protobuf plus gRPC server surface.
- Added local Dockerfiles plus `docker-compose.yml` entries for `voice-gateway` and `vad-service`.
- Wired gateway relay frame, flush, and reset routes so the ring buffer and RMS gate now feed the downstream FastAPI duplex websocket path through a narrow bridge contract.
- Updated `static/premium_voice.html`, `static/voice_agent.html`, and their JS helpers so both static voice surfaces now request bootstrap metadata first and visibly preserve websocket fallback behavior.
- Added focused Python, static, and Node tests for the Sprint 08 closeout slice.

Deferred beyond Sprint 08 close:

- browser-side LiveKit media transport and room join flow

---

## Public Interfaces to Lock Before Coding

### Browser Bootstrap Response

- `session_id`
- `mode`
- `livekit_url`
- `token`
- `fallback_ws_url`
- `features`

### VAD gRPC Contract

- Client stream message:
  - `session_id`
  - `sequence`
  - `sample_rate`
  - `pcm16`
- Server event message:
  - `state`
  - `probability`
  - `rms`
  - `segment_id`
  - `sequence`
  - `end_of_segment`

Backward-compatibility rule: the current websocket JSON contract remains valid throughout Sprint 08; any new timing fields are additive only.

---

## Acceptance / Done Criteria

- [x] The repo contains a scaffolded `services/voice-gateway/` and `services/vad-service/` with health/readiness endpoints and the planned contracts.
- [x] One static voice client can bootstrap bridge mode and visibly fall back to `/api/v1/voice/ws/duplex` when bridge mode is unavailable.
- [x] The existing duplex websocket path remains usable and is not rewritten as if LiveKit were already the canonical runtime.
- [x] Focused unit and integration tests cover bootstrap, fallback, and VAD segmentation behavior.
- [x] Sprint notes link to the ADR, planned bridge doc, and any implementation follow-up gaps.

---

## Out of Scope

- No Next.js 15 web app in this sprint.
- No Flutter client implementation in this sprint.
- No direct production cutover away from `/api/v1/voice/ws/duplex`.
- No semantic VAD, speaker diarization, or wake-word pipeline in Sprint 08.
- No provider swap to Deepgram or Cartesia in Sprint 08.
- No Kubernetes or multi-node LiveKit deployment work in Sprint 08.

---

## Risks / Open Questions

- LiveKit credentials, room policy, and deployment shape are not yet represented in the repo.
- Downstream latency may still be dominated by the existing duplex pipeline even after bridge scaffolding lands.
- A gateway plus downstream-websocket bridge can introduce double buffering if relay boundaries are not kept tight.
- The current fixed multilingual benchmark set still needs to be created before latency and voice-quality comparisons are trustworthy.

---

## Assumptions

- This sprint is a bridge sprint, not a direct cutover sprint.
- Existing runtime truth stays unchanged until implementation lands.
- The current provider path remains Groq plus Edge/local Indic TTS for Phase 1.
- Sprint 07 remains a useful duplex-runtime reference, but Sprint 08 is the formal next-session handoff target.

---

## Follow-On Sprint Boundaries

Sprint 08 is now the first step of an explicit voice-program sequence so later work does not leak into this implementation slice.

- Sprint 09: semantic VAD, stream continuity, barge-in refinement, reconnect recovery, and the multilingual benchmark set.
- Sprint 10: distributed voice state, multi-agent routing, memory reuse, tool calling, and initial speaker awareness.
- Sprint 11: load hardening, bulkheads, observability, and k6-based reliability baselines.
- Sprint 12: scale, security, deployment assets, compliance, and cutover-readiness decisions.

If a task does not help land the Sprint 08 bridge scaffold, move it into the matching Sprint 09-12 file instead of expanding this sprint.

---

## Next Session Execution Checklist

1. Read `tracking/sprints/sprint-08-livekit-voice-bridge-foundation.md`.
2. Skim `tracking/sprints/sprint-09-semantic-vad-continuity-and-session-recovery.md`, `tracking/sprints/sprint-10-voice-orchestration-state-and-tools.md`, `tracking/sprints/sprint-11-voice-load-hardening-and-observability.md`, and `tracking/sprints/sprint-12-livekit-scale-security-and-deployment.md` to keep later work out of Sprint 08.
3. Read `docs/decisions/ADR-015-livekit-bridge-hybrid-cutover.md`.
4. Read `docs/features/livekit-voice-bridge.md` and `tracking/daily/2026-03-18.md`.
5. Cross-read `docs/api/websocket-voice.md` and `docs/features/voice-pipeline.md` to stay grounded in the current runtime.
6. Review `src/api/websocket/voice_pkg/router.py`, `src/api/websocket/voice_pkg/duplex.py`, and `src/voice/duplex_pipeline.py`.
7. Open `static/premium_voice.html` and `static/assets/js/voice-agent-duplex.js` before changing the live demo flow.
8. Re-run focused voice tests before editing runtime code:
   - `uv run pytest tests/unit/test_voice_agent.py -q`
   - `uv run pytest tests/unit/test_pipecat_pipeline.py -q`

---

## Sprint Outcome (fill at end of sprint)

**What Shipped:**
- `services/voice-gateway/` now exposes relay frame, flush, and reset routes so gateway-buffered PCM can flow into the existing duplex websocket runtime without replacing it.
- `services/vad-service/` now exposes both gRPC and FastAPI analyze/reset surfaces for Sprint 08 acoustic segmentation.
- `static/premium_voice.html` and `static/voice_agent.html` now bootstrap through the gateway first and preserve websocket fallback with reconnect tokens and heartbeat handling.
- Focused verification passed for the Sprint 08 slice across Python, static, TypeScript build, and gateway Vitest coverage.

**What Slipped to Next Sprint:**
- Browser-side LiveKit media room join and direct media transport remain deferred until the follow-on voice-program work.

**Key Learnings:**
- A thin HTTP compatibility surface on the VAD service kept the gateway moving without blocking on a same-sprint Node gRPC client.
- Keeping the duplex websocket as the downstream engine let bridge infrastructure land without rewriting truthful runtime docs.
- Windows sandboxing can block Vitest process helpers, so gateway verification may need an unrestricted test run even when the code is correct.

**Bridge Verification Snapshot:**

| Scenario | Bootstrap | First Audio | Notes |
|----------|-----------|-------------|-------|
| Premium static fallback bootstrap | Pass | Websocket fallback | Premium page requests bootstrap metadata and stays on `/api/v1/voice/ws/duplex` when bridge media is unavailable |
| Voice Hub fallback bootstrap | Pass | Websocket fallback | Voice Hub now reuses bootstrap and heartbeat behavior without replacing the live duplex demo |
| Gateway relay flush | Pass | Downstream timing preserved | Buffered PCM can be flushed through the gateway into the existing duplex websocket runtime |

---

## Related Files

- `tracking/PROJECT_STATUS.md`
- `WORKFLOW_STATUS.md`
- `docs/decisions/ADR-015-livekit-bridge-hybrid-cutover.md`
- `docs/features/livekit-voice-bridge.md`
- `tracking/sprints/sprint-09-semantic-vad-continuity-and-session-recovery.md`
- `tracking/sprints/sprint-10-voice-orchestration-state-and-tools.md`
- `tracking/sprints/sprint-11-voice-load-hardening-and-observability.md`
- `tracking/sprints/sprint-12-livekit-scale-security-and-deployment.md`
- `tracking/daily/2026-03-18.md`
- `docs/api/websocket-voice.md`
- `docs/features/voice-pipeline.md`
- `src/api/websocket/voice_pkg/router.py`
- `src/api/websocket/voice_pkg/duplex.py`
- `src/voice/duplex_pipeline.py`
- `static/premium_voice.html`
- `static/voice_agent.html`
- `static/assets/js/voice-agent-bootstrap.js`
- `static/assets/js/voice-agent-ws.js`
- `services/voice-gateway/src/routes/relay.ts`
- `services/voice-gateway/src/services/relay-coordinator.ts`
- `services/voice-gateway/src/services/downstream-relay.ts`
- `services/voice-gateway/src/services/vad-client.ts`
- `services/vad-service/app/api.py`
- `services/vad-service/app/http_models.py`
- `services/vad-service/app/runtime.py`
