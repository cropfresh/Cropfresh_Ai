# Sprint 07 - Voice Duplex Productionization and Bedrock Removal

> **Period:** 2026-04-08 -> 2026-04-21
> **Theme:** Make the duplex voice path production-ready with realistic latency goals, Bedrock-free provider policy, stronger local-language speech quality, and cleaner live testing surfaces
> **Sprint Status:** Not Started

---

## Goals (Measurable)

1. `/api/v1/voice/ws/duplex` is the documented and operationally hardened canonical realtime voice interface.
2. Stage-level voice timing is visible for VAD, STT, first LLM output, first TTS chunk, first audio to client, and full turn completion.
3. Bedrock is removed from the intended production model path, fallback order, and operator documentation.
4. Voice quality for `kn`, `hi`, `te`, and `ta` is benchmarked with human-review notes and a clear primary/fallback TTS strategy.
5. The live voice test surfaces are reduced to the supported static pages and aligned with the current websocket contract.

---

## Current Reality to Carry Into the Sprint

- The canonical realtime route today is `/api/v1/voice/ws/duplex`.
- Current voice latency is still roughly `3-4s` end to end in the existing stack.
- The current duplex transport is JSON text messages carrying base64 audio chunks, not binary PCM frames.
- Pipecat remains experimental and is not the production-default voice path.
- Bedrock references still exist in code as of 2026-03-17, but this sprint removes them from runtime policy, docs, and fallbacks.

---

## Scope (Stories / Tasks)

### Duplex Contract and Transport

- [ ] `src/api/websocket/voice_pkg/router.py` - Keep `/api/v1/voice/ws/duplex` as the canonical realtime route and document compatibility expectations for `/api/v1/voice/ws`.
- [ ] Document and stabilize the current message contract before transport changes:
  - client: `audio_chunk`, `audio_end`, `bargein`, `language_hint`, `close`
  - server: `ready`, `pipeline_state`, `language_detected`, `response_sentence`, `response_audio`, `response_end`, `bargein`, `error`
- [ ] Add one-sprint compatibility handling for legacy JSON base64 audio if binary audio transport is introduced during implementation.
- [ ] Tests: websocket integration coverage for duplex audio input, response streaming, interruption, and language switching.

### Latency Instrumentation and Tuning

- [ ] Add stage timings for VAD start/end, STT duration, first LLM token or sentence, first TTS chunk, first audio sent, and full response completion.
- [ ] Emit Prometheus-friendly metrics and client-visible timing metadata for the live test UI.
- [ ] Retune duplex VAD and interruption thresholds for short farmer utterances and barge-in during playback.
- [ ] Reduce response buffering so TTS can begin on safe partial text boundaries instead of waiting for the full answer whenever possible.
- [ ] Benchmarks: fixed multilingual utterance set for `kn`, `hi`, `te`, and `en`.

### Bedrock Removal and Provider Policy

- [ ] Remove Bedrock from the production provider policy, docs, fallback order, and operator guidance.
- [ ] Standardize the recommended provider order as `groq -> vllm -> together`.
- [ ] Split AWS infrastructure documentation from model-provider guidance so App Runner, Aurora, VPC, and Secrets Manager remain documented without implying Bedrock is required.
- [ ] Tests: config parsing and provider-selection coverage with Bedrock-free defaults.

### Local-Language Voice Quality

- [ ] Benchmark STT and TTS quality for `kn`, `hi`, `te`, and `ta` using a shared internal sample set.
- [ ] Make the local Indic TTS path the primary candidate on warmed GPU workers when it meets quality and latency thresholds.
- [ ] Keep Edge TTS as the degraded fallback when local synthesis fails health or latency checks.
- [ ] Add a small evaluation rubric for naturalness, intelligibility, and interruption recovery.

### Static Live Testing and Handoff

- [ ] Align the supported live-test static pages with the current voice contract:
  - `static/voice_agent.html` as the engineering lab
  - `static/premium_voice.html` as the canonical live demo
- [ ] Retire or redirect unsupported legacy voice demo pages after confirming no active links still depend on them.
- [ ] Keep `WORKFLOW_STATUS.md`, `tracking/PROJECT_STATUS.md`, and the sprint file updated as implementation lands.

---

## Latency Targets

These targets are for stage-level optimization and user-perceived responsiveness, not a misleading "20ms full response" claim.

| Metric | Target | Notes |
|--------|--------|-------|
| Audio chunk cadence | 20ms budget | Media-frame and playback budget only |
| Speech end -> first audio | < 800ms P50 | Short-turn duplex target |
| Speech end -> first audio | < 1200ms P95 | Production target after tuning |
| Full spoken reply | < 2.0s P95 | Short-answer target |
| Barge-in cancel reaction | < 150ms | Includes playback interruption |

---

## Acceptance / Done Criteria

- [ ] Duplex websocket remains the documented source of truth and passes integration tests for normal turn-taking and interruption.
- [ ] Stage timings are visible in logs, metrics, and the live test UI.
- [ ] Bedrock is no longer the intended provider in docs, runtime defaults, or fallback policy.
- [ ] Local-language quality review exists for at least `kn`, `hi`, `te`, and `ta`, with an explicit primary/fallback TTS decision.
- [ ] Static live-test entry points are reduced to the supported pages and aligned with the documented websocket contract.
- [ ] Sprint notes link to benchmark artifacts, known gaps, and follow-up work.

---

## Out of Scope

- No mobile redesign beyond keeping the current voice test surfaces usable.
- No Pipecat-first rewrite of the production voice stack.
- No broad marketplace refactor unrelated to voice latency, provider policy, or live testing.
- No claim that full STT -> LLM -> TTS round-trip can reach 20ms.

---

## Risks / Open Questions

- Available hardware may not support low-latency local Indic TTS for every language without a fallback.
- Existing Pipecat tests still fail and may require cleanup to keep the experimental path honest.
- Binary transport may require coordinated client updates across static assets and any external consumers.
- Some Bedrock references still live outside the core voice path and will need careful removal to avoid config regressions.

---

## Assumptions

- Sprint 06 remains the active sprint while this file prepares the next implementation session.
- The duplex websocket path stays canonical for production, while Pipecat remains experimental.
- AWS infrastructure remains in use after Bedrock removal from model paths.
- The repo keeps one-sprint compatibility for older JSON-plus-base64 clients if the transport is upgraded mid-sprint.

---

## Next Session Execution Checklist

1. Read `tracking/PROJECT_STATUS.md` and this sprint file together.
2. Read `docs/api/websocket-voice.md` and `docs/features/voice-pipeline.md`.
3. Review `src/api/websocket/voice_pkg/router.py`, `src/api/websocket/voice_pkg/duplex.py`, and `src/voice/duplex_pipeline.py`.
4. Open `static/premium_voice.html` and `static/assets/js/voice-agent-duplex.js` before changing the live test flow.
5. Re-run focused voice tests before editing runtime code:
   - `uv run pytest tests/unit/test_voice_agent.py -q`
   - `uv run pytest tests/unit/test_pipecat_pipeline.py -q`

---

## Sprint Outcome (fill at end of sprint)

**What Shipped:**
- 

**What Slipped to Next Sprint:**
- 

**Key Learnings:**
- 

**Voice Benchmark Snapshot:**

| Scenario | First Audio | Full Reply | Notes |
|----------|-------------|------------|-------|
| | | | |

---

## Related Files

- `tracking/PROJECT_STATUS.md`
- `WORKFLOW_STATUS.md`
- `docs/api/websocket-voice.md`
- `docs/features/voice-pipeline.md`
- `docs/api/endpoints-reference.md`
- `src/api/websocket/voice_pkg/router.py`
- `src/api/websocket/voice_pkg/duplex.py`
- `src/voice/duplex_pipeline.py`
- `static/premium_voice.html`
- `static/voice_agent.html`
