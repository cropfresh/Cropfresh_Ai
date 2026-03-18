# CropFresh AI - Task Backlog

> **Priority:** P0 (critical) -> P1 (high) -> P2 (medium) -> P3 (nice-to-have)
> **Last Updated:** 2026-03-18

---

## P0 - Critical (Sprint 06 carryover)

- [ ] Historical ADCL backtest artifact using a real district order snapshot
- [ ] ADCL golden-set review against Aurora-backed live data
- [ ] eNAM credential validation for gated ADCL and rate-hub paths

## P1 - High Priority (Sprint 08 - LiveKit Voice Bridge Foundation)

- [x] Create `services/voice-gateway/` with `POST /sessions/bootstrap`, `GET /health`, and `GET /ready`
- [x] Add feature-flagged LiveKit session bootstrap while preserving fallback to `/api/v1/voice/ws/duplex`
- [ ] Implement a 5-second PCM ring buffer and RMS pre-gate in the gateway before downstream relay
- [x] Create `services/vad-service/` with FastAPI plus gRPC, Silero ONNX acoustic segmentation, and the agreed dual-threshold settings
- [ ] Keep the current Groq plus Edge/local Indic provider path in the downstream FastAPI duplex runtime for Phase 1
- [x] Extend the existing static voice pages with a minimal bridge bootstrap flow and visible fallback mode
- [x] Add focused unit/integration tests for bootstrap, fallback, and VAD segmentation behavior

## P2 - Medium Priority (Sprint 09 - Semantic VAD and Recovery)

- [ ] Add stage-level timing across the bridge and downstream duplex path
- [ ] Create the fixed multilingual voice benchmark set for `kn`, `hi`, `te`, and `ta`
- [ ] Add semantic endpointing, stronger barge-in continuity, and smarter interruption handling
- [ ] Add reconnect recovery, Redis-backed last-10-turn state, and heartbeat/dead-peer handling

## P2 - Medium Priority (Sprint 10 - Orchestration and Tooling)

- [ ] Add the distributed voice state machine over Redis pub/sub
- [ ] Add router plus specialist-agent voice orchestration for price, listing, logistics, and fallback flows
- [ ] Reuse the shared memory contract for voice turns and reconnect recovery
- [ ] Add first-pass speaker diarization and tool-routing coverage

## P3 - Planned Hardening (Sprint 11)

- [ ] Add bulkheads, circuit breakers, queue/stream guarantees, and degraded-mode behavior across voice services
- [ ] Add Prometheus, Grafana, logs, and traces for the voice path
- [ ] Add k6 scenarios for normal, spike, and sustained load

## P3 - Planned Deployment (Sprint 12)

- [ ] Add Kubernetes deployment assets for the voice services, LiveKit, and supporting infrastructure
- [ ] Add consent, audit-log, JWT, PII-scrubbing, and data-residency controls
- [ ] Add deployment, rollback, and cutover-readiness runbooks

## P3 - Backlog After Sprint 12

- [ ] Supabase/auth hardening follow-up after the voice program
- [ ] Expand rate-hub health monitoring and source diagnostics
- [ ] Broaden Karnataka rate-hub coverage beyond the initial query tuples and schedules
- [ ] Crop listings and order workflow hardening against the real database
- [ ] Buyer matching improvements (GPS plus buyer-preference signals)

## P3 - Nice-to-Have (Phase 3+)

- [ ] Speculative RAG for later voice latency experiments
- [ ] WhatsApp integration (Business API)
- [ ] Browser RAG (14+ ag-specific gov/news sources)
- [ ] Rate limiter for production
- [ ] Circuit breaker for LLM failover
- [ ] Vision Agent (YOLOv12 + DINOv2 crop grading)
- [ ] Flutter mobile app
- [ ] Stripe/Razorpay payment integration
- [ ] Push notifications
- [ ] Multi-region deployment
- [ ] Fine-tuned agri embeddings (Layer 2)
- [ ] A/B testing framework for agent prompts
