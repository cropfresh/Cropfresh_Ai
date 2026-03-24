# Sprint 11 - Voice Load Hardening and Observability

> **Period:** 2026-06-03 -> 2026-06-16
> **Theme:** Harden the voice stack for concurrency, isolate failures across services, and make latency and stream health measurable under load
> **Sprint Status:** Not Started

---

## Goals (Measurable)

1. Voice services tolerate degradation through bulkheads, circuit breakers, and explicit fallback behavior.
2. Prometheus, Grafana, logs, and traces cover the full voice path from bootstrap through first audio and completion.
3. A k6 suite exercises normal, spike, and sustained load scenarios for the voice stack.
4. Latency, error-rate, and stream-drop SLOs are measurable before deployment work starts.
5. At-least-once delivery is documented and enforced where queued voice jobs are introduced.

---

## Entry Assumptions

- Sprint 10 delivers the first stable multi-service state, routing, and tool boundaries.
- The voice stack is still pre-production and can accept hardening-driven contract changes where they are additive.

---

## Scope (Stories / Tasks)

### Concurrency and Failure Isolation

- [ ] Add service-level bulkheads for STT, orchestration, and TTS workloads.
- [ ] Add circuit breakers and timeout plus retry policy for AI-provider calls.
- [ ] Add queueing or stream-based handoff where at-least-once delivery is required.
- [ ] Define degraded-mode behavior when one subsystem is slow or unavailable.

### Observability Stack

- [ ] Expose Prometheus metrics for active sessions, first audio, full turn time, VAD trigger rate, packet loss, jitter, and RTT where available.
- [ ] Add distributed tracing across gateway, VAD, orchestration, and downstream runtime paths.
- [ ] Add structured JSON logs with correlation ids for every utterance lifecycle.
- [ ] Add dashboards and alert thresholds for latency, error rate, and drop rate.

### Load Testing

- [ ] Create k6 scenarios for:
  - steady normal load
  - 10x spike load
  - 1-hour sustained load
- [ ] Capture P50, P95, and P99 latency plus error rate and stream-drop rate.
- [ ] Record the hardware and environment assumptions used for every benchmark.

---

## First Implementation Slice

1. Add scrapeable voice metrics for session counts and stage timing.
2. Add circuit-breaker behavior around the slowest external calls.
3. Build the first k6 scenario and dashboard pass before tuning under load.
4. Record one baseline load report that future sprints can compare against.

---

## Acceptance / Done Criteria

- [ ] The voice stack exposes a coherent metrics and trace surface across the major services.
- [ ] Load-test artifacts exist for normal, spike, and sustained scenarios.
- [ ] Circuit-breaker and degraded-mode behavior is documented and test-covered.
- [ ] The repo has one clear latency and reliability baseline before Sprint 12 deployment work.

---

## Out of Scope

- No final production cluster cutover in this sprint.
- No full mobile rollout work in this sprint.
- No broad product refactor outside voice-service hardening.

---

## Risks / Open Questions

- Observability can add noise or overhead if too much data is emitted at once.
- Load tests can produce misleading numbers if the benchmark environment is not documented clearly.
- Queueing and delivery guarantees must be scoped carefully so they do not overcomplicate low-latency flows.

---

## Related Files

- `tracking/sprints/sprint-10-voice-orchestration-state-and-tools.md`
- `tracking/sprints/sprint-12-livekit-scale-security-and-deployment.md`
- `docs/features/livekit-voice-bridge.md`
- `infra/monitoring/`
- `docker-compose.yml`
- `tests/`

