# Sprint 12 - LiveKit Scale, Security, and Deployment

> **Period:** 2026-06-17 -> 2026-06-30
> **Theme:** Convert the hardened voice stack into a deployment-ready clustered system with security, compliance, and operational runbooks
> **Sprint Status:** Not Started

---

## Goals (Measurable)

1. The repo contains deployment-ready manifests for clustered LiveKit, supporting services, and safe rolling updates.
2. Security and compliance controls are in place for realtime voice traffic, transcripts, and session metadata.
3. Production-readiness runbooks exist for cutover, rollback, and incident response.
4. India-region deployment assumptions are explicit for latency and data-residency planning.
5. The voice program ends with one honest readiness snapshot instead of a vague “production ready” claim.

---

## Entry Assumptions

- Sprint 11 delivers baseline performance, failure handling, dashboards, and load-test artifacts.
- The voice bridge path is stable enough to be the deployment target candidate.

---

## Scope (Stories / Tasks)

### Cluster and Deployment Assets

- [ ] Add Kubernetes manifests for the voice gateway, VAD service, orchestration service, LiveKit, Redis, and supporting dependencies.
- [ ] Use HPA, PodDisruptionBudgets, and affinity rules where needed for zero-downtime behavior.
- [ ] Define ingress, TLS, and certificate management for the exposed voice surfaces.
- [ ] Add environment and secret wiring guidance for deployment targets.

### Security and Compliance

- [ ] Document and wire JWT room-token issuance and expiry behavior.
- [ ] Add transcript-side PII scrubbing for phone, bank, and Aadhaar-like data where relevant.
- [ ] Add consent, audit-log, and session-metadata capture requirements.
- [ ] Keep India-region deployment and data-residency assumptions explicit.

### Runbooks and Cutover Planning

- [ ] Create deployment, rollback, and incident-response runbooks.
- [ ] Define the bridge-to-canonical cutover criteria explicitly instead of assuming it.
- [ ] Capture final known gaps, deferred work, and readiness blockers.

---

## First Implementation Slice

1. Add deployment manifests and environment contracts for the core voice services.
2. Add token, consent, and audit-log policy docs before claiming deployment readiness.
3. Draft the cutover and rollback runbooks from the Sprint 11 load and reliability data.

---

## Acceptance / Done Criteria

- [ ] Cluster and deployment manifests exist for the main voice services.
- [ ] Security and compliance controls are documented and reflected in the deployment plan.
- [ ] Runbooks exist for deployment, rollback, and incident triage.
- [ ] The sprint outcome records whether the bridge path is ready for cutover, still needs an intermediate phase, or should remain behind a flag.

---

## Out of Scope

- No claim that mobile clients are fully production-ready.
- No claim that every marketplace feature is voice-complete.
- No unrelated marketplace hardening outside deployment-critical dependencies.

---

## Risks / Open Questions

- Deployment manifests can drift quickly if service boundaries continue moving late in the program.
- Compliance requirements may expand once real transcript handling and retention policy are finalized.
- Cutover readiness depends on Sprint 11 benchmark quality; weak benchmarks will weaken deployment confidence.

---

## Related Files

- `tracking/sprints/sprint-11-voice-load-hardening-and-observability.md`
- `docs/features/livekit-voice-bridge.md`
- `ROADMAP.md`
- `infra/`
- `docker-compose.yml`
- `tracking/PROJECT_STATUS.md`
