# CropFresh AI - Task Backlog

> **Priority:** P0 (critical) -> P1 (high) -> P2 (medium) -> P3 (nice-to-have)
> **Last Updated:** 2026-03-17

---

## P0 - Critical (Sprint 05)

- [ ] AgriEmbeddingWrapper with domain prefix and Hindi/Kannada normalization
- [ ] Adaptive Query Router (8 strategies, cost optimization)
- [ ] RAGAS evaluation baseline (20 golden queries)
- [ ] Multi-source rate-hub live-source validation and smoke tests
- [ ] Multi-source rate-hub CI hardening for repo-wide Ruff and mypy blockers
- [ ] eNAM API registration

## P1 - High Priority (Sprint 06)

- [x] Canonical `ADCLService` contract and compatibility shims for REST, wrapper, listings, and voice
- [x] Aurora repo methods and `adcl_reports` schema updates for `(week_start, district)` persistence
- [x] Remove mock-order and mock live-source runtime fallbacks from production ADCL paths
- [x] Live data wiring for internal demand, shared rate hub, IMD/Agromet, and gated eNAM support
- [x] `GET /api/v1/adcl/weekly` plus app-state wiring for shared ADCL, listing, and voice services
- [x] APScheduler refresh jobs, structured logs, metrics, freshness tracking, and source-health reporting
- [ ] Historical backtest artifact and 20-query ADCL golden set

## P2 - Medium Priority (Sprint 07)

- [ ] Supabase/auth hardening follow-up after Sprint 06
- [ ] Expand rate-hub health monitoring and source diagnostics
- [ ] Broaden Karnataka rate-hub coverage beyond the initial query tuples and schedules
- [ ] Crop listings and order workflow hardening against the real database
- [ ] Buyer matching improvements (GPS plus buyer-preference signals)

## P3 - Nice-to-Have (Phase 3+)

- [ ] Speculative RAG for voice latency reduction
- [ ] Voice pipeline optimization (target: <2s)
- [ ] Pipecat WebRTC full integration
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
