# CropFresh AI - Task Backlog

> **Priority:** P0 (critical) -> P1 (high) -> P2 (medium) -> P3 (nice-to-have)
> **Last Updated:** 2026-03-17

---

## P0 - Critical (Sprint 06 carryover)

- [ ] Historical ADCL backtest artifact using a real district order snapshot
- [ ] ADCL golden-set review against Aurora-backed live data
- [ ] eNAM credential validation for gated ADCL and rate-hub paths

## P1 - High Priority (Sprint 07 - Voice Duplex Productionization)

- [ ] Canonicalize `/api/v1/voice/ws/duplex` as the production realtime interface
- [ ] Add stage-level timing for VAD, STT, first LLM output, first TTS chunk, first audio sent, and full response completion
- [ ] Tune duplex VAD and interruption thresholds for short farmer utterances and barge-in
- [ ] Reduce duplex response buffering so TTS can start on safe partial text boundaries
- [ ] Remove Bedrock from recommended provider policy, docs, fallbacks, and runtime defaults
- [ ] Standardize the provider order as `groq -> vllm -> together`
- [ ] Benchmark STT/TTS quality for `kn`, `hi`, `te`, and `ta`
- [ ] Lock the production TTS fallback strategy for local-language voices
- [ ] Consolidate static voice test pages around `static/voice_agent.html` and `static/premium_voice.html`
- [ ] Fix the focused Pipecat test slice enough to keep the experimental path honest

## P2 - Medium Priority (Sprint 08 follow-up)

- [ ] Supabase/auth hardening follow-up after Sprint 07
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
