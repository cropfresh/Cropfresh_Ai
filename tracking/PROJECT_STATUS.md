# 📊 PROJECT_STATUS.md — CropFresh AI Current State

> **Last Updated:** 2026-02-27 (15:01 IST)
> **Version/Tag:** `v0.3-foundation`
> **Current Sprint:** Sprint 04 — Voice Pipeline & Scraping
> **Sprint Status:** 🟡 In Progress

---

## 🟢 What Is True Right Now

| Component | Status | Notes |
|-----------|--------|-------|
| Project Structure | ✅ Stable | Advanced folder layout, all modules organized |
| RAG Pipeline | ✅ Stable | RAPTOR + Hybrid Search + Qdrant Cloud |
| Multi-Agent System | ✅ Stable | Supervisor routes with 0.9 confidence |
| Memory System | ✅ Stable | Redis-backed session manager |
| Voice Agent | 🟡 In Progress | Pipecat integration underway, STT/TTS services built |
| APMC Scraper | ❌ Not Started | Scrapling-based scraper planned |
| Vision Agent | ❌ Not Started | YOLOv12 + DINOv2, planned for Phase 3 |
| Supabase Schema | ❌ Not Started | Schema migrations pending |
| Evaluation Framework | ❌ Not Started | LangSmith setup planned |

---

## 🎯 Top 3 Priorities (This Week)

1. **Complete Pipecat voice pipeline** — `src/voice/pipecat_bot.py` integration with real STT/TTS
2. **APMC mandi scraper** — Scrapling-based real-time price data pipeline
3. **Sprint 04 review & Sprint 05 plan** — Close current sprint, plan next with realistic scope

---

## 🚧 Active Blockers

- [ ] `pipecat_bot.py` — Pipecat WebSocket integration needs testing with live audio
- [ ] `APMC_SCRAPER` — No live APMC API; scraping strategy needs validation with Scrapling
- [ ] `Supabase` — Schema not yet migrated; farmer/listing data flows blocked

---

## ✅ Recently Completed Milestones

| Date | Milestone |
|------|-----------|
| Feb 27, 2026 | Advanced folder restructure complete |
| Feb 27, 2026 | Production-grade scraping upgrade (Scrapling + caching + scheduling) |
| Feb 26, 2026 | Voice domain separated into own module (pipecat/) |
| Feb 26, 2026 | Domain separation: RAG, Vision, Voice as distinct domains |
| Jan 10, 2026 | Advanced RAG Phase 1–4 complete (RAPTOR, Hybrid, HyDE, MMR) |
| Jan 9, 2026 | Multi-agent system live (Supervisor + 4 domain agents) |
| Jan 9, 2026 | Voice agent v1 (Edge-TTS + Whisper STT) |

---

## 📦 Stack Reference

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Python 3.11+ |
| AI Framework | LangGraph + LangChain |
| Vector DB | Qdrant Cloud |
| Graph DB | Neo4j |
| Primary DB | Supabase (PostgreSQL) |
| LLM | Groq (Llama/Mixtral) + Gemini Flash |
| Scraping | Scrapling (Playwright + Camoufox) |
| Voice | Pipecat + Edge-TTS + IndicWhisper |
| Caching | Redis |
| Package Manager | uv |

---

## 📁 Key File Locations

| Purpose | Path |
|---------|------|
| Product Vision | `PLAN.md` |
| Phase Milestones | `ROADMAP.md` |
| Current Sprint | `tracking/sprints/sprint-04-voice-pipeline.md` |
| Daily Logs | `tracking/daily/YYYY-MM-DD.md` |
| Architecture Decisions | `docs/decisions/ADR-*.md` |
| Agent Registry | `docs/agents/REGISTRY.md` |
| Test Strategy | `TESTING/STRATEGY.md` |
| AI Agent Instructions | `AGENTS.md` |

---

## 📈 Lightweight Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Agent routing accuracy | > 90% | ~87% (mock eval) |
| Voice round-trip latency | < 3s | TBD (Pipecat testing) |
| API cost per query | < ₹0.50 | ~₹0.20 (Groq) |
| Knowledge base documents | > 50 | 32 indexed |
| Test coverage | > 60% | ~35% |

---

_Update this file at the end of every sprint and after major milestones. AI agents should read this first before starting any work._
