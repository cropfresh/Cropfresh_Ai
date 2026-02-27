# Sprint 04 — Voice Pipeline & Production Scraping

> **Period:** 2026-02-27 → 2026-03-12
> **Theme:** Complete Pipecat voice pipeline + APMC real-time data
> **Sprint Status:** 🟡 In Progress

---

## 🎯 Goals (Measurable)

1. Pipecat voice bot (`pipecat_bot.py`) runs end-to-end with live STT and TTS
2. APMC mandi scraper fetches real-time prices for at least 5 major crops / 3 mandis
3. Scraped data lands in Qdrant and is queryable by Commerce Agent
4. All voice endpoints pass unit + integration tests
5. Coverage doesn't drop below 40%

---

## 📋 Scope (Stories / Tasks)

### Voice Pipeline
- [ ] `src/voice/pipecat_bot.py` — Complete Pipecat WebSocket integration with real audio stream
- [ ] `src/voice/pipecat/stt_service.py` — IndicWhisper STT with Groq fallback, test with Kannada samples
- [ ] `src/voice/pipecat/tts_service.py` — Edge-TTS in Kannada, validate output quality
- [ ] `src/api/websocket/voice_ws.py` — WebSocket endpoint stability + reconnection handling
- [ ] Tests: unit tests for `stt_service.py`, `tts_service.py`, integration test for full round-trip

### APMC Scraping
- [ ] `src/scrapers/apmc/apmc_scraper.py` — Scrapling-based scraper for APMC mandi data
- [ ] `src/scrapers/apmc/scheduler.py` — APScheduler cron for auto-refresh (every 6h)
- [ ] `src/scrapers/apmc/cache.py` — Redis cache layer (TTL = 1h for prices)
- [ ] `src/pipelines/data_pipeline.py` — Scrape → normalize → ingest to Qdrant
- [ ] Tests: offline fixture test + validator for scraped data fields

### Documentation
- [ ] Update `docs/api/` for any new or changed voice endpoints
- [ ] Update `WORKFLOW_STATUS.md` with all new file changes
- [ ] Fill in `tracking/daily/` log entries for each session

---

## 🚫 Out of Scope

- Supabase schema migrations (Sprint 05)
- Flutter mobile app (Phase 4)
- Vision Agent (Phase 3)
- Auth/OTP verification (Sprint 05)

---

## ⚠️ Risks / Open Questions

- Pipecat WebSocket stability under load — needs stress test
- APMC website HTML structure may change — monitor with diff tests
- IndicWhisper requires ~2GB RAM — ensure staging environment is configured
- Scrapling Camoufox mode adds latency — benchmark vs basic Playwright mode

---

## 📊 Sprint Outcome (fill at end)

**What Shipped:**
- [ ] (fill at sprint close)

**What Slipped to Sprint 05:**
- [ ] (fill at sprint close)

**Key Learnings:**
- (fill at sprint close)

**Agent Eval Scores This Sprint:**
| Agent | Routing Accuracy | Avg Latency | Notes |
|-------|-----------------|-------------|-------|
| (fill at sprint close) | | | |

---

## 🔗 Related Files

- `src/voice/pipecat_bot.py`
- `src/voice/pipecat/stt_service.py`
- `src/voice/pipecat/tts_service.py`
- `src/api/websocket/voice_ws.py`
- `docs/decisions/ADR-*.md` — record any new decisions here
- `tracking/PROJECT_STATUS.md` — update after sprint closes
