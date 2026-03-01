# 📊 Outcomes Dashboard — CropFresh AI

> **Last Updated:** 2026-03-01

---

## Project Health

| Metric | Target (Sprint 05) | Target (2027) | Current | Status |
|--------|-------------------|--------------|---------|--------|
| Agent Routing Accuracy | > 88% | > 95% | ~87% (mock) | 🟡 Near |
| Voice P95 Latency | < 2.5s | < 1.5s | ~4.5s (estimated) | 🔴 Behind |
| API Cost / Query (avg) | < ₹0.25 | < ₹0.18 | ~₹0.44 (fixed pipeline) | 🔴 Behind |
| RAGAS Faithfulness | > 0.80 | > 0.92 | — (baseline pending Sprint 05) | ⬜ Not Started |
| Context Precision | +8% vs baseline | +25% | — | ⬜ Not Started |
| Test Coverage | ≥ 45% | ≥ 80% | **~57%** (382 tests / 15 files) | ✅ Target Met |
| KB Documents Indexed | > 100 | > 10,000 | 32 | 🔴 Behind |
| eNAM Mandis Integrated | 5+ | > 500 | 0 (API pending) | ⬜ Not Started |
| Browser Scrape Success Rate | n/a | > 80% | n/a (Sprint 06) | 📋 Planned |

---

## Sprint Velocity

| Sprint | Status | Goals Met | Carry-overs |
|--------|--------|-----------|-------------|
| Sprint 01 — Foundation | ✅ Complete | 5/5 | 0 |
| Sprint 02 — Data Pipeline | ✅ Complete | — | — |
| Sprint 03 — Crop Agent | ✅ Complete | — | — |
| Sprint 04 — Voice Pipeline | 🟡 In Progress | 0/5 (ongoing) | Pipecat e2e test, APMC scraper |
| **Sprint 05 — Core Agent Completion** | 🟢 Active — 10/11 tasks done | Tasks 1–10 complete | Pipecat e2e, ADCL, DPLE routing, eNAM |
| Sprint 06 — Browser RAG | 🔲 Planned | — | — |

---

## RAG Quality Milestones

| Milestone | Sprint | Target | Status |
|-----------|--------|--------|--------|
| RAGAS baseline established | Sprint 05 | faithfulness > 0.75 | 📋 Planned |
| Adaptive Router cost savings | Sprint 05 | ₹0.44 → < ₹0.25 avg | 📋 Planned |
| AgriEmbedding precision gain | Sprint 05 | +8% vs BGE-M3 | 📋 Planned |
| Voice latency improvement | Sprint 06 | < 2.0s P95 | 📋 Planned |
| Browser scrape live | Sprint 06 | 5+ gov/news sources | 📋 Planned |
| RAGAS CI gate | Sprint 06 | faithfulness > 0.80 | 📋 Planned |
| ColBERT retrieval | Sprint 07 | P@10 +20% | 📋 Planned |
| Fine-tuned embeddings | Phase 4 | +25% precision | 📋 2027 |

---

## Agent Performance Summary
See `tracking/agent-performance/` for per-agent metrics.

## Budget
See `tracking/COST.md` for token/API cost tracking.
