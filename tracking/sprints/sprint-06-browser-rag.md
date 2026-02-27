# Sprint 06 — Browser RAG & Speculative Generation

> **Period:** 2026-03-27 → 2026-04-09
> **Theme:** Live web retrieval + speculative parallel generation for voice latency target
> **Sprint Status:** 🔲 Not Started
> **Depends on:** Sprint 05 (Adaptive Router, Agentic Orchestrator v1)

---

## 🎯 Goals (Measurable)

1. **Browser-Augmented RAG** ships — `BROWSER_SCRAPE` strategy live, scraping ≥ 5 gov/news sources with circuit breaker + TTL cache
2. **Speculative Draft Engine** ships — 3 parallel Groq 8B drafts + Gemini Flash verifier; voice P95 latency < 2.5s (target: < 2.0s)
3. **RAGAS Evaluation** automated in CI — RAGAS scores computed on golden dataset after every sprint; faithfulness > 0.80
4. **Auto GraphRAG** (v1) — LLM-powered auto-extraction of entities from newly ingested documents into Neo4j
5. Browser-scraped answers include proper citations (source URL + freshness timestamp)
6. Test coverage ≥ 45%

---

## 📋 Scope (Stories / Tasks)

### 1. Browser-Augmented RAG

- [ ] `ai/rag/browser_rag.py` — `BrowserRAGIntegration` class (integrates with ScraplingBaseScraper)
- [ ] `ai/rag/browser_rag.py` — `AgriSourceSelector.select_sources(query)` — maps query intent to target URLs
- [ ] `ai/rag/browser_rag.py` — `ContentExtractor` — Scrapling CSS/XPath selectors for each registered domain
- [ ] `ai/rag/browser_rag.py` — `QualityFilter` — min 150 words, no-error-page pattern check
- [ ] `ai/rag/browser_rag.py` — `LiveDocBuilder` — converts scraped text → `Document` objects with TTL metadata
- [ ] Qdrant collection `live_web_cache` — schema with `expires_at` field; APScheduler purge job every 30min
- [ ] Wire `BROWSER_SCRAPE` strategy from `AdaptiveQueryRouter` → `BrowserRAGIntegration.retrieve_live()`
- [ ] `scripts/test_browser_rag.py` — test scraping 3 sources (icar.org.in, krishijagran.com, imd.gov.in)
- [ ] **Metric**: Successfully scrape 5 sources; scraped content appears in RAG answers with citations

### 2. Speculative Draft Engine

- [ ] `ai/rag/speculative_engine.py` — `SpeculativeDraftEngine` class
- [ ] `split_into_subsets(documents, n=3)` — partition retrieved docs into 3 equal subsets
- [ ] `generate_drafts_parallel()` — `asyncio.gather()` for 3 simultaneous Groq 8B drafter calls
- [ ] `VerifierLLM.select_best_draft()` — Gemini Flash 2.0 selects the most accurate/complete draft
- [ ] Wire into `AgenticOrchestrator` — replaces sequential generation for `FULL_AGENTIC` + `BROWSER_SCRAPE` paths
- [ ] Benchmark voice end-to-end latency: sequential generation vs. speculative (target: < 2.0s P95)
- [ ] **Metric**: Measure and log P50/P95 voice round-trip latency before and after

### 3. RAGAS Continuous Evaluation

- [ ] `.github/workflows/rag_eval.yml` — GitHub Action running golden dataset evaluation on every push to `main`
- [ ] `scripts/run_rag_eval.py` — loads golden dataset, runs RAGAS, saves results to `tracking/agent-performance/`
- [ ] RAGAS score thresholds as CI gates: faithfulness < 0.75 → fail build
- [ ] Dashboard update in `tracking/OUTCOMES.md` after each run
- [ ] **Metric**: RAGAS faithfulness > 0.80 on 20-query golden dataset

### 4. Auto GraphRAG (v1)

- [ ] `ai/rag/graph_constructor.py` — `AutoEntityExtractor.extract_from_doc(document)` → `(entities, relations)` → Neo4j upsert
- [ ] Trigger auto-extraction on each new document ingested via `/api/v1/ingest`
- [ ] `scripts/test_auto_graph.py` — ingest 3 test documents, verify entities appear in Neo4j

### 5. Citation & Freshness Layer

- [ ] `ai/rag/browser_rag.py` — `CitationBuilder` — produces `CitedAnswer` with source URL + freshness label
- [ ] Update LLM generation prompts to include citation format: `"Based on [source] (retrieved Xh ago)..."`
- [ ] Voice output: citations read as "Source: ICAR website, retrieved 2 hours ago"

### 6. Documentation

- [ ] Create `tracking/daily/` entries each session
- [ ] Update sprint-06 outcome section at sprint close
- [ ] Update `docs/architecture/browser_scraping_rag.md` with any implementation changes
- [ ] Update `tracking/PROJECT_STATUS.md`

---

## 🚫 Out of Scope

- ColBERT late-interaction retriever (Sprint 07)
- ColPali multimodal PDF indexing (Sprint 07)
- Fine-tuned embedding model training (Phase 4)
- Flutter mobile integration (Phase 4)

---

## ⚠️ Risks / Open Questions

- **Anti-bot detection on gov.in sites**: icar.org.in and pmkisan.gov.in may block automated scraping. Have `StealthyFetcher` ready; track block rate in circuit breaker metrics.
- **Speculative draft cost**: 3 drafter calls × average 400 tokens = +₹0.06/query for `FULL_AGENTIC` path. Verify that latency savings justify cost on voice queries specifically.
- **RAGAS CI cost**: ~20 RAGAS evaluations × Groq API = ~₹0.50/run. Gate expensive evaluations to nightly runs only, not every PR.
- **Gemini Flash verifier latency**: verifier selection adds ~300ms. Benchmark to confirm net latency is still below 2.0s target.

---

## 📊 Sprint Outcome (fill at end)

**What Shipped:**
- [ ] (fill at sprint close)

**What Slipped to Sprint 07:**
- [ ] (fill at sprint close)

**Key Learnings:**
- (fill at sprint close)

**Performance Targets:**
| Metric | Sprint 05 End | Sprint 06 Target | Achieved |
|--------|--------------|-----------------|---------|
| Voice P95 latency | TBD | < 2.0s | TBD |
| RAGAS faithfulness | TBD | > 0.80 | TBD |
| Avg cost/query | TBD | < ₹0.22 | TBD |
| Browser scrape success rate | 0% | > 80% | TBD |

---

## 🔗 Related Files

**New files this sprint:**
- `ai/rag/browser_rag.py` — BrowserRAGIntegration + SourceSelector + CitationBuilder
- `ai/rag/speculative_engine.py` — SpeculativeDraftEngine
- `scripts/test_browser_rag.py`
- `scripts/run_rag_eval.py`
- `scripts/test_auto_graph.py`

**Reference:**
- [ADR-010: Browser RAG](../../docs/decisions/ADR-010-browser-scraping-rag.md)
- [Browser RAG Architecture](../../docs/architecture/browser_scraping_rag.md)
- [Agentic RAG System](../../docs/architecture/agentic_rag_system.md)
- [Sprint 05](./sprint-05-agentic-rag.md)
- `tracking/PROJECT_STATUS.md` — update after sprint closes
