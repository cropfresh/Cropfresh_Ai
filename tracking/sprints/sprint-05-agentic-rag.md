# Sprint 05 — Agentic RAG & Adaptive Intelligence

> **Period:** 2026-03-13 → 2026-03-26
> **Theme:** Upgrade RAG pipeline to agentic + adaptive architecture for 2027 competitiveness
> **Sprint Status:** 🔲 Not Started

---

## 🎯 Goals (Measurable)

1. **Adaptive Query Router** ships — 8 routing strategies with measurable cost reduction (target: avg ₹0.44 → ₹0.22/query)
2. **AgriEmbeddingWrapper (Layer 1)** live — BGE-M3 wrapped with domain instruction prefix + 50-term bilingual normalization map
3. **Agentic RAG Orchestrator (v1)** deployed — Retrieval Planner using Groq 8B, parallel tool execution working
4. **RAGAS Baseline** established — 20-query golden evaluation dataset with baseline faithfulness + context precision scores
5. **eNAM API registration** complete — API access approved and integrated into `LIVE_PRICE_API` strategy
6. All new modules have unit tests; coverage remains ≥ 40%

---

## 📋 Scope (Stories / Tasks)

### 1. Adaptive Query Router (8-Strategy)

- [ ] `ai/rag/query_analyzer.py` — Add `RetrievalStrategy` enum (8 strategies) and `RoutingDecision` Pydantic model
- [ ] `ai/rag/query_analyzer.py` — Add `AdaptiveQueryRouter` class with `_prefilter()` (rule-based, 0ms) and `_llm_classify()` (Groq 8B)
- [ ] Feature flag: `USE_ADAPTIVE_ROUTER` in `.env` with default `false` (safe rollout)
- [ ] `scripts/test_adaptive_router.py` — 30 test queries covering all 8 strategy classifications
- [ ] Instrument routing decisions in LangSmith with strategy + cost signal tag
- [ ] **Metric**: Run 50 real queries, compare routing classification vs. expected (target: >85% correct)

### 2. AgriEmbedding Layer 1 — Domain Wrapper

- [ ] `ai/rag/agri_embeddings.py` — `AgriEmbeddingWrapper` class (wraps EmbeddingManager)
- [ ] Implement `AGRI_QUERY_INSTRUCTION` prefix, bilingual `TERM_MAP` (50+ entries: Hindi/Kannada → English)
- [ ] `AgriEmbeddingWrapper.embed_query()`, `embed_documents()`, `_normalize_terms()`
- [ ] Wire into `KnowledgeBase` and `RAGRetriever` (replace `EmbeddingManager` calls where appropriate)
- [ ] `scripts/evaluate_agri_embeddings.py` — Compare retrieval precision: base BGE-M3 vs. AgriWrapper on 20-query golden set
- [ ] **Metric**: Document baseline precision before and after on golden dataset

### 3. Agentic RAG Orchestrator (v1 — Planner Only)

- [ ] `ai/rag/agentic_orchestrator.py` — `AgenticOrchestrator` class with `plan_retrieval()` and `execute_plan()`
- [ ] Implement `RetrievalPlanner` using Groq `llama-3.1-8b-instant` — JSON plan output with tool calls + `can_parallelize` flag
- [ ] Wire 4 tools in executor: `vector_search`, `graph_rag`, `price_api` (eNAM), `direct_llm`
- [ ] `AgenticSelfEvaluator` — lightweight RAGAS faithfulness + relevance scorer; retry if confidence < 0.75 (max 2 retries)
- [ ] `scripts/test_agentic_rag.py` — End-to-end test: simple → DIRECT_LLM path; complex → full orchestrator path
- [ ] **Metric**: Compare answer quality (RAGAS faithfulness) orchestrator vs. fixed 4-node pipeline

### 4. RAGAS Evaluation Baseline

- [ ] `scripts/create_golden_dataset.py` — Build 20-query golden evaluation set covering: agronomy (8Q), market (4Q), schemes (3Q), pest/disease (5Q)
- [ ] `ai/rag/evaluation.py` — Wire RAGAS metrics: `faithfulness`, `context_precision`, `context_recall`, `answer_relevancy`
- [ ] Run baseline evaluation on existing fixed pipeline → save scores to `tracking/agent-performance/rag-baseline-2026-03-13.json`
- [ ] Compare post-Sprint 05 scores to baseline

### 5. eNAM API Integration

- [ ] Register at [enam.gov.in](https://enam.gov.in) for API access (manual step, submit by March 13)
- [ ] `src/scrapers/enam_client.py` — Update with real API credentials once approved (mock fallback until then)
- [ ] Wire `LIVE_PRICE_API` strategy in Adaptive Router to call `enam_client.get_prices(commodity, mandi)`

### 6. Documentation

- [ ] Update `tracking/daily/` for each session during sprint
- [ ] Update `WORKFLOW_STATUS.md` with file changes
- [ ] Create `tracking/sprints/sprint-05-outcome.md` at sprint close

---

## 🚫 Out of Scope

- Speculative Draft Engine (Sprint 06 — requires Orchestrator v1 first)
- Browser-Augmented RAG scraping (Sprint 06)
- Fine-tuned agri embeddings / model training (Phase 4 / 2027)
- ColBERT late-interaction retriever (Sprint 07)
- Flutter mobile app, Supabase schema migrations

---

## ⚠️ Risks / Open Questions

- **eNAM API approval timeline**: Government APIs can take 2–4 weeks. Fallback: use existing `agmarknet.py` scraper as LIVE_PRICE_API source until eNAM approves.
- **Groq rate limits for Adaptive Router**: At 100 req/s, router calls (each ~80ms) may hit limits. Implement local rule-based cache for repeated queries in same session.
- **RAGAS dependencies**: `ragas` package requires `openai` — budget for 20 evaluation calls (~₹5). Alternative: use `langchain-evaluation` with Groq if OpenAI budget unavailable.
- **AgriEmbeddingWrapper dimension consistency**: New wrapper must produce same 1024-dim vectors as base BGE-M3 (it will — wrapping doesn't change model outputs, only input string).

---

## 📊 Sprint Outcome (fill at end)

**What Shipped:**
- [ ] (fill at sprint close)

**What Slipped to Sprint 06:**
- [ ] (fill at sprint close)

**Key Learnings:**
- (fill at sprint close)

**RAG Metrics (Before vs. After):**
| Metric | Before Sprint 05 | After Sprint 05 | Δ |
|--------|-----------------|-----------------|---|
| Avg cost/query | ₹0.44 | TBD | TBD |
| RAGAS faithfulness | — (no baseline) | TBD | TBD |
| Context precision | — (no baseline) | TBD | TBD |
| Voice P95 latency | ~4.5s | TBD | TBD |

---

## 🔗 Related Files

**New files this sprint:**
- `ai/rag/agri_embeddings.py` — AgriEmbeddingWrapper
- `ai/rag/agentic_orchestrator.py` — Retrieval Planner + Self-Evaluator
- `ai/rag/query_analyzer.py` — Extended with AdaptiveQueryRouter
- `scripts/test_adaptive_router.py`
- `scripts/test_agentic_rag.py`
- `scripts/create_golden_dataset.py`
- `scripts/evaluate_agri_embeddings.py`

**Reference:**
- [ADR-007: Agentic RAG Orchestrator](../../docs/decisions/ADR-007-agentic-rag-orchestrator.md)
- [ADR-008: Adaptive Query Router](../../docs/decisions/ADR-008-adaptive-query-router.md)
- [ADR-009: Agri Embeddings](../../docs/decisions/ADR-009-agri-embeddings.md)
- [Agentic RAG Architecture](../../docs/architecture/agentic_rag_system.md)
- [Adaptive Router Architecture](../../docs/architecture/adaptive_query_router.md)
- [RAG 2027 Research Report](../../rag_2027_research_report.md)
- `tracking/PROJECT_STATUS.md` — update after sprint closes
