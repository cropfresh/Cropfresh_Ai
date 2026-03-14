# Advanced Agentic RAG System — Implementation Plan

> **Created:** 2026-03-14
> **Status:** Approved for Implementation
> **Sprint:** Sprint 05
> **Goal:** Upgrade CropFresh AI's RAG pipeline to a production-grade agentic system with zero-hallucination guarantees, self-corrective loops, and automated quality gates.

---

## 1. Current State Assessment

After analyzing all 30+ files in `ai/rag/`, the current system has:

| Component | File | Status |
|-----------|------|--------|
| Qdrant vector store | `ai/rag/knowledge_base.py` | ✅ Working |
| RAPTOR tree indexing | `ai/rag/raptor.py` (815 lines) | ✅ Working |
| BM25 hybrid search | `ai/rag/hybrid_search.py` | ✅ Working |
| 8-Strategy adaptive query router | `ai/rag/query_analyzer.py` | ✅ Working |
| Cross-encoder reranker | `ai/rag/reranker.py` | ✅ Working |
| Cohere advanced reranker | `ai/rag/advanced_reranker.py` | ⚠️ Ensemble stub |
| CRAG document grader | `ai/rag/grader.py` | ✅ Working |
| Hallucination checker | `ai/rag/grader.py` (HallucinationChecker) | ✅ Working |
| Agentic orchestrator | `ai/rag/agentic/orchestrator.py` | ✅ Working |
| Speculative draft engine | `ai/rag/agentic/speculative.py` | ✅ Working |
| Self-evaluator | `ai/rag/agentic/evaluator.py` | ✅ Working |
| Retrieval planner | `ai/rag/agentic/planner.py` | ✅ Working |
| Graph retriever (Neo4j) | `ai/rag/graph_retriever.py` | ✅ Working |
| LangSmith observability | `ai/rag/observability.py` | ✅ Working |
| AgriEmbedding wrapper | `ai/rag/agri_embeddings.py` | ✅ Working |
| RAGAS evaluation | `ai/rag/evaluation.py` | ⚠️ Mock only |
| Contextual chunker | `ai/rag/contextual_chunker.py` | ❌ Empty stub |

### Critical Gaps

| Gap | Impact | Priority |
|-----|--------|----------|
| No query rewriting / HyDE | Poor recall for vague queries | 🔴 P0 |
| No source citations in answers | Farmers can't trust advice | 🔴 P0 |
| No "I don't know" confidence gate | Hallucinated answers on unknown topics | 🔴 P0 |
| Mock RAGAS evaluation | No real quality measurement | 🔴 P0 |
| No LangGraph state machine | Imperative loop instead of declarative graph | 🟡 P1 |
| No contextual chunking | Chunks lose document-level context | 🟡 P1 |
| No parent-child retrieval | Insufficient context for LLM | 🟡 P1 |
| No multi-hop reasoning | Complex queries get poor answers | 🟡 P1 |
| No Kannada/Hindi router keywords | English-only rule-based pre-filter | 🟢 P2 |

---

## 2. AWS Compatibility

> **No infrastructure changes required.** All improvements are pure Python code within `ai/rag/`.

| AWS Service | Already Configured | Impact |
|-------------|-------------------|--------|
| App Runner | ✅ `xjivm4x5cn.ap-south-1.awsapprunner.com` | Auto-redeploys on ECR push |
| ECR | ✅ `cropfresh-ai:latest` | Docker image copies `ai/` folder |
| Secrets Manager | ✅ 8 secrets (Groq, Qdrant, Neo4j, LangSmith) | No new secrets needed |
| GitHub Actions | ✅ `deploy-aws.yml` | Builds + pushes automatically |
| RDS PostgreSQL | ✅ pgvector enabled | No changes |

**Deployment flow (unchanged):**
```
git push main → GitHub Actions → Docker build → ECR push → App Runner auto-deploy
```

---

## 3. Proposed Changes

### Phase 1: Anti-Hallucination Pipeline (🔴 P0)

The highest-impact changes — eliminating hallucinations and ensuring every answer is grounded.

---

#### [NEW] `ai/rag/query_rewriter.py`

**Query Rewriting + HyDE (Hypothetical Document Embeddings)**

Three rewriting strategies:

| Strategy | When Used | How It Works |
|----------|-----------|-------------|
| **Step-back prompting** | Specific queries ("tomato leaf curl in Kolar") | Reformulate to broader query first, then retrieve |
| **HyDE** | Vague queries ("crop problem") | Generate hypothetical answer → embed it → retrieve |
| **Multi-query expansion** | Ambiguous queries | Generate 3 diverse reformulations → retrieve all → merge |

```python
class QueryRewriter:
    async def rewrite(query: str, strategy: str = "auto") -> list[str]
    async def generate_hyde(query: str) -> str
```

- Uses Groq Llama-3.1-8B-Instant (~₹0.001 per call)
- Falls back to original query if LLM fails

---

#### [NEW] `ai/rag/citation_engine.py`

**Source Attribution & Citation System**

Every answer includes inline `[1]`, `[2]` citation markers with a Sources section:

```
Tomato leaf curl is caused by whitefly-transmitted viruses [1].
Apply neem oil spray every 7 days as preventive measure [2].

Sources:
[1] IIHR Tomato Disease Guide, 2025
[2] KVK Kolar Pest Management Advisory
```

```python
class CitedAnswer(BaseModel):
    answer: str           # Answer with [1], [2] markers
    sources: list[Source]  # Source documents with labels
    citation_count: int
    all_verified: bool     # True if every citation maps to a doc

class CitationEngine:
    async def add_citations(answer: str, documents: list) -> CitedAnswer
    async def verify_citations(cited_answer: CitedAnswer) -> bool
```

---

#### [NEW] `ai/rag/confidence_gate.py`

**"I Don't Know" Fallback with Safety Classification**

| Query Category | Confidence Threshold | Decline Response |
|---------------|---------------------|-----------------|
| Safety-critical (pesticide doses, loans) | ≥ 0.85 | "Please consult your local KVK or agriculture officer." |
| General agronomy | ≥ 0.70 | "I don't have enough information about this." |
| Platform FAQ | ≥ 0.60 | Standard disclaimer |

```python
class ConfidenceGate:
    async def classify_safety(query: str) -> SafetyLevel
    async def check_grounding(answer: str, docs: list) -> float
    async def gate(query: str, answer: str, docs: list, eval_gate: EvalGate) -> GatedAnswer
```

- Extends existing `HallucinationChecker` in `grader.py`
- Prevents dangerous advice (wrong pesticide dosage, etc.)

---

#### [MODIFY] `ai/rag/grader.py`

**Enhance CRAG Document Grader**

Changes:
- Continuous relevance scoring (0–1) instead of binary yes/no
- Category-aware grading: market price docs must be recent, agronomy docs can be older
- Time-decay penalty: documents older than 7 days for price queries get `score × 0.5`
- Integration with `ConfidenceGate`

---

### Phase 2: Advanced Retrieval Strategies (🟡 P1)

---

#### [NEW] `ai/rag/contextual_chunker_v2.py`

**Anthropic-style Contextual Chunking** (replaces empty `contextual_chunker.py`)

- Prepend `<context>` header to every chunk before embedding:
  ```
  <context>This chunk is from an IIHR guide about tomato diseases in Karnataka.
  It covers whitefly-transmitted viral infections.</context>
  Tomato leaf curl virus (ToLCV) is the most common viral disease...
  ```
- Semantic chunking: split on topic boundaries using embedding similarity
- Metadata enrichment: add `{crop, season, region, data_type}` to each chunk

```python
class ContextualChunker:
    async def chunk_with_context(document: str, doc_metadata: dict) -> list[ContextualChunk]
```

---

#### [NEW] `ai/rag/parent_child_retriever.py`

**Parent-Child Document Retrieval**

- Store small chunks (256 tokens) for precise retrieval
- Return parent chunks (1024 tokens) for richer LLM context
- Parent↔child mapping in metadata
- Critical for agriculture: pest control advice needs crop + region context from parent section

---

#### [MODIFY] `ai/rag/hybrid_search.py`

Changes:
- Time-aware weighting for market/price queries (boost recent docs)
- Category-filtered BM25 (only search within relevant category)
- Agricultural domain tokenizer (handle "₹", "kg/ha", "quintal", Kannada terms)

---

#### [MODIFY] `ai/rag/query_analyzer.py`

Changes:
- Add `MULTI_HOP` route for sequential reasoning queries
- Add `HYDE` route for vague queries
- Add Kannada/Hindi keyword sets for rule-based pre-filter
- Wire up `QueryRewriter` for DECOMPOSE and MULTI_HOP routes

---

### Phase 3: LangGraph State Machine (🟡 P1)

---

#### [NEW] `ai/rag/rag_graph.py`

**LangGraph RAG State Machine** — replaces imperative orchestration loop

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  Query ──→ Analyze ──→ Rewrite ──→ Retrieve ──→ Grade   │
│                                         │           │    │
│                                         │    ┌──────┘    │
│                                         │    │           │
│                   ┌─────────────────────┘    │           │
│                   │                          ▼           │
│                   │              ┌─── Docs Relevant? ──┐ │
│                   │              │                      │ │
│                   │         Yes ─┘             No ──┐   │ │
│                   │              │                  │   │ │
│                   │              ▼                  ▼   │ │
│                   │          Generate         Rewrite   │ │
│                   │          + Cite           + Web     │ │
│                   │              │             Search   │ │
│                   │              ▼                      │ │
│                   │          Evaluate                   │ │
│                   │              │                      │ │
│                   │    ┌─── Confident? ───┐             │ │
│                   │    │                  │             │ │
│                   │  Yes              No (retry≤2)      │ │
│                   │    │                  │             │ │
│                   │    ▼                  ▼             │ │
│                   │  Return         Retry w/ Plan       │ │
│                   │  Answer         Feedback            │ │
│                   │                                     │ │
│                   │    No (max retries) ──→ "I Don't    │ │
│                   │                         Know"       │ │
│                   └─────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

**State schema:**
```python
class RAGState(TypedDict):
    query: str
    rewritten_queries: list[str]
    retrieved_docs: list[Document]
    grading_result: GradingResult
    answer: str
    cited_answer: CitedAnswer
    eval_gate: EvalGate
    retry_count: int
    route: RetrievalRoute
    total_cost_inr: float
```

**Key design:**
- Conditional edges based on grading and evaluation
- Maximum 2 retries (matches current system)
- Each node independently testable
- Full LangSmith tracing via `@trace_rag` decorator
- `langgraph>=0.2.0` already in `pyproject.toml`

---

#### [MODIFY] `ai/rag/agentic/orchestrator.py`

- Keep `AgenticOrchestrator` as public API (backward compat)
- Replace `orchestrate()` body with `LangGraphRAGPipeline.invoke()`
- Preserve all tool implementations (`_tool_vector_search`, etc.)

---

### Phase 4: RAGAS Evaluation & CI Guardrails (🔴 P0)

---

#### [NEW] `ai/rag/eval/golden_dataset.json`

**30 CropFresh-specific golden queries:**

| Category | Count | Examples |
|----------|-------|---------|
| Agronomy | 10 | "How to control aphids on tomato in Karnataka?" |
| Market | 8 | "What is the current onion price in Hubli mandi?" |
| Platform | 5 | "How do I register as a farmer on CropFresh?" |
| Multi-hop | 4 | "Should I sell my rice now or wait for monsoon?" |
| Kannada | 3 | "ಟೊಮ್ಯಾಟೊ ಬೆಲೆ ಎಷ್ಟು?" |

Each entry: `{question, ground_truth, expected_contexts, category, difficulty, language}`

---

#### [MODIFY] `ai/rag/evaluation.py`

Replace mock RAGAS with real evaluation:
- NLI-based faithfulness scoring (not hardcoded 0.9)
- Context precision/recall using ground truth contexts
- Answer usefulness metric (actionable farming advice?)
- Markdown report with per-query breakdown

---

#### [NEW] `ai/rag/eval/eval_ci_gate.py`

**CI Quality Gate** — runs in GitHub Actions before deployment:

```bash
uv run python -m ai.rag.eval.eval_ci_gate --dataset ai/rag/eval/golden_dataset.json
```

| Metric | Threshold | Action on Fail |
|--------|-----------|---------------|
| Faithfulness | ≥ 0.80 | Block deployment |
| Context Relevance | ≥ 0.75 | Block deployment |
| Answer Relevance | ≥ 0.75 | Block deployment |
| "I Don't Know" rate | ≤ 0.15 | Warning only |

---

## 4. Implementation Order

| # | Phase | Est. New LOC | Test LOC | Risk | Depends On |
|---|-------|-------------|----------|------|------------|
| 1 | Anti-Hallucination Pipeline | ~800 | ~400 | Low — additive | None |
| 2 | RAGAS Evaluation & CI Gate | ~400 | ~200 | Low — replaces mock | Phase 1 |
| 3 | Advanced Retrieval | ~600 | ~300 | Low — additive | None |
| 4 | LangGraph State Machine | ~500 | ~300 | Medium — refactors orchestrator | Phase 1, 3 |

**Total: ~2,300 LOC new code + ~1,200 LOC tests**

---

## 5. New Files Summary

```
ai/rag/
├── query_rewriter.py          [NEW]  Phase 1 — HyDE + step-back + multi-query
├── citation_engine.py         [NEW]  Phase 1 — Inline [1], [2] citations
├── confidence_gate.py         [NEW]  Phase 1 — "I don't know" safety gate
├── contextual_chunker_v2.py   [NEW]  Phase 3 — Anthropic-style contextual chunks
├── parent_child_retriever.py  [NEW]  Phase 3 — Small retrieval, big context
├── rag_graph.py               [NEW]  Phase 4 — LangGraph state machine
├── eval/
│   ├── golden_dataset.json    [NEW]  Phase 2 — 30 golden queries
│   └── eval_ci_gate.py        [NEW]  Phase 2 — CI quality gate script
├── grader.py                  [MOD]  Phase 1 — Continuous scoring, time-decay
├── hybrid_search.py           [MOD]  Phase 3 — Time-aware, domain tokenizer
├── query_analyzer.py          [MOD]  Phase 3 — MULTI_HOP, HyDE routes, Kannada
├── evaluation.py              [MOD]  Phase 2 — Real RAGAS, not mock
└── agentic/orchestrator.py    [MOD]  Phase 4 — LangGraph-backed
```

---

## 6. Verification Plan

### Unit Tests

| Test File | Coverage |
|-----------|----------|
| `tests/unit/test_query_rewriter.py` | HyDE, step-back, multi-query, Kannada, LLM failure fallback |
| `tests/unit/test_citation_engine.py` | Multi-source citations, orphan detection, verification |
| `tests/unit/test_confidence_gate.py` | Safety classification, grounding threshold, decline response |
| `tests/unit/test_contextual_chunker.py` | Context injection, semantic splitting, metadata |
| `tests/unit/test_rag_graph.py` | Full pipeline, retry loop, decline path |
| `tests/unit/test_parent_child_retriever.py` | Mapping integrity, parent expansion |

**Run command:**
```bash
uv run pytest tests/unit/test_query_rewriter.py tests/unit/test_citation_engine.py tests/unit/test_confidence_gate.py tests/unit/test_rag_graph.py -v
```

### Import Verification
```bash
uv run python -c "from ai.rag.query_rewriter import QueryRewriter; from ai.rag.citation_engine import CitationEngine; from ai.rag.confidence_gate import ConfidenceGate; from ai.rag.rag_graph import LangGraphRAGPipeline; print('All imports OK')"
```

### Evaluation Baseline
```bash
uv run python -m ai.rag.eval.eval_ci_gate --dataset ai/rag/eval/golden_dataset.json --mode stub
```

---

## 7. Related Documents

| Document | Location |
|----------|----------|
| PLAN.md (Master plan) | `PLAN.md` |
| Project Status | `tracking/PROJECT_STATUS.md` |
| ADR-007: Agentic RAG Orchestrator | `docs/decisions/ADR-007.md` |
| ADR-008: 8-Strategy Adaptive Router | `docs/decisions/ADR-008.md` |
| ADR-009: Two-Layer Agri Embeddings | `docs/decisions/ADR-009.md` |
| Test Strategy | `TESTING/STRATEGY.md` |
| Deployment Guide | `DEPLOYMENT.md` |
| AWS App Runner Config | `infra/aws/app-runner.yaml` |
