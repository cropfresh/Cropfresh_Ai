# Advanced Agentic RAG System — Architecture

> **Status**: Proposed | **Sprint**: 05-06 | **Owner**: CropFresh AI Team  
> **ADR**: [ADR-007](../decisions/ADR-007-agentic-rag-orchestrator.md)

---

## Overview

The Advanced Agentic RAG Orchestrator replaces the current fixed 4-node LangGraph pipeline with an **autonomous retrieval-planning system** that:

- Selects optimal retrieval tools per query (vector, graph, live API, browser, or none)
- Generates parallel speculative drafts for speed
- Self-evaluates generated answers and retries on low confidence

This delivers **–51% voice latency** and **+13% answer accuracy** vs. the fixed pipeline (based on Google Speculative RAG research, 2024).

---

## System Architecture

```
                    ┌─────────────────────────────────┐
                    │         Agentic Orchestrator       │
                    │       ai/rag/agentic_orchestrator.py│
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │       Retrieval Planner          │
                    │  Groq Llama-3.1-8B (fast/cheap) │
                    │  Plans tool calls + order        │
                    └──────────┬───────────────────────┘
                               │ Parallel execution
         ┌─────────────────────┼─────────────────────────────┐
         ▼                     ▼              ▼               ▼
┌──────────────┐  ┌─────────────────┐  ┌──────────┐  ┌────────────┐
│ Vector RAG   │  │  Graph RAG      │  │ Live API │  │  Browser   │
│ (Qdrant)     │  │  (Neo4j)        │  │(eNAM/IMD)│  │  Scraper   │
│ RAPTOR+Hybrid│  │  Entity+Multihop│  │ Real-time│  │ Scrapling  │
└──────┬───────┘  └────────┬────────┘  └────┬─────┘  └─────┬──────┘
       │                   │                │               │
       └───────────────────┴────────────────┴───────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │      Speculative Draft Engine    │
                    │  3 Parallel Drafts (Groq 8B)     │
                    │  Each from different doc subset  │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │         Verifier LLM             │
                    │    Gemini Flash 2.0              │
                    │  Selects best draft + validates  │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │       Self-Evaluator             │
                    │  RAGAS: faithfulness + relevance │
                    │  confidence < 0.75 → retry plan  │
                    └──────────────┬──────────────────┘
                                   │
                              Final Answer
```

---

## Component Breakdown

### 1. Retrieval Planner

**Model**: Groq `llama-3.1-8b-instant` (avg ~80ms, ~₹0.001/call)

**Inputs**: query text, has_image, session context, available tools manifest

**Outputs**: ordered list of tool calls with parameters and can_parallelize flags

**Planning prompt structure**:
```
System: You are a retrieval orchestrator for CropFresh agricultural AI.
        Available tools: [vector_search, graph_rag, price_api, weather_api, 
                         browser_scrape, direct_llm]
        Plan the minimum tools needed to answer accurately and cheaply.

User: Query: {query}
      Context: {session_summary}

Output JSON:
{
  "plan": [
    {"tool": "price_api", "params": {"commodity": "tomato", "location": "Hubli"}},
    {"tool": "vector_search", "params": {"query": "tomato sell vs hold strategy"}}
  ],
  "can_parallelize": true,
  "confidence_threshold": 0.80
}
```

---

### 2. Speculative Draft Engine

**Concept**: Split retrieved documents into 3 subsets → run 3 parallel draft LLMs → verifier picks best.

**Why**: Parallelism trades compute for latency (3 simultaneous 8B calls ≈ 1 simultaneous 70B call, with better accuracy diversity).

```python
async def generate_speculative_drafts(
    documents: list[Document],
    query: str,
    n_subsets: int = 3,
) -> list[Draft]:
    # Split docs into n_subsets
    subsets = split_into_subsets(documents, n_subsets)
    
    # Generate n_subsets drafts in parallel
    drafts = await asyncio.gather(*[
        drafter_llm.generate(query, subset)
        for subset in subsets
    ])
    return drafts  # Verifier picks best
```

**Drafter LLM**: Groq `llama-3.1-8b-instant` (ultra-fast, 940 tokens/s on Groq hardware)  
**Verifier LLM**: Groq `llama-3.3-70b-versatile` or `Gemini Flash 2.0`

---

### 3. Self-Evaluation Loop

Uses a lightweight RAGAS-style scorer to decide if the answer is good enough:

```python
class AgenticSelfEvaluator:
    """Lightweight answer quality gating."""
    
    async def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_docs: list[Document],
    ) -> EvalGate:
        faithfulness = await self._check_faithfulness(answer, retrieved_docs)
        relevance = await self._check_relevance(query, answer)
        confidence = (faithfulness * 0.6) + (relevance * 0.4)
        
        return EvalGate(
            confidence=confidence,
            should_retry=confidence < 0.75,
            reason=f"faithfulness={faithfulness:.2f}, relevance={relevance:.2f}"
        )
```

**Max retries**: 2 (to prevent infinite loops)  
**Retry strategy**: Revise plan with feedback from evaluator

---

## Performance Targets

| Metric | Before | After | Method |
|--------|--------|-------|--------|
| Voice latency P95 | ~4.5s | < 2.0s | Speculative parallel drafts |
| Simple query latency | ~2.5s | < 0.5s | Adaptive router → DIRECT_LLM |
| Answer faithfulness | Unknown | > 0.85 | Self-evaluation gate |
| API cost per query avg | ₹0.44 | ₹0.22 | Adaptive routing + speculative |
| Complex query accuracy | ~72% | > 88% | Verifier selection |

---

## Integration Points

| Component | How Connected |
|-----------|--------------|
| `ai/rag/query_analyzer.py` | `AdaptiveQueryRouter` determines if agentic path needed |
| `ai/rag/raptor.py` | One of the vector retrieval tools in the plan |
| `ai/rag/graph_retriever.py` | Graph traversal tool in the plan |
| `src/scrapers/enam_client.py` | Live price API tool |
| `src/scrapers/imd_weather.py` | Weather API tool |
| `ai/rag/browser_rag.py` | Browser scraping tool |
| `ai/rag/observability.py` | Every orchestrator call is LangSmith-traced |

---

## File Locations

| File | Purpose |
|------|---------|
| `ai/rag/agentic_orchestrator.py` | Main orchestrator (NEW — Sprint 05) |
| `ai/rag/speculative_engine.py` | Speculative draft engine (NEW — Sprint 06) |
| `ai/rag/self_evaluator.py` | Self-evaluation loop (NEW — Sprint 06) |
| `scripts/test_agentic_rag.py` | Integration test script |

---

## Related Documents

- [ADR-007: Agentic RAG Decision](../decisions/ADR-007-agentic-rag-orchestrator.md)
- [Adaptive Query Router](./adaptive_query_router.md)
- [Browser Scraping RAG](./browser_scraping_rag.md)
- [RAG Architecture Overview](./rag_architecture.md)
