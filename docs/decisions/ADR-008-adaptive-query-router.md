# ADR-008: Adaptive Query Router (8-Strategy)

**Date**: 2026-02-27  
**Status**: Proposed  
**Deciders**: CropFresh AI Team  
**Extends**: `ai/rag/query_analyzer.py` (current 4-type basic router)

---

## Context

The current `QueryAnalyzer` has 4 routing types: `vector | web | decompose | direct`.  
This is insufficient for production because:

- `web` is too coarse — doesn't distinguish live price API vs. browser scraping vs. real-time weather
- `vector` doesn't distinguish between RAPTOR tree search, flat dense search, or Graph RAG
- No `vision` routing when farmers attach crop photos
- No cost signal — expensive Gemini Pro is called for a "Hello" greeting

**Cost impact**: ~40% of CropFresh queries are simple (greetings, basic definitions) and don't need retrieval at all, but the current router sends them through the full pipeline at ~₹0.45/query when they could be answered for ~₹0.03.

---

## Decision

Upgrade `QueryAnalyzer` to an **8-Strategy Adaptive Router** with an explicit cost signal per strategy:

| Strategy | Trigger | LLM Calls | Avg Cost |
|----------|---------|-----------|----------|
| `DIRECT_LLM` | Greeting, simple fact | 1 (cheap) | ₹0.03 |
| `VECTOR_ONLY` | Domain knowledge Q | 1 retrieval + 1 LLM | ₹0.12 |
| `GRAPH_TRAVERSAL` | Relational/entity Q | Graph + LLM | ₹0.15 |
| `LIVE_PRICE_API` | Current price query | 1 API + 1 LLM | ₹0.05 |
| `WEATHER_API` | Weather/forecast | 1 API + 1 LLM | ₹0.05 |
| `BROWSER_SCRAPE` | Novel web data | Browser + LLM | ₹0.25 |
| `MULTIMODAL` | Image attached | Vision + RAG + LLM | ₹0.35 |
| `FULL_AGENTIC` | Complex multi-step | Full orchestrator | ₹0.55 |

Implementation: extend `ai/rag/query_analyzer.py` with `AdaptiveQueryRouter`.

```python
class AdaptiveQueryRouter:
    """8-strategy router with explicit cost signals per route."""
    
    ROUTING_PROMPT = """You are classifying an agricultural query for CropFresh AI.

Query: {query}
Has image attached: {has_image}
Session turncount: {turn_count}
User language: {language}

Choose the most efficient strategy from:
- DIRECT_LLM: greeting, thanks, who-are-you, simple definition
- VECTOR_ONLY: farming practice, crop cultivation, pest/disease knowledge
- GRAPH_TRAVERSAL: which farmers grow X? relationships between entities
- LIVE_PRICE_API: current mandi prices, today's rates (needs eNAM)
- WEATHER_API: today's weather, rainfall forecast, agro-advisory
- BROWSER_SCRAPE: latest news, scheme updates, data not in KB
- MULTIMODAL: image is attached and needs visual analysis
- FULL_AGENTIC: complex multi-step, compare + recommend + plan

Return JSON: {"strategy": "...", "confidence": 0.0-1.0, "reason": "..."}"""
```

---

## Consequences

**Positive:**
- ✅ **~50% average cost reduction** per query (most traffic is simple)
- ✅ **< 1s responses** for DIRECT_LLM queries (no retrieval latency)
- ✅ **Explicit cost telemetry** per strategy (tracked in LangSmith)
- ✅ **Extensible** — add new strategies without changing the agent system

**Negative:**
- ⚠️ Router itself calls LLM → adds ~150ms overhead for all queries
- ⚠️ Wrong classification costs more (FULL_AGENTIC for a simple query)

**Mitigation:**  
Use `Groq Llama-3.1-8b-instant` for the router — ultra-low latency (~80ms), tiny cost (< ₹0.001/call). Rule-based pre-filtering for obvious `DIRECT_LLM` queries before the LLM call.

---

## Migration Path

1. Keep existing `QueryAnalyzer` class untouched (backward compatibility)
2. Add `AdaptiveQueryRouter` as a new class in same file
3. Feature-flag: `USE_ADAPTIVE_ROUTER=true` in `.env`
4. A/B test: 10% traffic to adaptive router → measure cost + quality

---

## Related

- [ADR-007: Agentic RAG Orchestrator](./ADR-007-agentic-rag-orchestrator.md)  
- [ADR-009: Agricultural Embeddings Fine-tuning](./ADR-009-agri-embeddings.md)
- Architecture: [`adaptive_query_router.md`](../architecture/adaptive_query_router.md)
- Implementation: upgrade `ai/rag/query_analyzer.py` (Sprint 05)
