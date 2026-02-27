# Adaptive Query Router — Architecture

> **Status**: Proposed | **Sprint**: 05 | **Owner**: CropFresh AI Team  
> **ADR**: [ADR-008](../decisions/ADR-008-adaptive-query-router.md)

---

## Overview

The Adaptive Query Router is a **pre-retrieval intelligence layer** that classifies every incoming query and assigns it the most efficient retrieval strategy — from zero retrieval (direct LLM) up to full agentic orchestration.

This is the **primary cost control mechanism** in the CropFresh AI stack. Without it, every query — including "Hello" and "Thanks" — incurs the same RAG pipeline cost.

---

## The 8-Strategy Decision Tree

```
Incoming Query
     │
     │  ① Rule-based pre-filter (0ms, free)
     ▼
─────────────────────────────────
Greeting? Simple hi/thanks/bye?
     │YES                │NO
     ▼                   ▼
DIRECT_LLM        LLM Classification
(₹0.03)           (Groq 8B, ~80ms, ₹0.001)
                        │
            ┌───────────┼───────────────────────┐
            ▼           ▼           ▼            ▼
      Image?      Price/rate?   Weather?   Relational?
         │             │             │          │
         ▼             ▼             ▼          ▼
    MULTIMODAL   LIVE_PRICE_API  WEATHER_API  GRAPH_
    (₹0.35)       (₹0.05)        (₹0.05)     TRAVERSAL
                                              (₹0.15)
            ▼           ▼
     Browser scrape?  Complex multi-step?
          │YES              │YES
          ▼                 ▼
    BROWSER_SCRAPE    FULL_AGENTIC
      (₹0.25)           (₹0.55)
          │NO              │NO
          ▼                ▼
      VECTOR_ONLY      VECTOR_ONLY
       (₹0.12)          (₹0.12)
```

---

## Strategy Reference Table

| ID | Strategy | When Used | Tools Called | Avg Cost | Avg Latency |
|----|----------|-----------|-------------|---------|------------|
| 1 | `DIRECT_LLM` | Greetings, who-am-I, yes/no | LLM only | ₹0.03 | 0.4s |
| 2 | `VECTOR_ONLY` | Farming practice, agronomy Q | RAPTOR + Dense + Rerank | ₹0.12 | 1.0s |
| 3 | `GRAPH_TRAVERSAL` | Which farmers? Entity relations | Neo4j + LLM | ₹0.15 | 1.2s |
| 4 | `LIVE_PRICE_API` | "Tomato price today in Hubli?" | eNAM → LLM | ₹0.05 | 0.8s |
| 5 | `WEATHER_API` | "Rain forecast next 3 days?" | IMD API → LLM | ₹0.05 | 0.7s |
| 6 | `BROWSER_SCRAPE` | Scheme news, latest advisories | Scrapling → Qdrant → LLM | ₹0.25 | 3-5s |
| 7 | `MULTIMODAL` | Image of crop + question | YOLOv12 + ColPali + LLM | ₹0.35 | 2.0s |
| 8 | `FULL_AGENTIC` | Complex multi-factor decisions | Orchestrator all tools | ₹0.55 | 2.5s |

---

## Expected Traffic Distribution

Based on typical agricultural chatbot usage patterns (projected):

| Strategy | % of Queries | Contribution to Avg Cost |
|----------|-------------|------------------------|
| DIRECT_LLM | 25% | ₹0.0075 |
| VECTOR_ONLY | 38% | ₹0.0456 |
| LIVE_PRICE_API | 18% | ₹0.0090 |
| WEATHER_API | 6% | ₹0.0030 |
| GRAPH_TRAVERSAL | 5% | ₹0.0075 |
| BROWSER_SCRAPE | 3% | ₹0.0075 |
| FULL_AGENTIC | 4% | ₹0.0220 |
| MULTIMODAL | 1% | ₹0.0035 |
| **Total** | 100% | **₹0.206 avg** |

**Before router**: ~₹0.44 avg/query → **After router**: ~₹0.21 avg/query (**–52% cost**)

---

## Implementation: Class Design

```python
# ai/rag/query_analyzer.py — extended

class RetrievalStrategy(str, Enum):
    """8 routing strategies for the adaptive query router."""
    DIRECT_LLM       = "direct_llm"        # No retrieval
    VECTOR_ONLY      = "vector_only"        # KB dense + sparse
    GRAPH_TRAVERSAL  = "graph_traversal"    # Neo4j query
    LIVE_PRICE_API   = "live_price_api"     # eNAM real-time
    WEATHER_API      = "weather_api"        # IMD forecast
    BROWSER_SCRAPE   = "browser_scrape"     # Live web
    MULTIMODAL       = "multimodal"         # Vision + RAG
    FULL_AGENTIC     = "full_agentic"       # All tools + planning


class RoutingDecision(BaseModel):
    """Output of the adaptive router."""
    strategy: RetrievalStrategy
    confidence: float                  # 0.0 - 1.0
    reason: str                        # Explanation for LangSmith logs
    estimated_cost_inr: float          # Cost signal for monitoring
    entities: dict[str, list[str]]     # Extracted: crops, locations, etc.
    requires_live_data: bool
    requires_image: bool


class AdaptiveQueryRouter:
    """
    8-Strategy Adaptive Query Router.
    
    Uses Groq Llama-3.1-8B for fast, cheap classification
    with rule-based pre-filter for obvious cases.
    """
    
    # Pre-filter: rule-based, zero latency
    GREETING_PATTERNS = {
        "hello", "hi", "hey", "namaste", "vanakkam", "thanks",
        "thank you", "bye", "goodbye", "who are you", "what is cropfresh"
    }
    
    IMAGE_TRIGGERS = {"photo", "image", "picture", "pic", "photo of", "[image"}
    
    PRICE_TRIGGERS = {
        "price", "rate", "cost", "mandi rate", "today rate", "current price",
        "what is the price", "how much for", "selling price", "bhaav"
    }
    
    WEATHER_TRIGGERS = {
        "weather", "rain", "rainfall", "forecast", "temperature", "humidity",
        "monsoon", "barish", "garmi", "sardi"
    }
    
    async def route(self, query: str, has_image: bool = False) -> RoutingDecision:
        """Route a query to the optimal retrieval strategy."""
        
        # Step 1: Rule-based pre-filter (free, instant)
        pre_result = self._prefilter(query, has_image)
        if pre_result:
            return pre_result
        
        # Step 2: LLM classification (Groq 8B, ~80ms, ₹0.001)
        return await self._llm_classify(query, has_image)
    
    def _prefilter(self, query: str, has_image: bool) -> RoutingDecision | None:
        """Fast rule-based routing for obvious cases."""
        q = query.lower().strip()
        
        if has_image:
            return RoutingDecision(
                strategy=RetrievalStrategy.MULTIMODAL,
                confidence=0.95, reason="Image attached",
                estimated_cost_inr=0.35,
                entities={}, requires_live_data=False, requires_image=True
            )
        
        if any(p in q for p in self.GREETING_PATTERNS):
            return RoutingDecision(
                strategy=RetrievalStrategy.DIRECT_LLM,
                confidence=0.98, reason="Greeting detected by rule",
                estimated_cost_inr=0.03,
                entities={}, requires_live_data=False, requires_image=False
            )
        
        if any(p in q for p in self.PRICE_TRIGGERS) and any(
            kw in q for kw in ["today", "now", "current", "aaj", "abhi"]
        ):
            return RoutingDecision(
                strategy=RetrievalStrategy.LIVE_PRICE_API,
                confidence=0.90, reason="Live price query detected",
                estimated_cost_inr=0.05,
                entities={}, requires_live_data=True, requires_image=False
            )
        
        return None  # Falls through to LLM classification
```

---

## Observability

Every routing decision is logged to LangSmith with:
- `strategy_chosen` (string)
- `routing_confidence` (float)
- `estimated_cost_inr` (float)
- `routing_latency_ms` (float)
- `pre_filter_matched` (bool)

This lets us track the routing accuracy over time and retrain the classifier if needed.

---

## A/B Testing Plan

| Phase | Traffic Split | Duration | Goal |
|-------|-------------|----------|------|
| Canary | 5% adaptive / 95% fixed | Sprint 05 | Validate cost reduction |
| Ramp | 25% / 75% | Sprint 06 | Check quality parity |
| Majority | 75% / 25% | Sprint 07 | Confirm P95 latency improvement |
| Full | 100% adaptive | Sprint 08 | Deprecated fixed router |

---

## Related Documents

- [ADR-008: Adaptive Router Decision](../decisions/ADR-008-adaptive-query-router.md)
- [Agentic RAG System](./agentic_rag_system.md)
- [Browser Scraping RAG](./browser_scraping_rag.md)
- Implementation file: `ai/rag/query_analyzer.py`
