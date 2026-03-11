# CropFresh AI — Agent Performance Baseline

> **Last Updated:** 2026-03-11
> **Methodology:** Manual testing + script evaluation
> **Status:** Initial baseline — formal RAGAS evaluation pending

---

## Routing Accuracy

| Agent | Test Queries | Correct Routing | Accuracy |
|-------|-------------|----------------|----------|
| agronomy_agent | 10 | 9 | 90% |
| commerce_agent | 10 | 8 | 80% |
| platform_agent | 5 | 4 | 80% |
| general_agent (fallback) | 5 | 5 | 100% |
| voice_agent | 10 | 8 | 80% |
| knowledge_agent | 5 | 4 | 80% |
| **Overall** | **45** | **38** | **~84%** |

**Known Misroutes:**
- "tomato price" sometimes routes to `knowledge_agent` instead of `commerce_agent`
- "how to register" sometimes routes to `general_agent` instead of `platform_agent`

---

## Response Quality

| Agent | Avg Grounding Score | Avg Relevance | Notes |
|-------|-------------------|---------------|-------|
| commerce_agent | 0.8 | 0.85 | Best with live data |
| agronomy_agent | 0.75 | 0.8 | Good RAG retrieval |
| general_agent | 0.6 | 0.7 | Generic responses |

---

## Latency (P95)

| Agent | Without Cache | With Cache |
|-------|-------------|-----------|
| commerce_agent | ~3.2s | ~1.5s |
| agronomy_agent | ~2.8s | ~1.2s |
| voice (end-to-end) | ~4.0s | ~2.5s |

---

## Next Steps

- [ ] Create 20 golden query evaluation set
- [ ] Run RAGAS evaluation for faithfulness, relevance, precision
- [ ] Establish formal cost-per-query tracking
