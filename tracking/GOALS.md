# 🎯 Project Goals & OKRs — CropFresh AI

## Vision
Build India's most intelligent agricultural marketplace for Karnataka farmers.

---

## Q1 2026 Objectives (Jan → Mar 14)

### O1: Build Reliable Data Infrastructure
- KR1: APMC/eNAM scraper fetching 100+ mandi prices daily
- KR2: Qdrant knowledge base with 10K+ agricultural documents
- KR3: Neo4j graph with 50+ crop-market relationships
- KR4: < 500ms average vector retrieval latency

### O2: Deploy Production AI Agent
- KR1: Crop + Market Agents with > 90% routing accuracy
- KR2: < 3 second voice round-trip response time
- KR3: Kannada voice input support functional
- KR4: RAGAS evaluation baseline established

### O3: Establish Development Velocity
- KR1: Sprint cadence maintained (2-week sprints, retrospective each)
- KR2: ≥ 40% test coverage on core RAG + agent modules
- KR3: Daily development logs maintained each session
- KR4: All ADRs written before implementation begins

---

## Q2 2026 Objectives (Mar 14 → Jun 30) — RAG 2027 Upgrade

> **Research basis:** [RAG 2027 Research Report](../rag_2027_research_report.md) (Feb 27, 2026)

### O1: Adaptive Intelligence (Sprint 05)
- KR1: Adaptive Query Router live — average query cost drops from ₹0.44 → < ₹0.25/query (–43%)
- KR2: AgriEmbeddingWrapper deployed — context precision improves ≥ +8% vs. BGE-M3 baseline
- KR3: Agentic Orchestrator v1 live — complex queries routed through retrieval planner successfully
- KR4: RAGAS golden dataset (20 queries) baseline score: faithfulness > 0.75

### O2: Live Web Intelligence (Sprint 06)
- KR1: Browser-Augmented RAG live — ≥ 5 gov/news sources scraped with > 80% success rate
- KR2: Speculative Draft Engine deployed — voice P95 latency < 2.0s (from ~4.5s)
- KR3: RAGAS CI automated — faithfulness score gates every `main` push
- KR4: All browser-scraped answers include source citation + freshness label

### O3: Retrieval Precision (Sprint 07)
- KR1: ColBERT late-interaction retriever integrated — retrieval P@10 improves ≥ +20% on golden set
- KR2: Community-level GraphRAG queries working (theme-level questions)
- KR3: RAGAS faithfulness ≥ 0.85 and context precision ≥ 0.80

---

## 2027 North Star Targets

| Metric | Target |
|--------|--------|
| Voice P95 latency | < 1.5s |
| Avg API cost per query | < ₹0.18 |
| RAGAS faithfulness | > 0.92 |
| KB documents indexed | > 10,000 |
| Embedding domain precision | +25% vs generic BGE-M3 |
| Mandis with live price data | > 500 |

---

## Key Milestones
See `tracking/milestones/` for detailed milestone tracking.
