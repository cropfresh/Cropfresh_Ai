# Sprint Retrospective — RAG Subsystem

## 🟢 What Went Well

- **Adaptive Query Router (ADR-008)**: Successfully implemented an 8-strategy routing system combining a rapid rule-based pre-filter with a Groq 8B LLM classifier, reducing costs by ~52%.
- **Agentic Orchestrator (ADR-007)**: Deployed an autonomous multi-tool retrieval planner that delegates to specialized tools (Vector Search, Graph RAG, Price API, Web Scrape).
- **Speculative Draft Engine**: Integrated parallel generation of 3 drafts evaluated by a verifier LLM, cutting latency by over 50%.
- **Browser RAG Integration (ADR-010)**: Live web retrieval layer is fully functional, using Playwright and Scrapling to fetch real-time agricultural updates, prices, and weather.
- **Enhanced Retrieval Pipeline**: Advanced strategies including Parent Document, Sentence Window, Auto-Merging, and MMR are successfully implemented to maximize context relevance and diversity.
- **RAPTOR Indexing**: Hierarchical document indexing with UMAP + GMM clustering and recursive LLM summarization is successfully built for multi-level abstraction retrieval.
- **Self-Evaluator (CRAG)**: RAGAS-style confidence gating (faithfulness + relevance) acts as a robust safety net, retrying responses with confidence < 0.75.

## 🟡 What Could Improve

- **Graph RAG Neo4j Integration**: The `GraphConstructor` currently has a stubbed `pass` implementation for `ingest_to_neo4j` and `multi_hop_reasoning`, relying on simulated logic instead of actual Cypher queries.
- **BM25 Sparse Index Initialization**: In `hybrid_search.py`, the initial build of the BM25 index from the knowledge base is bypassed; it currently requires documents to be indexed one-by-one as they are ingested.
- **LLM Triple Extraction Parsing**: The LLM parsing in `GraphConstructor._parse_triples` relies on brittle parenthesis string-splitting `(Subject, Relation, Object)`, which is prone to failure on complex extracted texts.
- **Rule-Based Routing Fallbacks**: The keyword matching in `QueryAnalyzer` and `AdaptiveQueryRouter` rule-based fallbacks is static and can be fragile as the vocabulary of intents expands.

## 🔴 Action Items

- [ ] Complete Neo4j Cypher query implementations for graph ingestion (using `UNWIND` batching) and multi-hop traversal in `graph_constructor.py`.
- [ ] Refactor the triple extraction prompt and parsing logic in `GraphConstructor` to use structured JSON for reliable LLM output parsing.
- [ ] Implement a robust batch-hydration pipeline for the BM25 index in `HybridRetriever.initialize()` to process existing knowledge base documents on startup.
- [ ] Add comprehensive automated tests for the Adaptive Query Router's rule-based pre-filters to prevent regressions as new intents are added.
