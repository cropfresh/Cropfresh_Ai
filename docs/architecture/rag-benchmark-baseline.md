# Advanced Agentic RAG Benchmark Baseline

> Last updated: 2026-03-16
> Latest implementation change: live benchmark runner now evaluates the real `KnowledgeAgent -> src/rag/graph.py -> ai.rag.graph.run_rag_graph()` path.

## Benchmark Modes

| Mode | Purpose | Data source | Score type |
|------|---------|-------------|------------|
| Heuristic lower bound | Smoke checks and offline fallback | JSON dataset + pipeline output | Keyword/overlap proxy |
| Live semantic benchmark | Headline benchmark and release gate | JSON dataset + resolved live references + pipeline output | RAGAS-style semantic score |

## Accepted Gate for Phase 1

The first milestone targets the `core_live` subset only:

- Categories: `market`, `agronomy`, `pest`, `scheme`, `kannada`
- Deferred from the first `0.90` gate: weather forecasting and multi-hop sell/hold reasoning
- Release gate: median of 3 live runs

| Metric | Target |
|--------|--------|
| Overall semantic score | `>= 0.90` |
| Faithfulness | `>= 0.93` |
| Answer relevancy | `>= 0.90` |
| Context precision | `>= 0.85` |
| Context recall | `>= 0.90` |
| Citation coverage | `>= 0.95` |
| Freshness compliance | `1.00` |
| Hallucination rate | `<= 0.07` |

## Latest Accepted Baseline

The last accepted baseline is still the heuristic report from **2026-03-14**. That report is useful only as a lower bound because it did not score semantic understanding and it did not use live pipeline outputs end-to-end.

| Metric | 2026-03-14 heuristic lower bound |
|--------|----------------------------------|
| Faithfulness | `~0.49` |
| Answer relevancy | `~0.37` |
| Context precision | `~0.69` |
| Hallucination rate | `~0.51` |
| Composite score | `~0.40` |

## Current Status

As of **2026-03-16**, the benchmark stack has been upgraded to:

- load JSON-backed `core_live` and `full` datasets
- resolve live market references before scoring
- query the canonical runtime path instead of comparing canned answers
- enforce citation coverage and live-source freshness
- write raw run artifacts to `reports/rag/`

The next accepted baseline should be recorded only after this command succeeds for 3 runs:

```bash
uv run python scripts/eval_guardrail.py --live --subset core_live --runs 3
```

## Dataset Definitions

`core_live`
- phase-1 gate set for market, agronomy, pest, scheme, and Kannada queries

`full`
- broader regression set including deferred weather and multi-hop items

## Known Failure Buckets to Watch

- Kannada scheme queries falling into agronomy-like keyword paths
- live market answers missing `as_of` timestamps or fresh source metadata
- weakly grounded pesticide/safety answers that should abstain
- stale live-source data passing retrieval but failing freshness checks

## Reports

Raw benchmark outputs are written under `reports/rag/`.

- Markdown report: semantic scores plus guardrail summary
- JSON report: machine-readable semantic metrics
- `_guardrail.json`: pass/fail snapshot with citation and freshness checks
- `_extras.json`: live-run deterministic extras

## Commands

```bash
uv run python scripts/eval_guardrail.py --live --subset core_live --runs 3
uv run python scripts/eval_guardrail.py --heuristic --subset full
```
