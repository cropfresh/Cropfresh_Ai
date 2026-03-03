# Task 34: Deep Research Tool — Multi-Source Parallel Web Analysis

> **Priority:** 🟠 P1 | **Phase:** Sprint 05 Phase 5 | **Effort:** 1 day
> **Status:** ✅ COMPLETE — 2026-03-03
> **Files:** `src/tools/deep_research.py` [NEW], `tests/unit/test_deep_research.py` [NEW], `scripts/test_deep_research.py` [NEW]
> **Sprint:** Sprint 05 — Core Agent Completion

---

## 📌 Problem Statement

The existing `web_search` tool returns only snippet-level results (300 chars per site, 5 sites max).
For complex agricultural research questions — policy comparisons, multi-state price analysis, season
predictions — this is not enough. Farmers and buyers need answers that are **verified across 10–15
authoritative sources**, with discrepancies surfaced and citations provided inline.

---

## 🔬 Architecture: Agentic Map-Reduce Pipeline

```
User Query
    │
    ▼
[1] WebSearchTool.search(max_results=15)  → up to 15 URLs
    │
    ▼
[2] fetch_all_pages()  ──── async ────►  Jina Reader r.jina.ai/<url>
    │  (15 pages in parallel)             Full Markdown, 12 KB each
    ▼
[3] extract_all_facts()  ── async ──►  Groq llama-3.1-8b-instant (fast)
    │  (15 LLM calls in parallel)         Each call: "extract relevant facts or SKIP"
    ▼
[4] synthesise_answer()                Groq llama-3.3-70b-versatile (quality)
    │  (single call with all facts)      Compare sources, cite inline, max 600 words
    ▼
DeepResearchResult(answer, sources, pages_fetched, pages_useful)
```

---

## 🏗️ Implementation Spec

### Module: `src/tools/deep_research.py`

```python
class DeepResearchTool:
    async def research(query: str, max_pages: int = 12) -> DeepResearchResult:
        """Full 5-step pipeline."""

    def format_for_llm(result: DeepResearchResult) -> str:
        """Format for LLM context injection."""

# Auto-registers as "deep_research" tool in global ToolRegistry (category: web)
```

### Models

| Model                | Fields                                                |
| -------------------- | ----------------------------------------------------- |
| `PageContent`        | `url, markdown, success, error`                       |
| `ExtractedFact`      | `url, facts, skipped`                                 |
| `DeepResearchResult` | `query, answer, sources, pages_fetched, pages_useful` |

### Groq LLM Usage

| Step   | Model                     | Role                              |
| ------ | ------------------------- | --------------------------------- |
| Map    | `llama-3.1-8b-instant`    | Extract per-page facts (fast)     |
| Reduce | `llama-3.3-70b-versatile` | Synthesise final answer (quality) |

---

## ✅ Acceptance Criteria

| #   | Criterion                                                                        | Status                     |
| --- | -------------------------------------------------------------------------------- | -------------------------- |
| 1   | `DeepResearchTool.research()` returns `DeepResearchResult` with answer + sources | ✅                         |
| 2   | Fetch step visits 10-15 pages **simultaneously** via Jina Reader                 | ✅ `asyncio.gather`        |
| 3   | Map step filters irrelevant pages with SKIP response                             | ✅ `ExtractedFact.skipped` |
| 4   | Reduce step synthesises with inline citations `[1]`, `[2]`                       | ✅ Groq 70B                |
| 5   | Graceful fallback when no URLs or all pages fail                                 | ✅                         |
| 6   | Auto-registered in `ToolRegistry` as `"deep_research"` (category: `web`)         | ✅                         |
| 7   | Unit tests pass with full mocks (no real API calls)                              | ✅ 28 tests                |

---

## 📚 New Files

- `src/tools/deep_research.py` — core pipeline (420 lines)
- `tests/unit/test_deep_research.py` — 28 unit tests (9 test classes)
- `scripts/test_deep_research.py` — smoke test (mock + `--live` mode)

## 📝 Modified Files

- `src/tools/__init__.py` — added `deep_research` import + `DeepResearchTool`, `DeepResearchResult` to `__all__`
