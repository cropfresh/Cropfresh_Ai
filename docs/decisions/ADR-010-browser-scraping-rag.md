# ADR-010: Browser-Augmented RAG (Live Web Retrieval)

**Date**: 2026-02-27  
**Status**: Proposed  
**Deciders**: CropFresh AI Team  
**Builds on**: `src/scrapers/base_scraper.py` (Scrapling + Camoufox)

---

## Context

CropFresh's static knowledge base (Qdrant) has a **knowledge staleness problem**:

1. **Government schemes** (PM-KISAN, PMFBY, eNAM onboarding rules) change quarterly. KB goes stale within weeks.
2. **Novel disease outbreaks** (e.g., new tomato yellow fever virus strain in 2026 rabi season) won't be in the KB until manually ingested.
3. **Market disruptions** (export bans, bumper harvest in competing states) require same-day intelligence for "sell or hold" decisions.
4. **Regulatory changes** (APMC Act amendments, pesticide ban notifications) are time-critical.

The current web search fallback uses **Tavily API in mock mode** — real-world queries hit the "CORRECT" path only for pre-known URLs. This is inadequate for production.

---

## Decision

Build a **BrowserRAGIntegration** layer that:

1. **Routes `BROWSER_SCRAPE` strategy** requests (from Adaptive Router) to the browser pipeline
2. **Uses existing `ScraplingBaseScraper`** infrastructure (circuit breaker, cache, rate limiting)
3. **Extracts structured content** and converts to `Document` objects for on-the-fly Qdrant upsert
4. **Optionally persists** scraped data to a `scraped_live` Qdrant collection for short-term caching (TTL: 4h)

### Source Priority Matrix

| Query Type | Primary Source | Fallback | TTL |
|-----------|---------------|---------|-----|
| Govt scheme updates | data.gov.in, pmkisan.gov.in | Tavily search | 6h |
| Pesticide bans | cibrc.nic.in | Google News | 12h |
| Export policy | apeda.gov.in | Economic Times | 4h |
| Disease advisories | icar.org.in, KVK sites | AgriWatch | 6h |
| Weather alerts | imd.gov.in | OpenWeatherMap | 1h |
| News / analysis | Krishi Jagran, DD Kisan | Bing News | 2h |

### Architecture

```
AdaptiveRouter → BROWSER_SCRAPE
        │
        ▼
BrowserRAGIntegration
    ├── SourceSelector (picks best source for query)
    ├── ScraplingBaseScraper (anti-bot, circuit breaker)
    │       ├── StealthyFetcher (gov.in, complex sites)
    │       └── BasicFetcher (static news pages)
    ├── ContentExtractor (Scrapling CSS/XPath → text)
    ├── LiveDocumentBuilder (text → Document → embed)
    ├── QdrantLiveCollection (upsert with TTL metadata)
    └── RAGRetriever (search live + KB collections)
```

### Implementation sketch

```python
class BrowserRAGIntegration:
    """Live browser retrieval integrated into RAG pipeline."""
    
    async def retrieve_live(
        self,
        query: str,
        sources: list[str] | None = None,
    ) -> list[Document]:
        """Scrape live web data and return as RAG documents."""
        
        # 1. Select best sources for this query
        target_urls = self.source_selector.select(query, sources)
        
        # 2. Scrape with existing Scrapling infrastructure
        scrape_tasks = [self.scraper.fetch(url) for url in target_urls]
        pages = await asyncio.gather(*scrape_tasks, return_exceptions=True)
        
        # 3. Extract structured content
        docs = []
        for url, page in zip(target_urls, pages):
            if isinstance(page, Exception):
                logger.warning(f"Scrape failed: {url} — {page}")
                continue
            text = self._extract_content(page)
            doc = Document(
                text=text,
                metadata={"source": url, "scraped_at": datetime.utcnow().isoformat(), "ttl_hours": 4}
            )
            docs.append(doc)
        
        # 4. Optional: upsert to live Qdrant collection
        if docs:
            await self.qdrant_live.upsert(docs)
        
        return docs
```

---

## Consequences

**Positive:**
- ✅ Answers queries about events that happened **today** (no KB stale lag)
- ✅ Reuses battle-tested `ScraplingBaseScraper` with circuit breaker + cache
- ✅ Live data persisted to Qdrant → next farmer asking same query gets cached answer
- ✅ Farmer trust increases when answers cite "from icar.org.in, retrieved 2h ago"

**Negative:**
- ⚠️ Gov.in sites are often slow/unreliable — circuit breaker will occasionally open
- ⚠️ Anti-bot detection risk — mitigated by `StealthyFetcher` + Camoufox
- ⚠️ Higher latency for `BROWSER_SCRAPE` queries (~2–5s) — acceptable vs. no answer
- ⚠️ Content quality varies — needs post-scrape quality filter

**Mitigation:**
- Max 3-second scrape timeout; fall back to Tavily if circuit open
- Quality filter: minimum 150 words, no error-page pattern match
- Only invoke `BROWSER_SCRAPE` for < 5% of queries (per adaptive router)

---

## Alternatives Considered

| Approach | Reason Rejected |
|----------|----------------|
| Tavily API only | Paid API, ~$0.01/search, no gov.in coverage |
| Bing Web Search API | Generic, no agri-specific filtering |  
| Pre-scheduled crawl only | 6-hour delay — misses intraday market events |
| Playwright raw browser | ScraplingBaseScraper already wraps Playwright better |

---

## Related

- Architecture: [`browser_scraping_rag.md`](../architecture/browser_scraping_rag.md)
- Base scraper: `src/scrapers/base_scraper.py`
- Implementation: `ai/rag/browser_rag.py` (Sprint 06)
- Test script: `scripts/test_browser_rag.py`
