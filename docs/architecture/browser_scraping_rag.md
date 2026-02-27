# Browser-Augmented RAG — Architecture

> **Status**: Proposed | **Sprint**: 06 | **Owner**: CropFresh AI Team  
> **ADR**: [ADR-010](../decisions/ADR-010-browser-scraping-rag.md)

---

## Overview

Browser-Augmented RAG extends the static Qdrant knowledge base with **live web retrieval** for time-sensitive agricultural queries. It integrates directly with the existing `ScraplingBaseScraper` infrastructure (circuit breaker, rate limiting, caching) and converts scraped content into RAG documents on the fly.

This layer is invoked **only** when the Adaptive Query Router routes to `BROWSER_SCRAPE` strategy — typically for scheme updates, novel disease alerts, regulatory changes, and news that cannot be in the KB yet.

---

## System Architecture

```
Adaptive Router → BROWSER_SCRAPE
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│                 BrowserRAGIntegration                     │
│                 ai/rag/browser_rag.py                    │
│                                                          │
│  ┌────────────────┐   ┌─────────────────────────────┐   │
│  │ SourceSelector │   │    ScraplingBaseScraper      │   │
│  │                │   │    (base_scraper.py)         │   │
│  │ Maps query →   │   │                             │   │
│  │ best URLs to   │   │  ├── BasicFetcher (fast)    │   │
│  │ scrape         │   │  └── StealthyFetcher (anti-bot)│ │
│  └───────┬────────┘   └──────────────┬──────────────┘   │
│          │                           │                    │
│          └──────── Target URLs ──────┘                    │
│                         │                                 │
│              ┌──────────▼──────────┐                     │
│              │  ContentExtractor   │                      │
│              │  Scrapling CSS/XPath│                      │
│              │  → structured text  │                      │
│              └──────────┬──────────┘                      │
│                         │                                 │
│              ┌──────────▼──────────┐                     │
│              │  QualityFilter      │                      │
│              │  min 150 words      │                      │
│              │  no error pages     │                      │
│              └──────────┬──────────┘                      │
│                         │                                 │
│              ┌──────────▼──────────┐                     │
│              │  LiveDocBuilder     │                      │
│              │  text → Document    │                      │
│              │  with TTL metadata  │                      │
│              └──────────┬──────────┘                      │
│                    ┌────┴──────┐                          │
│                    │           │                           │
│          ┌─────────▼───┐  ┌───▼──────────────────┐      │
│          │  Immediate  │  │  Qdrant Live Collection│      │
│          │  RAG Use    │  │  (TTL: 2-6 hours)      │      │
│          └─────────────┘  └──────────────────────┘       │
└──────────────────────────────────────────────────────────┘
        │
        ▼
  Retrieved Documents → Reranker → LLM → Answer with Citation
```

---

## Source Registry

### Primary Agricultural Sources (gov.in)

| Source | URL Domain | Content | TTL | Fetcher |
|--------|-----------|---------|-----|---------|
| PM-KISAN Scheme | pmkisan.gov.in | Beneficiary lists, payment status | 6h | Stealthy |
| ICAR Advisories | icar.org.in | Crop disease bulletins, advisories | 12h | Basic |
| CIB-RC Pesticides | cibrc.nic.in | Approved/banned pesticide list | 24h | Basic |
| APEDA Export Rules | apeda.gov.in | Export policy, commodity rules | 12h | Basic |
| IMD Agro Advisory | imd.gov.in | District-level agro-met advisories | 2h | Basic |
| eNAM Dashboard | enam.gov.in | Mandi arrival, price dashboard | 30min | Stealthy |
| Agrimarket | agrimarket.nic.in | Historical price series | 4h | Basic |

### Agricultural News Sources

| Source | URL | Content | TTL | Fetcher |
|--------|-----|---------|-----|---------|
| Krishi Jagran | krishijagran.com | Farm news, scheme updates | 2h | Basic |
| AgriWatch | agriwatch.com | Commodity market news | 1h | Basic |
| Big News Network | bignn.com/agriculture | Karnataka agri news | 2h | Basic |
| The Hindu BusinessLine | bl.com | Crop market policy news | 3h | Basic |

---

## SourceSelector Logic

```python
class AgriSourceSelector:
    """Maps query intent to best URLs to scrape."""
    
    INTENT_SOURCE_MAP = {
        "scheme_update": [
            "https://pmkisan.gov.in/Notice.aspx",
            "https://agricoop.nic.in/en/news",
            "https://icar.org.in/content/news",
        ],
        "pesticide_ban": [
            "https://cibrc.nic.in/news.htm",
            "https://agricoop.nic.in/en/pesticide",
        ],
        "disease_alert": [
            "https://icar.org.in/content/news",
            "https://nhb.gov.in/Reports/",
        ],
        "export_policy": [
            "https://apeda.gov.in/apedawebsite/News_&_Events.htm",
        ],
        "weather_agriculture": [
            "https://imd.gov.in/pages/agromet_main.php",
        ],
        "market_news": [
            "https://agriwatch.com/news",
            "https://krishijagran.com/news",
        ],
    }
    
    async def select_sources(
        self,
        query: str,
        max_sources: int = 3,
    ) -> list[TargetSource]:
        """Use LLM to classify intent then map to sources."""
        intent = await self._classify_intent(query)
        urls = self.INTENT_SOURCE_MAP.get(intent, self.INTENT_SOURCE_MAP["market_news"])
        return [TargetSource(url=u, intent=intent) for u in urls[:max_sources]]
```

---

## ContentExtractor Selectors

```python
# Scrapling CSS/XPath selectors per domain
EXTRACTORS = {
    "icar.org.in": {
        "title":   "h1.page-title::text",
        "content": "div.field-body p::text",
        "date":    "span.date-display-single::text",
    },
    "pmkisan.gov.in": {
        "title":   "h2::text",
        "content": "div.notice-board p::text",
        "date":    "td.date::text",
    },
    "cibrc.nic.in": {
        "title":   "h3::text",
        "content": "div#content p::text",
        "date":    "span.news-date::text",
    },
    # Default: generic extraction
    "_default": {
        "title":   "h1::text, h2::text",
        "content": "article p::text, main p::text, div.content p::text",
        "date":    "time::attr(datetime), span.date::text",
    }
}
```

---

## Document TTL and Lifecycle

Scraped documents are stored in a dedicated Qdrant collection `live_web_cache` with TTL metadata. A background APScheduler job runs every 30 minutes to purge expired documents.

```python
# Document metadata for live collection
LiveDocMetadata = {
    "source_url": "https://icar.org.in/...",
    "scraped_at": "2026-02-27T10:30:00Z",
    "ttl_hours": 6,
    "expires_at": "2026-02-27T16:30:00Z",  # scraped_at + ttl
    "scraper_type": "basic",
    "intent": "disease_alert",
    "quality_score": 0.87,
    "word_count": 342,
}

# Purge job (APScheduler)
@scheduler.scheduled_job('interval', minutes=30)
async def purge_expired_live_documents():
    now = datetime.utcnow().isoformat()
    await qdrant.delete(
        collection_name="live_web_cache",
        filter=qdrant_models.Filter(
            must=[qdrant_models.FieldCondition(
                key="expires_at",
                range=qdrant_models.Range(lte=now)
            )]
        )
    )
```

---

## Failure Modes and Fallbacks

| Failure | Detection | Fallback |
|---------|-----------|---------|
| Circuit open (too many failures) | `CircuitState.OPEN` | Return vector KB results with staleness warning |
| Scrape timeout (> 3s) | `asyncio.timeout(3)` | Try next source in priority list |
| Quality filter fail | < 150 words or error page pattern | Skip source, try next |
| All sources fail | Empty result after all retries | Fall back to vector KB + "information may be dated" disclaimer |
| Anti-bot block | HTTP 403 / CAPTCHA detected | Switch to StealthyFetcher; if still blocked, circuit open |

---

## Integration with Citation System

All browser-scraped answers include source citations:

```python
class CitedAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    freshness_label: str  # "Real-time", "Live (2h ago)", "Cached (5h ago)"

class Citation(BaseModel):
    source_url: str
    source_name: str     # "ICAR.org.in", "PM-KISAN Portal"
    scraped_at: datetime
    excerpt: str         # Relevant snippet used
```

Farmer-facing format: *"Based on [ICAR Advisory](https://icar.org.in/...) (retrieved 2h ago)..."*

---

## Related Documents

- [ADR-010: Browser RAG Decision](../decisions/ADR-010-browser-scraping-rag.md)
- [Agentic RAG System](./agentic_rag_system.md)
- [Adaptive Query Router](./adaptive_query_router.md)
- Base scraper: `src/scrapers/base_scraper.py`
- Implementation: `ai/rag/browser_rag.py` (Sprint 06)
- Test: `scripts/test_browser_rag.py`
