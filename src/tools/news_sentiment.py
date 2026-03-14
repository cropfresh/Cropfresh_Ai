"""
News Sentiment Scraper
=======================
Fetches recent news headlines for agricultural commodities and classifies
sentiment to give the Pricing Agent a market-signal boost.

Architecture:
  - Scrapes DuckDuckGo news RSS (no API key needed)
  - Falls back to a structured web search via httpx
  - Classifies sentiment using keyword matching (fast, zero-latency)
  - Returns a normalized score: -1.0 (very bearish) → +1.0 (very bullish)

The sentiment score is intentionally lightweight — the LLM in the PricingAgent
does the nuanced interpretation.

Author: CropFresh AI Team
Version: 1.0.0
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import httpx
from loguru import logger

# ── News item model ──────────────────────────────────────────────────────────

@dataclass
class NewsItem:
    title: str
    snippet: str
    published_at: Optional[datetime] = None
    source: str = ""
    url: str = ""


@dataclass
class CommoditySentiment:
    commodity: str
    score: float            # -1.0 to +1.0
    label: str              # "bullish" | "bearish" | "neutral"
    top_headlines: list[str] = field(default_factory=list)
    sample_count: int = 0
    fetched_at: datetime = field(default_factory=datetime.now)


# ── Keyword sets ─────────────────────────────────────────────────────────────

#! Keep this list tuned — wrong keywords cause false signals.
_BULLISH_KEYWORDS = [
    "export", "shortage", "deficit", "demand surge", "price rise",
    "bumper demand", "rally", "crop fail", "flood", "drought impact",
    "supply crunch", "hike", "ban on export lifted",
]

_BEARISH_KEYWORDS = [
    "bumper crop", "surplus", "glut", "oversupply", "crash", "price fall",
    "import", "low demand", "weak demand", "abundant supply", "slump",
    "export ban",
]


# ── Score classifier ─────────────────────────────────────────────────────────

def _classify_text(text: str) -> float:
    """
    Return a score in [-1, 1] based on keyword hit-count in `text`.
    Positive = bullish (price likely to rise), negative = bearish.
    """
    text_lower = text.lower()
    bullish_hits = sum(1 for kw in _BULLISH_KEYWORDS if kw in text_lower)
    bearish_hits = sum(1 for kw in _BEARISH_KEYWORDS if kw in text_lower)
    total = bullish_hits + bearish_hits
    if total == 0:
        return 0.0
    return (bullish_hits - bearish_hits) / total


# ── DuckDuckGo RSS fetch (primary) ───────────────────────────────────────────

_DDG_RSS = "https://duckduckgo.com/rss/news?q={query}&kl=in-en"
_TITLE_RE = re.compile(r"<title><!\[CDATA\[(.+?)]]></title>")
_SNIP_RE  = re.compile(r"<description><!\[CDATA\[(.+?)]]></description>")


async def _fetch_ddg_news(query: str, max_items: int = 10) -> list[NewsItem]:
    """Fetch headline + snippet from the DuckDuckGo news RSS feed."""
    _DDG_RSS.format(query=httpx.URL(query).path)
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(
                "https://duckduckgo.com/rss/news",
                params={"q": query, "kl": "in-en"},
                headers={"User-Agent": "CropFreshBot/1.0"},
            )
            resp.raise_for_status()
            xml = resp.text
    except Exception as e:
        logger.warning(f"DuckDuckGo RSS fetch failed: {e}")
        return []

    titles   = _TITLE_RE.findall(xml)[1:]   # skip channel title
    snippets = _SNIP_RE.findall(xml)

    items: list[NewsItem] = []
    for i, title in enumerate(titles[:max_items]):
        snippet = snippets[i] if i < len(snippets) else ""
        items.append(NewsItem(title=title, snippet=snippet, source="duckduckgo"))

    return items


# ── Main scraper ─────────────────────────────────────────────────────────────

class NewsSentimentScraper:
    """
    Fetches recent agricultural news and returns a commodity sentiment signal.

    Usage::

        scraper = NewsSentimentScraper()
        result  = await scraper.get_sentiment("tomato", "Karnataka")
        print(result.score, result.label)
    """

    async def get_sentiment(
        self,
        commodity: str,
        location: str = "India",
        max_articles: int = 10,
    ) -> CommoditySentiment:
        """
        Fetch & score recent news for `commodity` in `location`.

        Returns a CommoditySentiment with a score from -1 (bearish) to +1 (bullish).
        Falls back to neutral score (0.0) if no news can be fetched.
        """
        query = f"{commodity} price market India {location} agriculture"
        items = await _fetch_ddg_news(query, max_items=max_articles)

        if not items:
            logger.info(f"No news fetched for {commodity} — returning neutral sentiment")
            return CommoditySentiment(
                commodity=commodity,
                score=0.0,
                label="neutral",
                top_headlines=[],
                sample_count=0,
            )

        scores = [_classify_text(item.title + " " + item.snippet) for item in items]
        avg_score = sum(scores) / len(scores)

        if avg_score > 0.15:
            label = "bullish"
        elif avg_score < -0.15:
            label = "bearish"
        else:
            label = "neutral"

        headlines = [item.title for item in items[:5]]

        logger.info(
            f"News sentiment for {commodity}: {label} (score={avg_score:.2f}, n={len(items)})"
        )

        return CommoditySentiment(
            commodity=commodity,
            score=round(avg_score, 3),
            label=label,
            top_headlines=headlines,
            sample_count=len(items),
        )
