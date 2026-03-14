"""
RSS News Scraper
================
Scraper for agricultural news via RSS feeds.
"""

from datetime import datetime

from loguru import logger

from .constants import SOURCE_URLS, DataSource
from .models import NewsArticle


class RSSNewsScraper:
    """
    Scraper for agricultural news via RSS feeds.

    Sources:
    - Rural Voice
    - Krishak Jagat
    - Agri Farming
    """

    RSS_FEEDS = {
        "rural_voice": SOURCE_URLS[DataSource.RURAL_VOICE],
        "agri_farming": "https://agrifarming.in/feed",
    }

    def __init__(self):
        try:
            import feedparser
            self._feedparser = feedparser
        except ImportError:
            logger.warning("feedparser not installed — install via: pip install feedparser")
            self._feedparser = None

    async def get_news(
        self,
        source: str = "rural_voice",
        limit: int = 10,
    ) -> list[NewsArticle]:
        """Get agricultural news articles from RSS feed."""
        if not self._feedparser:
            logger.error("feedparser not available")
            return []

        if source not in self.RSS_FEEDS:
            logger.error(f"Unknown news source: {source}")
            return []

        try:
            feed = self._feedparser.parse(self.RSS_FEEDS[source])
            articles = []

            for entry in feed.entries[:limit]:
                articles.append(
                    NewsArticle(
                        title=entry.get("title", "Untitled"),
                        summary=entry.get("summary", ""),
                        url=entry.get("link", ""),
                        source=source,
                        published_date=datetime.now(),
                    )
                )

            logger.info(f"Fetched {len(articles)} articles from {source}")
            return articles

        except Exception as e:
            logger.error(f"RSS feed parsing failed: {e}")
            return []
