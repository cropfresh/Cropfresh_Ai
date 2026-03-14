"""
Scraper Cache Mixin
===================
Handles disk caching of scraped pages.
"""

import hashlib
import json
from datetime import datetime
from typing import Optional
from loguru import logger

from .models import ScrapingResult
from .base import BaseWebScraper


class ScraperCacheMixin(BaseWebScraper):
    """Mixin providing disk-based caching functionalities."""

    def _url_hash(self, url: str) -> str:
        """Generate hash for URL (for caching)."""
        return hashlib.md5(url.encode()).hexdigest()[:12]

    def _get_cached(self, url: str) -> Optional[ScrapingResult]:
        """Get cached result if available and not expired."""
        cache_file = self.cache_dir / f"{self._url_hash(url)}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            cached_time = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - cached_time > self.cache_ttl:
                return None  # Cache expired

            result = ScrapingResult(**data)
            result.cached = True
            return result

        except Exception as e:
            logger.warning("Cache read failed: {}", str(e))
            return None

    def _set_cached(self, url: str, result: ScrapingResult) -> None:
        """Cache scraping result."""
        cache_file = self.cache_dir / f"{self._url_hash(url)}.json"

        try:
            data = result.model_dump()
            data['timestamp'] = result.timestamp.isoformat()

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.warning("Cache write failed: {}", str(e))
