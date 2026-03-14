"""
Content Extractors and Quality Filters for Browser RAG.
"""

import re
from typing import Optional

from loguru import logger


class ContentExtractor:
    """
    Extracts clean text from scraped HTML using domain-specific CSS selectors.

    Falls back to a generic broad-content selector when domain-specific
    selectors fail or no match is found.
    """

    # Domain-specific CSS selectors
    DOMAIN_SELECTORS: dict[str, str] = {
        "farmer.gov.in":         ".content-area, #mainContent, .text",
        "icar.org.in":           ".views-row, .field-items, article",
        "agrifarming.in":        "article, .post-content, .entry-content",
        "commodityindia.com":    ".news-list, .article-body, .post",
        "agriwatch.com":         ".entry-content, .post-content",
        "krishijagran.com":      "article, .story-content, .content",
        "mausam.imd.gov.in":    ".agromet, .content, #main-content",
        "apeda.gov.in":          ".news, .content, table",
        "cibrc.nic.in":          "table, .content, #content",
        "_default":              "article, main, .content, .post, #content, p",
    }

    def extract_text(self, html: str, domain: str, css_override: Optional[str] = None) -> str:
        """
        Extract main text content from HTML.

        Args:
            html: Raw HTML content
            domain: Source domain (for selector lookup)
            css_override: Optional CSS selector from TargetSource

        Returns:
            Extracted clean text
        """
        try:
            from scrapling import Adaptor

            adaptor = Adaptor(html, auto_match=False)
            selector = css_override or self.DOMAIN_SELECTORS.get(
                domain, self.DOMAIN_SELECTORS["_default"]
            )

            # Try domain-specific selector
            elements = adaptor.css(selector)
            if elements:
                text = " ".join(el.text for el in elements if el.text)
                return self._clean_text(text)

            # Fallback to broad paragraph extraction
            paragraphs = adaptor.css("p")
            text = " ".join(p.text for p in paragraphs if p.text)
            return self._clean_text(text)

        except ImportError:
            logger.warning("Scrapling not installed — falling back to regex extraction")
            return self._regex_extract(html)

        except Exception as e:
            logger.warning(f"ContentExtractor: extraction failed for {domain}: {e}")
            return self._regex_extract(html)

    def _clean_text(self, text: str) -> str:
        """Remove excessive whitespace and normalize text."""
        # Remove multiple spaces/newlines
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove common noise patterns
        text = re.sub(r'(Cookie Policy|Accept Cookies|Privacy Policy).*', '', text)
        return text[:5000]  # Cap at 5000 chars

    def _regex_extract(self, html: str) -> str:
        """Fallback: extract text between paragraph tags using regex."""
        paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', html, re.DOTALL | re.IGNORECASE)
        text = ' '.join(re.sub(r'<[^>]+>', '', p) for p in paragraphs)
        return self._clean_text(text)


class QualityFilter:
    """
    Filters out low-quality or error pages from scraped content.

    Applies minimum word count and error-page detection heuristics.
    """

    MIN_WORDS = 150
    ERROR_PATTERNS = [
        "404", "page not found", "access denied", "forbidden",
        "service unavailable", "captcha", "cloudflare",
        "we noticed unusual activity", "please enable javascript",
    ]

    def is_valid(self, text: str) -> bool:
        """
        Check if scraped text is valid content.

        Args:
            text: Extracted text

        Returns:
            True if content passes quality threshold
        """
        if not text or len(text.split()) < self.MIN_WORDS:
            return False

        text_lower = text.lower()
        if any(pattern in text_lower for pattern in self.ERROR_PATTERNS):
            return False

        return True
