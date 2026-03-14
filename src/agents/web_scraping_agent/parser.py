"""
HTML Parser Mixin
=================
Converts raw HTML to Markdown strings gracefully.
"""

import re
from loguru import logger
from readability import Document
from markdownify import markdownify as md


class HTMLParserMixin:
    """Mixin for translating HTML into cleaner Markdown formats."""

    def _html_to_markdown(self, html: str) -> str:
        """Convert HTML to clean markdown using Readability and Markdownify."""
        if not html or len(html.strip()) == 0:
            return ""

        try:
            # Bolierplate Removal via Readability
            doc = Document(html)
            main_html = doc.summary()
            
            # Markdownify
            markdown = md(
                main_html, 
                strip=['a', 'img', 'script', 'style'], 
                heading_style="ATX",
                bullets="-",
            )
            
            # Clean up excessive newlines
            markdown = re.sub(r'\n{3,}', '\n\n', markdown)
            return markdown.strip()
            
        except Exception as e:
            logger.warning(f"Readability/Markdownify failed, falling back to basic extraction: {e}")
            # Fallback regex filter
            html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
            html = re.sub(r'<[^>]+>', ' ', html)
            html = re.sub(r'\s+', ' ', html)
            return html.strip()
