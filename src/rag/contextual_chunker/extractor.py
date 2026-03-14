"""
Contextual Chunker Extractor Mixin
==================================
Mixin for extracting entities, keywords, and section headers.
"""

import re
from typing import Any

from .constants import ENTITY_PATTERNS


class ExtractorMixin:
    """Mixin for extraction tasks within the chunker."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Assumes child class calls this and has initialized config
        self._compiled_patterns = {
            entity_type: [re.compile(p, re.IGNORECASE) for p in patterns]
            for entity_type, patterns in ENTITY_PATTERNS.items()
        }

    def _extract_section_headers(self, text: str) -> list[tuple[str, int]]:
        """Extract section headers and their positions."""
        headers = []
        patterns = [
            r'^(#{1,3})\s+(.+)$',  # Markdown headers
            r'^([A-Z][A-Z\s]+):?\s*$',  # ALL CAPS HEADERS
            r'^(\d+\.\s+.+)$',  # Numbered sections
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                header_text = match.group(0).strip()
                header_text = re.sub(r'^#+\s*', '', header_text)
                headers.append((header_text, match.start()))
        headers.sort(key=lambda x: x[1])
        return headers

    def _find_section_for_position(
        self,
        headers: list[tuple[str, int]],
        position: int,
    ) -> str:
        """Find the section header that applies to a given position."""
        applicable_header = ""
        for header_text, header_pos in headers:
            if header_pos <= position:
                applicable_header = header_text
            else:
                break
        return applicable_header

    def _extract_entities(self, text: str) -> list[str]:
        """Extract named entities from text."""
        entities = []
        for entity_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entity = f"{entity_type}:{match.group()}"
                    if entity not in entities:
                        entities.append(entity)
        return entities[:20]

    def _extract_keywords(self, text: str, max_keywords: int = 10) -> list[str]:
        """Extract key phrases from text based on frequency."""
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        stop_words = {
            'that', 'this', 'with', 'from', 'have', 'been', 'were', 'will',
            'they', 'their', 'there', 'which', 'about', 'would', 'could',
            'should', 'does', 'these', 'those', 'after', 'before',
        }
        filtered = [w for w in words if w not in stop_words]
        freq: dict[str, int] = {}
        for word in filtered:
            freq[word] = freq.get(word, 0) + 1
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:max_keywords]]
