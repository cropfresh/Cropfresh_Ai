"""
Contextual Chunker Splitter Mixin
=================================
Mixin for logic regarding chunk boundaries, simple and semantic.
"""

import re
from typing import Any


class SplitterMixin:
    """Mixin for text chunking mechanisms."""
    
    # Assumes self.config is available

    def _simple_chunk(self, text: str) -> list[tuple[str, int, int]]:
        """Simple character-based chunking with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.config.chunk_size
            if end < len(text):
                search_start = max(end - 100, start)
                best_break = end
                for break_char in ['. ', '.\n', '! ', '? ', '\n\n']:
                    pos = text.rfind(break_char, search_start, end + 50)
                    if pos > start:
                        best_break = pos + len(break_char)
                        break
                end = best_break
            chunk_text = text[start:end].strip()
            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append((chunk_text, start, end))
            start = end - self.config.chunk_overlap
            start = max(start, end - self.config.chunk_size // 2)
        return chunks

    def _semantic_chunk(self, text: str) -> list[tuple[str, int, int]]:
        """
        Semantic boundary-aware chunking.
        Respects paragraph boundaries, section headers, list items, etc.
        """
        chunks = []
        paragraphs = re.split(r'\n\s*\n', text)
        current_chunk = ""
        current_start = 0
        char_pos = 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                char_pos += 2
                continue
            if len(current_chunk) + len(para) + 2 > self.config.chunk_size:
                if len(current_chunk) >= self.config.min_chunk_size:
                    chunks.append((
                        current_chunk.strip(),
                        current_start,
                        current_start + len(current_chunk),
                    ))
                current_chunk = para
                current_start = char_pos
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                    current_start = char_pos
            char_pos += len(para) + 2
        
        if len(current_chunk) >= self.config.min_chunk_size:
            chunks.append((
                current_chunk.strip(),
                current_start,
                current_start + len(current_chunk),
            ))
        return chunks
