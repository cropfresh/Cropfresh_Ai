"""
Contextual Chunk Enrichment — Anthropic-style context injection (ADR-010 Phase 4).

Implements Anthropic's Contextual Retrieval technique:
  1. For each chunk, generate a short context summary from the full document
  2. Prepend this summary to the chunk text before embedding
  3. Reduces retrieval failure from 15% to ~5% (Anthropic benchmark)

Reference: https://www.anthropic.com/news/contextual-retrieval
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


class ContextualChunk(BaseModel):
    """A chunk enriched with document-level context."""

    original_text: str
    context_prefix: str = ""
    enriched_text: str = ""
    chunk_index: int = 0
    source_doc_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        """Return the enriched text for embedding."""
        return self.enriched_text or self.original_text


class ContextualChunkEnricher:
    """Enriches chunks with document-level context before embedding.

    Instead of embedding isolated chunks, each chunk gets a short
    context prefix explaining where it fits in the larger document.
    This dramatically improves retrieval accuracy.
    """

    # ? Context prefix template for heuristic mode
    CONTEXT_TEMPLATE = (
        "This excerpt is from a document about {topic}. "
        "Document section: {section}. "
    )

    def __init__(self, llm: Any = None):
        """Initialize enricher with optional LLM for context generation."""
        self.llm = llm

    async def enrich_chunks(
        self,
        chunks: list[str],
        full_document: str,
        doc_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> list[ContextualChunk]:
        """Enrich chunks with document-level context.

        Args:
            chunks: Raw text chunks from a document.
            full_document: The original full document text.
            doc_id: Document identifier.
            metadata: Optional document metadata.

        Returns:
            List of ContextualChunk with context prefix prepended.
        """
        if not chunks:
            return []

        meta = metadata or {}
        topic = meta.get("title", meta.get("category", "agriculture"))
        enriched: list[ContextualChunk] = []

        for idx, chunk_text in enumerate(chunks):
            section = self._infer_section(chunk_text, idx, len(chunks))

            if self.llm is not None:
                prefix = await self._llm_context(full_document, chunk_text)
            else:
                prefix = self.CONTEXT_TEMPLATE.format(
                    topic=topic, section=section,
                )

            enriched_text = f"{prefix}{chunk_text}"
            enriched.append(ContextualChunk(
                original_text=chunk_text,
                context_prefix=prefix,
                enriched_text=enriched_text,
                chunk_index=idx,
                source_doc_id=doc_id,
                metadata=meta,
            ))

        logger.info(
            f"ContextualChunkEnricher: enriched {len(enriched)} chunks | "
            f"doc={doc_id} | topic={topic}"
        )
        return enriched

    def _infer_section(self, chunk: str, index: int, total: int) -> str:
        """Heuristic section inference from content and position."""
        if index == 0:
            return "introduction"
        if index >= total - 1:
            return "conclusion"

        lower = chunk.lower()
        #! Agriculture-specific section markers
        section_map = {
            "pest management": ["pest", "disease", "insect", "fungus"],
            "soil and nutrition": ["soil", "nutrient", "fertiliz", "compost"],
            "irrigation": ["irrigat", "water", "rainfall", "drip"],
            "market information": ["price", "market", "mandi", "cost"],
            "seed selection": ["seed", "variety", "cultivar", "hybrid"],
            "harvest and post-harvest": ["harvest", "yield", "post-harvest"],
            "government schemes": ["scheme", "subsidy", "government", "pm-kisan"],
        }
        for section, keywords in section_map.items():
            if any(kw in lower for kw in keywords):
                return section

        return f"section {index + 1}"

    async def _llm_context(self, full_doc: str, chunk: str) -> str:
        """Generate context prefix using LLM."""
        try:
            from src.orchestrator.llm_provider import LLMMessage

            prompt = (
                f"Given this document:\n{full_doc[:2000]}\n\n"
                f"Write a 1-sentence context for this chunk:\n{chunk[:500]}\n\n"
                f"Format: 'This excerpt is from... and discusses...'"
            )
            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.llm.generate(
                messages, temperature=0.0, max_tokens=80,
            )
            return response.content.strip() + " "
        except Exception as e:
            logger.warning(f"LLM context generation failed: {e}")
            return ""
