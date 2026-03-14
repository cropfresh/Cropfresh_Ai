"""
Contextual Chunker LLM Context Mixin
====================================
Mixin for generating enriched contexts utilizing an LLM or simple heuristics.
"""

from loguru import logger

from .constants import CONTEXT_PROMPT


class LLMContextMixin:
    """Mixin to handle context generation for chunks."""

    # Assumes self.llm and self.config are available

    async def _generate_llm_context(
        self,
        chunk_text: str,
        title: str,
        source: str,
        summary: str,
    ) -> str:
        """Generate context using LLM."""
        try:
            prompt = CONTEXT_PROMPT.format(
                title=title or "Unknown",
                source=source or "Unknown",
                summary=summary or "Agricultural knowledge document",
                chunk=chunk_text[:1000],  # Limit chunk size
            )
            response = await self.llm.agenerate([prompt])
            context = response.generations[0][0].text.strip()

            if len(context) > self.config.context_max_length * 4:
                context = context[:self.config.context_max_length * 4] + "..."
            return context
        except Exception as e:
            logger.warning(f"LLM context generation failed: {e}")
            return self._generate_simple_context(chunk_text, title, source, "", 0, 1)

    def _generate_simple_context(
        self,
        chunk_text: str,
        title: str,
        source: str,
        section: str,
        index: int,
        total: int,
    ) -> str:
        """Generate simple rule-based context."""
        parts = []
        if title:
            parts.append(f"From '{title}'")
        if source:
            parts.append(f"source: {source}")
        if section:
            parts.append(f"in section: {section}")
        if total > 1:
            if index == 0:
                parts.append("(beginning of document)")
            elif index == total - 1:
                parts.append("(end of document)")
            else:
                parts.append(f"(part {index + 1} of {total})")
        return ". ".join(parts) if parts else ""
