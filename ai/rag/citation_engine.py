"""
Citation Engine — Source Attribution for RAG Answers
====================================================
Adds inline [1], [2] citation markers to LLM-generated answers
and appends a Sources section for traceability.

Critical for farmer trust — they need to know where advice comes from.
Reference: ADR-010 — Advanced Agentic RAG System
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field


class Source(BaseModel):
    """A single source document reference."""
    index: int
    title: str = "Unknown Source"
    text_snippet: str = ""
    doc_id: str = ""
    confidence: float = 1.0


class CitedAnswer(BaseModel):
    """Answer with inline citations and source list."""
    answer: str
    sources: list[Source] = Field(default_factory=list)
    citation_count: int = 0
    all_verified: bool = False
    raw_answer: str = ""


CITATION_PROMPT = """You are a citation specialist for an Indian agricultural AI assistant.

Given an answer and source documents, add inline citation markers [1], [2], etc.
to each factual claim in the answer. Each marker should reference the document
that supports that claim.

Rules:
- Only cite claims that are directly supported by the documents
- Use [1], [2], etc. matching the document numbers below
- If a claim has no supporting document, mark it with [?]
- Do not change the answer content, only add citation markers

Documents:
{documents}

Answer to cite:
{answer}

Return ONLY the answer text with [N] markers inserted after each factual claim."""


class CitationEngine:
    """
    Adds source attribution and inline citations to RAG answers.

    Usage:
        engine = CitationEngine(llm=llm_provider)
        cited = await engine.add_citations(answer, documents)
        if cited.all_verified:
            return cited.answer
    """

    def __init__(self, llm: Any = None):
        self.llm = llm

    async def add_citations(
        self,
        answer: str,
        documents: list[Any],
    ) -> CitedAnswer:
        """
        Add inline [1], [2] citations and Sources section to an answer.

        Args:
            answer: Raw LLM-generated answer
            documents: Source documents used for generation

        Returns:
            CitedAnswer with markers and source list
        """
        if not answer or not documents:
            return CitedAnswer(answer=answer, raw_answer=answer)

        sources = self._build_sources(documents)

        if self.llm is not None:
            cited_text = await self._cite_with_llm(answer, documents)
        else:
            cited_text = self._cite_heuristic(answer, documents)

        citation_count = len(re.findall(r"\[\d+\]", cited_text))

        return CitedAnswer(
            answer=cited_text,
            sources=sources,
            citation_count=citation_count,
            all_verified=citation_count > 0 and "[?]" not in cited_text,
            raw_answer=answer,
        )

    def _build_sources(self, documents: list[Any]) -> list[Source]:
        """Extract source metadata from documents."""
        sources = []
        for i, doc in enumerate(documents):
            text = getattr(doc, "text", str(doc))
            metadata = getattr(doc, "metadata", {})
            title = metadata.get("title", metadata.get("source", f"Document {i + 1}"))
            doc_id = getattr(doc, "id", "")

            sources.append(Source(
                index=i + 1,
                title=str(title),
                text_snippet=text[:150].strip(),
                doc_id=str(doc_id),
            ))
        return sources

    async def _cite_with_llm(self, answer: str, documents: list[Any]) -> str:
        """Use LLM to add precise inline citations."""
        from src.orchestrator.llm_provider import LLMMessage

        docs_text = "\n\n".join(
            f"[{i + 1}] {getattr(doc, 'text', str(doc))[:500]}"
            for i, doc in enumerate(documents[:5])
        )
        prompt = CITATION_PROMPT.format(documents=docs_text, answer=answer)
        messages = [LLMMessage(role="user", content=prompt)]

        try:
            response = await self.llm.generate(
                messages, temperature=0.0, max_tokens=500,
            )
            cited = response.content.strip()
            if cited and len(cited) > len(answer) * 0.5:
                return cited
        except Exception as e:
            logger.warning(f"LLM citation failed: {e}")

        return self._cite_heuristic(answer, documents)

    def _cite_heuristic(self, answer: str, documents: list[Any]) -> str:
        """Keyword-overlap fallback for citation placement."""
        sentences = re.split(r"(?<=[.!?])\s+", answer)
        cited_sentences = []

        for sentence in sentences:
            s_words = set(sentence.lower().split())
            best_idx, best_score = 0, 0.0

            for i, doc in enumerate(documents[:5]):
                doc_words = set(getattr(doc, "text", str(doc)).lower().split())
                if not s_words:
                    continue
                overlap = len(s_words & doc_words) / len(s_words)
                if overlap > best_score:
                    best_score = overlap
                    best_idx = i

            # ? Only add citation if overlap is meaningful
            if best_score > 0.3:
                cited_sentences.append(f"{sentence} [{best_idx + 1}]")
            else:
                cited_sentences.append(sentence)

        return " ".join(cited_sentences)

    def format_sources_section(self, sources: list[Source]) -> str:
        """Format the Sources footer for display."""
        if not sources:
            return ""
        lines = ["\n\n**Sources:**"]
        for s in sources:
            lines.append(f"[{s.index}] {s.title}")
        return "\n".join(lines)
