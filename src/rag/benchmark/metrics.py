"""
RAGAS Evaluation Metrics — Multi-dimensional RAG quality scoring (ADR-010 Phase 5).

Implements RAGAS-style metrics for CropFresh RAG evaluation:
  - Faithfulness: Is every claim in the answer supported by context?
  - Answer Relevancy: Does the answer address the question?
  - Context Precision: Are retrieved docs actually useful?
  - Hallucination Rate: % of unsupported claims

Each metric returns 0.0–1.0. Combined into a weighted composite score.
"""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


class EvalMetrics(BaseModel):
    """Multi-dimensional evaluation result."""

    faithfulness: float = Field(ge=0.0, le=1.0, default=0.0)
    answer_relevancy: float = Field(ge=0.0, le=1.0, default=0.0)
    context_precision: float = Field(ge=0.0, le=1.0, default=0.0)
    hallucination_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    composite_score: float = Field(ge=0.0, le=1.0, default=0.0)
    reasoning: str = ""


class RAGEvaluator:
    """Multi-metric RAGAS-style evaluator for CropFresh RAG.

    Weights (configurable):
      faithfulness:       0.40  (most critical — anti-hallucination)
      answer_relevancy:   0.30
      context_precision:  0.20
      hallucination_rate: 0.10  (penalty, subtracted)
    """

    WEIGHTS = {
        "faithfulness": 0.40,
        "answer_relevancy": 0.30,
        "context_precision": 0.20,
        "hallucination_penalty": 0.10,
    }

    def __init__(self, llm: Any = None):
        self.llm = llm

    async def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str = "",
    ) -> EvalMetrics:
        """Run all RAGAS metrics on a single QA pair.

        Args:
            query: User question.
            answer: Generated answer.
            contexts: Retrieved context passages.
            ground_truth: Expected answer (if available).

        Returns:
            EvalMetrics with all scores.
        """
        if self.llm is not None:
            return await self._llm_evaluate(
                query, answer, contexts, ground_truth,
            )
        return self._heuristic_evaluate(
            query, answer, contexts, ground_truth,
        )

    def _heuristic_evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str,
    ) -> EvalMetrics:
        """Heuristic evaluation when LLM is unavailable."""
        faith = self._heuristic_faithfulness(answer, contexts)
        relevancy = self._heuristic_relevancy(query, answer)
        precision = self._heuristic_context_precision(query, contexts)
        hallucination = max(0.0, 1.0 - faith)

        composite = (
            self.WEIGHTS["faithfulness"] * faith
            + self.WEIGHTS["answer_relevancy"] * relevancy
            + self.WEIGHTS["context_precision"] * precision
            - self.WEIGHTS["hallucination_penalty"] * hallucination
        )

        return EvalMetrics(
            faithfulness=round(faith, 3),
            answer_relevancy=round(relevancy, 3),
            context_precision=round(precision, 3),
            hallucination_rate=round(hallucination, 3),
            composite_score=round(max(0.0, composite), 3),
            reasoning="heuristic evaluation",
        )

    def _heuristic_faithfulness(
        self, answer: str, contexts: list[str],
    ) -> float:
        """Estimate faithfulness by keyword overlap with contexts."""
        if not contexts or not answer:
            return 0.0

        answer_words = set(re.findall(r"\w+", answer.lower()))
        context_words: set[str] = set()
        for ctx in contexts:
            context_words.update(re.findall(r"\w+", ctx.lower()))

        #! Remove common stopwords for better signal
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "in", "on",
            "at", "to", "for", "of", "and", "or", "not", "it", "this",
            "that", "with", "from", "by", "as", "be", "has", "have",
        }
        answer_content = answer_words - stopwords
        context_content = context_words - stopwords

        if not answer_content:
            return 0.5

        overlap = len(answer_content & context_content)
        return min(1.0, overlap / len(answer_content))

    def _heuristic_relevancy(self, query: str, answer: str) -> float:
        """Estimate answer relevancy by query keyword coverage."""
        if not query or not answer:
            return 0.0

        query_words = set(re.findall(r"\w+", query.lower()))
        answer_words = set(re.findall(r"\w+", answer.lower()))

        stopwords = {"what", "how", "when", "where", "is", "the", "a"}
        query_content = query_words - stopwords
        if not query_content:
            return 0.5

        covered = len(query_content & answer_words)
        return min(1.0, covered / len(query_content))

    def _heuristic_context_precision(
        self, query: str, contexts: list[str],
    ) -> float:
        """Estimate how many retrieved contexts are actually useful."""
        if not contexts:
            return 0.0

        query_words = set(re.findall(r"\w+", query.lower()))
        useful = 0
        for ctx in contexts:
            ctx_words = set(re.findall(r"\w+", ctx.lower()))
            overlap = len(query_words & ctx_words)
            if overlap >= 2:
                useful += 1

        return useful / len(contexts)

    async def _llm_evaluate(
        self, query: str, answer: str,
        contexts: list[str], ground_truth: str,
    ) -> EvalMetrics:
        """LLM-based evaluation for production accuracy."""
        try:
            from src.orchestrator.llm_provider import LLMMessage

            ctx_text = "\n---\n".join(c[:500] for c in contexts[:5])
            prompt = (
                "Evaluate this RAG response. Return JSON only:\n"
                f"Question: {query}\nAnswer: {answer}\n"
                f"Context: {ctx_text}\n"
                '{"faithfulness": 0.0-1.0, "answer_relevancy": 0.0-1.0, '
                '"context_precision": 0.0-1.0, "reasoning": "..."}'
            )
            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.llm.generate(
                messages, temperature=0.0, max_tokens=200,
            )
            result = json.loads(response.content)

            faith = float(result.get("faithfulness", 0.5))
            rel = float(result.get("answer_relevancy", 0.5))
            prec = float(result.get("context_precision", 0.5))
            halluc = max(0.0, 1.0 - faith)

            composite = (
                self.WEIGHTS["faithfulness"] * faith
                + self.WEIGHTS["answer_relevancy"] * rel
                + self.WEIGHTS["context_precision"] * prec
                - self.WEIGHTS["hallucination_penalty"] * halluc
            )

            return EvalMetrics(
                faithfulness=faith,
                answer_relevancy=rel,
                context_precision=prec,
                hallucination_rate=halluc,
                composite_score=round(max(0.0, composite), 3),
                reasoning=result.get("reasoning", ""),
            )
        except Exception as e:
            logger.warning(f"LLM evaluation failed: {e}")
            return self._heuristic_evaluate(
                query, answer, contexts, ground_truth,
            )
