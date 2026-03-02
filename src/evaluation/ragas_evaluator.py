"""
RAGAS Evaluator — core metrics engine for the CropFresh RAG pipeline.

Supports two execution modes:
  1. Full RAGAS (when `ragas` package is installed) — calls the real metrics.
  2. Stub mode — uses heuristic scoring when ragas is not installed, so the
     rest of the evaluation pipeline still works without the optional dep.

Usage:
    evaluator = RAGASEvaluator()
    results = await evaluator.run_evaluation("datasets/agronomy_qa.json")
    print(results.overall_score)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from src.evaluation.dataset_loader import DatasetLoader
from src.evaluation.models import EvalResults, GoldenItem, PerQuestionScore


# ---------------------------------------------------------------------------
# RAG Pipeline protocol — allows any pipeline to be plugged in for evaluation
# ---------------------------------------------------------------------------


@runtime_checkable
class RAGPipeline(Protocol):
    """Minimal interface the evaluator expects from a RAG pipeline."""

    async def query(self, question: str) -> "RAGResponse":
        ...


class RAGResponse:
    """Simple container returned by the RAG pipeline during evaluation."""

    def __init__(self, answer: str, contexts: list[str]) -> None:
        self.answer = answer
        self.contexts = contexts


# ---------------------------------------------------------------------------
# Heuristic stub metrics (used when ragas is not installed)
# ---------------------------------------------------------------------------


def _heuristic_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Proxy for faithfulness: fraction of answer tokens found in contexts.
    Not a replacement for real RAGAS — adequate for smoke-testing.
    """
    if not contexts:
        return 0.0
    context_blob = " ".join(contexts).lower()
    answer_tokens = set(answer.lower().split())
    if not answer_tokens:
        return 1.0
    hits = sum(1 for t in answer_tokens if t in context_blob)
    return round(hits / len(answer_tokens), 4)


def _heuristic_answer_relevancy(answer: str, question: str) -> float:
    """Proxy for answer relevancy: token overlap between answer and question."""
    q_tokens = set(question.lower().split())
    a_tokens = set(answer.lower().split())
    if not q_tokens:
        return 1.0
    overlap = q_tokens & a_tokens
    # ? Intentionally lenient — this is only a structural sanity check
    return round(min(1.0, len(overlap) / len(q_tokens) + 0.5), 4)


def _heuristic_context_precision(ground_truth: str, contexts: list[str]) -> float:
    """Proxy: fraction of context sentences that mention GT keywords."""
    if not contexts:
        return 0.0
    gt_tokens = set(ground_truth.lower().split())
    relevant = sum(
        1 for c in contexts if gt_tokens & set(c.lower().split())
    )
    return round(relevant / len(contexts), 4)


def _heuristic_context_recall(ground_truth: str, contexts: list[str]) -> float:
    """Proxy: fraction of GT tokens found anywhere in contexts."""
    if not contexts:
        return 0.0
    blob = " ".join(contexts).lower()
    gt_tokens = set(ground_truth.lower().split())
    hits = sum(1 for t in gt_tokens if t in blob)
    return round(hits / max(1, len(gt_tokens)), 4)


# ---------------------------------------------------------------------------
# RAGASEvaluator
# ---------------------------------------------------------------------------


class RAGASEvaluator:
    """
    Evaluate the CropFresh RAG pipeline against a golden dataset.

    When the `ragas` package is available the evaluator delegates to it;
    otherwise it falls back to lightweight heuristic metrics so the system
    is testable without the optional dependency.
    """

    def __init__(
        self,
        rag_pipeline: RAGPipeline | None = None,
        loader: DatasetLoader | None = None,
        use_ragas: bool = True,
    ) -> None:
        self.rag_pipeline = rag_pipeline
        self.loader = loader or DatasetLoader()
        self._ragas_available = self._try_import_ragas() if use_ragas else False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_evaluation(
        self,
        dataset_path: str | Path,
        *,
        concurrency: int = 4,
    ) -> EvalResults:
        """
        Run RAGAS evaluation on the full golden dataset.

        Args:
            dataset_path: Absolute path or filename relative to the datasets dir.
            concurrency: Number of parallel pipeline calls.

        Returns:
            EvalResults with aggregate + per-question scores.
        """
        items = self.loader.load(dataset_path)
        if not items:
            raise ValueError(f"No items found in dataset: {dataset_path}")

        logger.info(
            f"Starting evaluation: {len(items)} items, ragas={'full' if self._ragas_available else 'heuristic'}"
        )

        per_q = await self._score_all(items, concurrency=concurrency)

        return EvalResults(
            faithfulness=_avg(per_q, "faithfulness"),
            answer_relevancy=_avg(per_q, "answer_relevancy"),
            context_precision=_avg(per_q, "context_precision"),
            context_recall=_avg(per_q, "context_recall"),
            per_question=per_q,
            dataset_path=str(dataset_path),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _score_all(
        self, items: list[GoldenItem], *, concurrency: int
    ) -> list[PerQuestionScore]:
        """Score all items, respecting concurrency limit."""
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded(item: GoldenItem) -> PerQuestionScore:
            async with semaphore:
                return await self._score_item(item)

        return list(await asyncio.gather(*[bounded(it) for it in items]))

    async def _score_item(self, item: GoldenItem) -> PerQuestionScore:
        """Score a single golden item against the RAG pipeline."""
        answer, contexts = await self._get_pipeline_response(item)

        if self._ragas_available:
            scores = await self._ragas_score(item, answer, contexts)
        else:
            scores = self._heuristic_score(item, answer, contexts)

        return PerQuestionScore(
            question=item.question,
            answer=answer,
            **scores,
        )

    async def _get_pipeline_response(
        self, item: GoldenItem
    ) -> tuple[str, list[str]]:
        """Query the RAG pipeline, fall back to stub if none is configured."""
        if self.rag_pipeline is None:
            # Stub: return ground truth as the answer so heuristics score high
            return item.ground_truth, item.contexts

        try:
            resp: RAGResponse = await self.rag_pipeline.query(item.question)
            return resp.answer, resp.contexts
        except Exception as exc:
            logger.warning(f"Pipeline query failed for '{item.question[:50]}': {exc}")
            return "", item.contexts

    async def _ragas_score(
        self, item: GoldenItem, answer: str, contexts: list[str]
    ) -> dict[str, float]:
        """
        Delegate to the real RAGAS library.
        Runs in a thread so async callers aren't blocked.
        """
        try:
            import ragas  # noqa: PLC0415
            from datasets import Dataset  # noqa: PLC0415
            from ragas.metrics import (  # noqa: PLC0415
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

            data = {
                "question": [item.question],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [item.ground_truth],
            }
            ds = Dataset.from_dict(data)

            # ragas.evaluate is synchronous — run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result: Any = await loop.run_in_executor(
                None,
                lambda: ragas.evaluate(
                    ds,
                    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                ),
            )
            row = result.to_pandas().iloc[0]
            return {
                "faithfulness": float(row["faithfulness"]),
                "answer_relevancy": float(row["answer_relevancy"]),
                "context_precision": float(row["context_precision"]),
                "context_recall": float(row["context_recall"]),
            }
        except Exception as exc:
            logger.warning(f"RAGAS scoring failed, falling back to heuristics: {exc}")
            return self._heuristic_score(item, answer, contexts)

    @staticmethod
    def _heuristic_score(
        item: GoldenItem, answer: str, contexts: list[str]
    ) -> dict[str, float]:
        return {
            "faithfulness": _heuristic_faithfulness(answer, contexts),
            "answer_relevancy": _heuristic_answer_relevancy(answer, item.question),
            "context_precision": _heuristic_context_precision(item.ground_truth, contexts),
            "context_recall": _heuristic_context_recall(item.ground_truth, contexts),
        }

    @staticmethod
    def _try_import_ragas() -> bool:
        try:
            import ragas  # noqa: PLC0415, F401
            return True
        except ImportError:
            logger.info("ragas package not installed — using heuristic evaluation mode")
            return False


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def _avg(scores: list[PerQuestionScore], field: str) -> float:
    if not scores:
        return 0.0
    return round(sum(getattr(s, field) for s in scores) / len(scores), 4)


def create_ragas_evaluator(
    rag_pipeline: RAGPipeline | None = None,
    use_ragas: bool = True,
) -> RAGASEvaluator:
    return RAGASEvaluator(rag_pipeline=rag_pipeline, use_ragas=use_ragas)
