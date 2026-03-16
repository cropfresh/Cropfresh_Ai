from __future__ import annotations

import asyncio
from collections import defaultdict

from loguru import logger

from src.evaluation.models import EvalResults, GoldenItem, PerQuestionScore
from src.evaluation.ragas_evaluator import RAGASEvaluator
from src.rag.benchmark.models import GoldenEntry, LiveRunExtras, ResolvedReference


class LiveBenchmarkRunner:
    """Run semantic benchmark evaluation against the canonical runtime path."""

    def __init__(self, pipeline, scorer: RAGASEvaluator | None = None):
        self.pipeline = pipeline
        self.scorer = scorer or RAGASEvaluator(use_ragas=True)
        self.ragas_timeout_seconds = 15

    async def run(self, entries: list[GoldenEntry], resolver) -> tuple[EvalResults, LiveRunExtras]:
        per_question: list[PerQuestionScore] = []
        category_scores: dict[str, list[float]] = defaultdict(list)
        cited = 0
        live_checks = 0
        fresh_checks = 0
        failures: list[str] = []

        for entry in entries:
            reference = await resolver.resolve(entry)
            debug_result = await self.pipeline.answer(entry.query)
            item = self._to_golden_item(entry, reference)
            scores = await self._score_item(item, debug_result.answer, debug_result.contexts)
            per_question.append(
                PerQuestionScore(question=item.question, answer=debug_result.answer, **scores)
            )
            category_scores[entry.category].append(sum(scores.values()) / len(scores))
            if debug_result.citations:
                cited += 1
            if entry.mode == "live":
                live_checks += 1
                if _is_fresh_live_answer(reference, debug_result.source_details):
                    fresh_checks += 1
                else:
                    extras_failure = f"{entry.id}: missing or stale live source metadata"
                    if extras_failure not in failures:
                        failures.append(extras_failure)

        results = EvalResults(
            faithfulness=_avg(per_question, "faithfulness"),
            answer_relevancy=_avg(per_question, "answer_relevancy"),
            context_precision=_avg(per_question, "context_precision"),
            context_recall=_avg(per_question, "context_recall"),
            per_question=per_question,
            dataset_path="ai/rag/evaluation/datasets",
        )
        extras = LiveRunExtras(
            citation_coverage=round(cited / max(len(entries), 1), 4),
            freshness_compliance=round(fresh_checks / max(live_checks, 1), 4) if live_checks else 1.0,
            per_category={key: round(sum(values) / len(values), 4) for key, values in category_scores.items()},
            failures=failures,
        )
        return results, extras

    async def _score_item(
        self,
        item: GoldenItem,
        answer: str,
        contexts: list[str],
    ) -> dict[str, float]:
        if self.scorer._ragas_available:
            try:
                return await asyncio.wait_for(
                    self.scorer._ragas_score(item, answer, contexts),
                    timeout=self.ragas_timeout_seconds,
                )
            except Exception as exc:
                logger.warning("Live benchmark falling back to heuristic scoring: {}", exc)
                self.scorer._ragas_available = False
        return self.scorer._heuristic_score(item, answer, contexts)

    @staticmethod
    def _to_golden_item(entry: GoldenEntry, reference: ResolvedReference) -> GoldenItem:
        return GoldenItem(
            question=entry.query,
            ground_truth=reference.ground_truth,
            contexts=reference.contexts,
            agent_domain=entry.category,
            difficulty=entry.difficulty,
            language=entry.language,
        )


def _avg(scores: list[PerQuestionScore], field: str) -> float:
    if not scores:
        return 0.0
    return round(sum(getattr(score, field) for score in scores) / len(scores), 4)


def _is_fresh_live_answer(reference: ResolvedReference, source_details: list) -> bool:
    if not reference.freshness_ok or not source_details:
        return False
    fresh_flags = [detail.is_fresh for detail in source_details if detail.source]
    return bool(fresh_flags) and all(flag is True for flag in fresh_flags)
