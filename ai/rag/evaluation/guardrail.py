from __future__ import annotations

from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field

from ai.rag.evaluation.dataset_loader import BenchmarkDatasetLoader
from ai.rag.evaluation.live_runner import LiveBenchmarkRunner
from ai.rag.evaluation.metrics import RAGEvaluator
from ai.rag.evaluation.models import GoldenEntry
from ai.rag.evaluation.pipeline_adapter import KnowledgeAgentPipelineAdapter
from ai.rag.evaluation.reference_resolver import ReferenceResolver
from ai.rag.evaluation.reporting import write_guardrail_report
from ai.rag.evaluation.runtime import build_benchmark_agent
from src.evaluation.models import EvalResults
from src.evaluation.ragas_evaluator import RAGASEvaluator


class GuardrailThresholds(BaseModel):
    """Minimum thresholds for live and heuristic benchmark gates."""

    min_faithfulness: float = 0.93
    min_answer_relevancy: float = 0.90
    min_context_precision: float = 0.85
    min_context_recall: float = 0.90
    min_citation_coverage: float = 0.95
    min_freshness_compliance: float = 1.0
    max_hallucination_rate: float = 0.07
    min_composite_score: float = 0.90

    @classmethod
    def for_mode(cls, mode: Literal["live", "heuristic"]) -> "GuardrailThresholds":
        if mode == "heuristic":
            return cls(
                min_faithfulness=0.35,
                min_answer_relevancy=0.30,
                min_context_precision=0.40,
                min_context_recall=0.0,
                min_citation_coverage=0.20,
                min_freshness_compliance=0.50,
                max_hallucination_rate=0.70,
                min_composite_score=0.30,
            )
        return cls()


class GuardrailResult(BaseModel):
    """Single benchmark run verdict."""

    mode: Literal["live", "heuristic"]
    subset: str
    passed: bool = False
    total_queries: int = 0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float | None = None
    hallucination_rate: float = 0.0
    composite_score: float = 0.0
    citation_coverage: float = 0.0
    freshness_compliance: float = 0.0
    failures: list[str] = Field(default_factory=list)
    per_category: dict[str, float] = Field(default_factory=dict)
    report_path: str = ""


class EvalGuardrail:
    """Evaluate the canonical runtime against the JSON-backed benchmark datasets."""

    def __init__(
        self,
        dataset_loader: BenchmarkDatasetLoader | None = None,
        resolver: ReferenceResolver | None = None,
        thresholds: GuardrailThresholds | None = None,
        agent=None,
    ):
        self.dataset_loader = dataset_loader or BenchmarkDatasetLoader()
        self.resolver = resolver or ReferenceResolver()
        self.thresholds = thresholds
        self.agent = agent
        self._pipeline = None

    async def run(
        self,
        mode: Literal["live", "heuristic"] = "live",
        subset: str = "core_live",
        report_path: str | Path | None = None,
    ) -> GuardrailResult:
        entries = self.dataset_loader.load(subset)
        semantic_results: EvalResults | None = None
        extras = None
        if mode == "live":
            result, semantic_results, extras = await self._run_live(entries, subset)
        else:
            result = await self._run_heuristic(entries, subset)
        threshold_set = self.thresholds or GuardrailThresholds.for_mode(mode)
        self._apply_thresholds(result, threshold_set)
        if report_path is not None:
            result.report_path = str(
                write_guardrail_report(
                    result,
                    Path(report_path),
                    semantic_results=semantic_results,
                    extras=extras,
                )
            )
        logger.info(
            "EvalGuardrail {} | mode={} subset={} score={:.3f} queries={}",
            "PASSED" if result.passed else "FAILED",
            mode,
            subset,
            result.composite_score,
            result.total_queries,
        )
        return result

    async def _run_live(self, entries: list[GoldenEntry], subset: str) -> tuple[GuardrailResult, EvalResults, object]:
        runner = LiveBenchmarkRunner(
            pipeline=self._pipeline_adapter(),
            scorer=RAGASEvaluator(use_ragas=True),
        )
        scores, extras = await runner.run(entries, self.resolver)
        result = GuardrailResult(
            mode="live",
            subset=subset,
            total_queries=len(entries),
            faithfulness=scores.faithfulness,
            answer_relevancy=scores.answer_relevancy,
            context_precision=scores.context_precision,
            context_recall=scores.context_recall,
            hallucination_rate=round(max(0.0, 1.0 - scores.faithfulness), 4),
            composite_score=round(scores.overall_score, 4),
            citation_coverage=extras.citation_coverage,
            freshness_compliance=extras.freshness_compliance,
            failures=list(extras.failures),
            per_category=extras.per_category,
        )
        return result, scores, extras

    async def _run_heuristic(self, entries: list[GoldenEntry], subset: str) -> GuardrailResult:
        evaluator = RAGEvaluator(llm=None)
        cited = 0
        live_checks = 0
        fresh_checks = 0
        scores_by_category: dict[str, list[float]] = {}
        totals = {"faithfulness": 0.0, "relevancy": 0.0, "precision": 0.0, "hallucination": 0.0, "composite": 0.0}
        for entry in entries:
            reference = await self.resolver.resolve(entry)
            debug_result = await self._pipeline_adapter().answer(entry.query)
            metrics = await evaluator.evaluate(entry.query, debug_result.answer, debug_result.contexts, reference.ground_truth)
            totals["faithfulness"] += metrics.faithfulness
            totals["relevancy"] += metrics.answer_relevancy
            totals["precision"] += metrics.context_precision
            totals["hallucination"] += metrics.hallucination_rate
            totals["composite"] += metrics.composite_score
            scores_by_category.setdefault(entry.category, []).append(metrics.composite_score)
            cited += int(bool(debug_result.citations))
            if entry.mode == "live":
                live_checks += 1
                if reference.freshness_ok and debug_result.source_details and all(
                    detail.is_fresh is True for detail in debug_result.source_details
                ):
                    fresh_checks += 1
        count = max(len(entries), 1)
        return GuardrailResult(
            mode="heuristic",
            subset=subset,
            total_queries=len(entries),
            faithfulness=round(totals["faithfulness"] / count, 4),
            answer_relevancy=round(totals["relevancy"] / count, 4),
            context_precision=round(totals["precision"] / count, 4),
            context_recall=None,
            hallucination_rate=round(totals["hallucination"] / count, 4),
            composite_score=round(totals["composite"] / count, 4),
            citation_coverage=round(cited / count, 4),
            freshness_compliance=round(fresh_checks / max(live_checks, 1), 4) if live_checks else 1.0,
            per_category={key: round(sum(values) / len(values), 4) for key, values in scores_by_category.items()},
        )

    def _pipeline_adapter(self) -> KnowledgeAgentPipelineAdapter:
        if self._pipeline is None:
            self._pipeline = KnowledgeAgentPipelineAdapter(self.agent or build_benchmark_agent())
        return self._pipeline

    @staticmethod
    def _apply_thresholds(result: GuardrailResult, thresholds: GuardrailThresholds) -> None:
        checks = [
            (result.faithfulness >= thresholds.min_faithfulness, f"faithfulness {result.faithfulness:.3f} < {thresholds.min_faithfulness:.3f}"),
            (result.answer_relevancy >= thresholds.min_answer_relevancy, f"answer_relevancy {result.answer_relevancy:.3f} < {thresholds.min_answer_relevancy:.3f}"),
            (result.context_precision >= thresholds.min_context_precision, f"context_precision {result.context_precision:.3f} < {thresholds.min_context_precision:.3f}"),
            (result.hallucination_rate <= thresholds.max_hallucination_rate, f"hallucination_rate {result.hallucination_rate:.3f} > {thresholds.max_hallucination_rate:.3f}"),
            (result.composite_score >= thresholds.min_composite_score, f"composite_score {result.composite_score:.3f} < {thresholds.min_composite_score:.3f}"),
            (result.citation_coverage >= thresholds.min_citation_coverage, f"citation_coverage {result.citation_coverage:.3f} < {thresholds.min_citation_coverage:.3f}"),
            (result.freshness_compliance >= thresholds.min_freshness_compliance, f"freshness_compliance {result.freshness_compliance:.3f} < {thresholds.min_freshness_compliance:.3f}"),
        ]
        if result.context_recall is not None and thresholds.min_context_recall > 0:
            checks.append(
                (result.context_recall >= thresholds.min_context_recall, f"context_recall {result.context_recall:.3f} < {thresholds.min_context_recall:.3f}")
            )
        for passed, message in checks:
            if not passed and message not in result.failures:
                result.failures.append(message)
        result.passed = not result.failures
