from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Iterable

from src.rag.benchmark.guardrail import EvalGuardrail, GuardrailResult, GuardrailThresholds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CropFresh RAG guardrail")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="Run semantic live benchmark mode")
    mode.add_argument("--heuristic", action="store_true", help="Run heuristic smoke mode")
    parser.add_argument("--subset", default="core_live", choices=["core_live", "full"], help="Benchmark dataset subset")
    parser.add_argument("--runs", type=int, default=1, help="Number of repeated runs to execute")
    parser.add_argument("--report", type=str, help="Optional report path for markdown and JSON output")
    return parser


def summarize_results(results: list[GuardrailResult]) -> GuardrailResult:
    mode = results[0].mode
    subset = results[0].subset
    summary = GuardrailResult(
        mode=mode,
        subset=subset,
        total_queries=results[0].total_queries,
        faithfulness=_median_metric(results, "faithfulness"),
        answer_relevancy=_median_metric(results, "answer_relevancy"),
        context_precision=_median_metric(results, "context_precision"),
        context_recall=_median_optional_metric(results, "context_recall"),
        hallucination_rate=_median_metric(results, "hallucination_rate"),
        composite_score=_median_metric(results, "composite_score"),
        citation_coverage=_median_metric(results, "citation_coverage"),
        freshness_compliance=_median_metric(results, "freshness_compliance"),
        per_category=_median_categories(results),
    )
    EvalGuardrail._apply_thresholds(summary, GuardrailThresholds.for_mode(mode))
    return summary


def render_result_block(result: GuardrailResult, run_index: int | None = None) -> str:
    header = f"RUN {run_index}" if run_index is not None else "SUMMARY"
    recall = f"{result.context_recall:.4f}" if result.context_recall is not None else "n/a"
    lines = [
        "=" * 60,
        f"{header} | mode={result.mode} subset={result.subset} status={'PASS' if result.passed else 'FAIL'}",
        "-" * 60,
        f"queries={result.total_queries} composite={result.composite_score:.4f} faithfulness={result.faithfulness:.4f}",
        f"relevancy={result.answer_relevancy:.4f} precision={result.context_precision:.4f} recall={recall}",
        f"hallucination={result.hallucination_rate:.4f} citations={result.citation_coverage:.4f} freshness={result.freshness_compliance:.4f}",
    ]
    if result.report_path:
        lines.append(f"report={result.report_path}")
    if result.failures:
        lines.extend([f"failure={failure}" for failure in result.failures])
    return "\n".join(lines)


def is_ascii_safe(text: str) -> bool:
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


async def main(args: argparse.Namespace) -> int:
    mode = "heuristic" if args.heuristic else "live"
    guardrail = EvalGuardrail(thresholds=GuardrailThresholds.for_mode(mode))
    results: list[GuardrailResult] = []
    for run_index in range(1, max(args.runs, 1) + 1):
        report_path = _report_path(args.report, mode, args.subset, run_index, args.runs)
        result = await guardrail.run(mode=mode, subset=args.subset, report_path=report_path)
        results.append(result)
        block = render_result_block(result, run_index)
        if not is_ascii_safe(block):
            raise ValueError("Console output must remain ASCII-safe.")
        print(block)
    summary = summarize_results(results)
    summary_block = render_result_block(summary)
    if not is_ascii_safe(summary_block):
        raise ValueError("Summary output must remain ASCII-safe.")
    print(summary_block)
    print("=" * 60)
    return 0 if summary.passed else 1


def _median_metric(results: Iterable[GuardrailResult], field: str) -> float:
    return round(median(getattr(result, field) for result in results), 4)


def _median_optional_metric(results: Iterable[GuardrailResult], field: str) -> float | None:
    values = [getattr(result, field) for result in results if getattr(result, field) is not None]
    return round(median(values), 4) if values else None


def _median_categories(results: list[GuardrailResult]) -> dict[str, float]:
    categories = {key for result in results for key in result.per_category}
    return {
        category: round(median(result.per_category.get(category, 0.0) for result in results), 4)
        for category in sorted(categories)
    }


def _report_path(base_path: str | None, mode: str, subset: str, run_index: int, runs: int) -> Path:
    if base_path:
        candidate = Path(base_path)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = Path("reports/rag") / f"{mode}_{subset}_{timestamp}.md"
    if candidate.suffix != ".md":
        candidate = candidate.with_suffix(".md")
    if runs == 1:
        return candidate
    return candidate.with_name(f"{candidate.stem}_run{run_index}{candidate.suffix}")


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    raise SystemExit(asyncio.run(main(parsed)))
