from __future__ import annotations

from scripts.eval_guardrail import (
    _report_path,
    is_ascii_safe,
    render_result_block,
    summarize_results,
)
from src.rag.benchmark.guardrail import GuardrailResult


def test_rendered_cli_block_is_ascii_safe():
    result = GuardrailResult(
        mode="live",
        subset="core_live",
        passed=True,
        total_queries=10,
        faithfulness=0.94,
        answer_relevancy=0.91,
        context_precision=0.88,
        context_recall=0.9,
        hallucination_rate=0.06,
        composite_score=0.91,
        citation_coverage=1.0,
        freshness_compliance=1.0,
    )

    block = render_result_block(result, 1)

    assert is_ascii_safe(block) is True
    assert "status=PASS" in block


def test_summarize_results_uses_median_values():
    run_one = GuardrailResult(mode="live", subset="core_live", total_queries=5, faithfulness=0.91, answer_relevancy=0.90, context_precision=0.85, context_recall=0.90, hallucination_rate=0.09, composite_score=0.89, citation_coverage=1.0, freshness_compliance=1.0)
    run_two = GuardrailResult(mode="live", subset="core_live", total_queries=5, faithfulness=0.94, answer_relevancy=0.92, context_precision=0.88, context_recall=0.93, hallucination_rate=0.06, composite_score=0.92, citation_coverage=1.0, freshness_compliance=1.0)
    run_three = GuardrailResult(mode="live", subset="core_live", total_queries=5, faithfulness=0.95, answer_relevancy=0.93, context_precision=0.89, context_recall=0.94, hallucination_rate=0.05, composite_score=0.93, citation_coverage=1.0, freshness_compliance=1.0)

    summary = summarize_results([run_one, run_two, run_three])

    assert summary.faithfulness == 0.94
    assert summary.composite_score == 0.92
    assert summary.passed is True


def test_report_path_adds_run_suffix_for_multi_run_output():
    report_path = _report_path("reports/rag/latest", "live", "core_live", 2, 3)

    assert str(report_path).endswith("latest_run2.md")
