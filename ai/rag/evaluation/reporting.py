from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from src.evaluation.report_generator import ReportGenerator

if TYPE_CHECKING:
    from ai.rag.evaluation.guardrail import GuardrailResult
    from ai.rag.evaluation.models import LiveRunExtras
    from src.evaluation.models import EvalResults


def write_guardrail_report(
    result: "GuardrailResult",
    output_path: Path,
    semantic_results: "EvalResults | None" = None,
    extras: "LiveRunExtras | None" = None,
) -> Path:
    """Persist a compact markdown and JSON summary for a benchmark run."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path = output_path.with_suffix(".json")
    if semantic_results is not None:
        ReportGenerator().generate(semantic_results, output_path)
        output_path.write_text(
            output_path.read_text(encoding="utf-8") + "\n" + _build_markdown(result),
            encoding="utf-8",
        )
        summary_json_path = output_path.with_name(f"{output_path.stem}_guardrail.json")
    else:
        output_path.write_text(_build_markdown(result), encoding="utf-8")
    payload = result.model_dump()
    if extras is not None:
        payload["extras"] = extras.model_dump()
        output_path.with_name(f"{output_path.stem}_extras.json").write_text(
            json.dumps(extras.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    summary_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def _build_markdown(result: "GuardrailResult") -> str:
    lines = [
        "# CropFresh RAG Guardrail Report",
        "",
        f"- Mode: `{result.mode}`",
        f"- Subset: `{result.subset}`",
        f"- Passed: `{result.passed}`",
        f"- Queries: `{result.total_queries}`",
        "",
        "## Metrics",
        "",
        f"- Faithfulness: `{result.faithfulness:.4f}`",
        f"- Answer relevancy: `{result.answer_relevancy:.4f}`",
        f"- Context precision: `{result.context_precision:.4f}`",
        f"- Context recall: `{result.context_recall:.4f}`" if result.context_recall is not None else "- Context recall: `n/a`",
        f"- Hallucination rate: `{result.hallucination_rate:.4f}`",
        f"- Composite score: `{result.composite_score:.4f}`",
        f"- Citation coverage: `{result.citation_coverage:.4f}`",
        f"- Freshness compliance: `{result.freshness_compliance:.4f}`",
        "",
        "## Per Category",
        "",
    ]
    for category, score in sorted(result.per_category.items()):
        lines.append(f"- {category}: `{score:.4f}`")
    if result.failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in result.failures)
    return "\n".join(lines) + "\n"
