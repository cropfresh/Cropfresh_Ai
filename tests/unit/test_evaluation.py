from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai.rag.evaluation.dataset_loader import BenchmarkDatasetLoader
from ai.rag.evaluation.golden_dataset import get_golden_dataset
from ai.rag.evaluation.guardrail import EvalGuardrail
from ai.rag.evaluation.models import LiveRunExtras
from ai.rag.evaluation.runtime import build_benchmark_agent
from src.agents.knowledge_models import BenchmarkDebugResult, BenchmarkSourceDetail
from src.evaluation.models import EvalResults, PerQuestionScore


class FakeAgent:
    async def answer_with_debug(self, question: str, context: str = "") -> BenchmarkDebugResult:
        del context
        return BenchmarkDebugResult(
            answer="Ragi thrives in red soil. [1]",
            contexts=["Ragi thrives in red soil in Karnataka."],
            citations=["[1]"],
            route="vector_only",
            source_details=[
                BenchmarkSourceDetail(source="kb", title="Ragi Guide", is_fresh=True),
            ],
        )


def _write_dataset(tmp_path: Path, entries: list[dict]) -> Path:
    dataset_path = tmp_path / "core_live.json"
    dataset_path.write_text(json.dumps(entries), encoding="utf-8")
    return dataset_path


def test_json_backed_dataset_contains_live_and_static_entries():
    dataset = get_golden_dataset("core_live")

    assert any(entry.mode == "live" for entry in dataset)
    assert any(entry.mode == "static" for entry in dataset)
    assert {entry.category for entry in dataset} >= {"market", "agronomy", "pest", "scheme", "kannada"}


@pytest.mark.asyncio
async def test_live_guardrail_uses_semantic_runner(monkeypatch, tmp_path):
    _write_dataset(
        tmp_path,
        [{"id": "m1", "query": "What is tomato price today?", "category": "market", "mode": "live"}],
    )

    async def fake_run(self, entries, resolver):
        del self, entries, resolver
        return (
            EvalResults(
                faithfulness=0.95,
                answer_relevancy=0.92,
                context_precision=0.89,
                context_recall=0.93,
                per_question=[PerQuestionScore(question="q", answer="a", faithfulness=0.95, answer_relevancy=0.92, context_precision=0.89, context_recall=0.93)],
                dataset_path="core_live",
            ),
            LiveRunExtras(citation_coverage=1.0, freshness_compliance=1.0, per_category={"market": 0.92}),
        )

    monkeypatch.setattr("ai.rag.evaluation.guardrail.LiveBenchmarkRunner.run", fake_run)
    guardrail = EvalGuardrail(
        dataset_loader=BenchmarkDatasetLoader(datasets_dir=tmp_path),
        agent=FakeAgent(),
    )

    result = await guardrail.run(mode="live", subset="core_live", report_path=tmp_path / "report.md")

    assert result.passed is True
    assert result.composite_score >= 0.90
    assert Path(result.report_path).exists()


@pytest.mark.asyncio
async def test_heuristic_guardrail_scores_pipeline_answers(tmp_path):
    _write_dataset(
        tmp_path,
        [
            {
                "id": "a1",
                "query": "How to grow ragi?",
                "category": "agronomy",
                "mode": "static",
                "ground_truth": "Ragi thrives in red soil.",
                "contexts": ["Ragi thrives in red soil in Karnataka."],
            }
        ],
    )
    guardrail = EvalGuardrail(
        dataset_loader=BenchmarkDatasetLoader(datasets_dir=tmp_path),
        agent=FakeAgent(),
    )

    result = await guardrail.run(mode="heuristic", subset="core_live")

    assert result.total_queries == 1
    assert result.composite_score > 0.30
    assert result.citation_coverage == 1.0


def test_build_benchmark_agent_disables_bedrock_by_default(monkeypatch):
    settings = SimpleNamespace(
        has_llm_configured=True,
        llm_provider="bedrock",
        llm_model="claude-sonnet-4",
        groq_api_key="",
        together_api_key="",
        vllm_base_url="http://localhost:8000/v1",
        aws_region="ap-south-1",
        aws_profile="",
        qdrant_host="localhost",
        qdrant_port=6333,
        qdrant_api_key="",
    )

    monkeypatch.setattr("ai.rag.evaluation.runtime._load_benchmark_settings", lambda: settings)
    monkeypatch.delenv("RAG_BENCHMARK_DISABLE_LLM", raising=False)
    monkeypatch.delenv("RAG_BENCHMARK_ALLOW_BEDROCK", raising=False)
    called = {"value": False}

    def fake_create_llm_provider(**kwargs):
        called["value"] = True
        return object()

    monkeypatch.setattr("ai.rag.evaluation.runtime.create_llm_provider", fake_create_llm_provider)

    agent = build_benchmark_agent()

    assert agent.llm is None
    assert called["value"] is False


def test_build_benchmark_agent_allows_bedrock_when_overridden(monkeypatch):
    settings = SimpleNamespace(
        has_llm_configured=True,
        llm_provider="bedrock",
        llm_model="claude-sonnet-4",
        groq_api_key="",
        together_api_key="",
        vllm_base_url="http://localhost:8000/v1",
        aws_region="ap-south-1",
        aws_profile="",
        qdrant_host="localhost",
        qdrant_port=6333,
        qdrant_api_key="",
    )

    monkeypatch.setattr("ai.rag.evaluation.runtime._load_benchmark_settings", lambda: settings)
    monkeypatch.setenv("RAG_BENCHMARK_ALLOW_BEDROCK", "true")
    created = object()

    monkeypatch.setattr(
        "ai.rag.evaluation.runtime.create_llm_provider",
        lambda **kwargs: created,
    )

    agent = build_benchmark_agent()

    assert agent.llm is created
