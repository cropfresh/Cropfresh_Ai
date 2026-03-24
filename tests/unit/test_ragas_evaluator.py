"""
Unit tests for the RAGAS Evaluation Framework (Task 18).

Tests cover:
- GoldenItem / EvalResults model validation
- DatasetLoader load/save round-trip
- Heuristic scoring functions
- RAGASEvaluator stub-mode evaluation (no real ragas dep required)
- ReportGenerator markdown and JSON output
- EvalRunner.run_dataset orchestration
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.evaluation.dataset_loader import DatasetLoader
from src.evaluation.eval_runner import EvalRunner
from src.evaluation.models import EvalResults, GoldenItem, PerQuestionScore
from src.evaluation.ragas_evaluator import (
    RAGASEvaluator,
    RAGResponse,
    _heuristic_answer_relevancy,
    _heuristic_context_precision,
    _heuristic_context_recall,
    _heuristic_faithfulness,
    create_ragas_evaluator,
)
from src.evaluation.report_generator import ReportGenerator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_item() -> GoldenItem:
    return GoldenItem(
        question="What is the price of tomato in Bangalore?",
        ground_truth="Tomato price in Bangalore is ₹25–30/kg.",
        contexts=["Tomato prices in Bangalore APMC: ₹28/kg modal price today."],
        agent_domain="commerce",
        difficulty="easy",
        language="en",
    )


@pytest.fixture
def sample_items(sample_item: GoldenItem) -> list[GoldenItem]:
    return [
        sample_item,
        GoldenItem(
            question="How to prevent blight?",
            ground_truth="Apply copper fungicide every 10–14 days.",
            contexts=["Copper-based fungicides are recommended for blight management."],
            agent_domain="agronomy",
            difficulty="medium",
            language="en",
        ),
    ]


@pytest.fixture
def temp_dataset(sample_items: list[GoldenItem], tmp_path: Path) -> Path:
    """Write sample items to a temp JSON file."""
    data = [item.model_dump() for item in sample_items]
    path = tmp_path / "test_qa.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def eval_results(sample_items: list[GoldenItem]) -> EvalResults:
    per_q = [
        PerQuestionScore(
            question=it.question,
            answer=it.ground_truth,
            faithfulness=0.82,
            answer_relevancy=0.77,
            context_precision=0.75,
            context_recall=0.72,
        )
        for it in sample_items
    ]
    return EvalResults(
        faithfulness=0.82,
        answer_relevancy=0.77,
        context_precision=0.75,
        context_recall=0.72,
        per_question=per_q,
        dataset_path="/tmp/test.json",
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestGoldenItem:
    def test_defaults(self, sample_item: GoldenItem) -> None:
        assert sample_item.agent_domain == "commerce"
        assert sample_item.language == "en"
        assert isinstance(sample_item.contexts, list)

    def test_serialisation_round_trip(self, sample_item: GoldenItem) -> None:
        dumped = sample_item.model_dump()
        restored = GoldenItem(**dumped)
        assert restored.question == sample_item.question
        assert restored.ground_truth == sample_item.ground_truth


class TestEvalResults:
    def test_overall_score(self, eval_results: EvalResults) -> None:
        expected = (0.82 + 0.77 + 0.75 + 0.72) / 4
        assert abs(eval_results.overall_score - expected) < 1e-6

    def test_meets_targets_all_pass(self, eval_results: EvalResults) -> None:
        targets = eval_results.meets_targets()
        assert targets["faithfulness"] is True
        assert targets["answer_relevancy"] is True
        assert targets["context_precision"] is True
        assert targets["context_recall"] is True

    def test_meets_targets_failure(self) -> None:
        results = EvalResults(
            faithfulness=0.70,   # below target 0.80
            answer_relevancy=0.80,
            context_precision=0.75,
            context_recall=0.75,
        )
        assert results.meets_targets()["faithfulness"] is False

    def test_worst_questions(self, eval_results: EvalResults) -> None:
        worst = eval_results.worst_questions(1)
        assert len(worst) == 1

    def test_per_question_average(self) -> None:
        pq = PerQuestionScore(
            question="Q?",
            answer="A",
            faithfulness=0.8,
            answer_relevancy=0.8,
            context_precision=0.8,
            context_recall=0.8,
        )
        assert abs(pq.average - 0.8) < 1e-6


# ---------------------------------------------------------------------------
# DatasetLoader tests
# ---------------------------------------------------------------------------


class TestDatasetLoader:
    def test_load_valid_file(self, temp_dataset: Path, tmp_path: Path) -> None:
        loader = DatasetLoader(datasets_dir=tmp_path)
        items = loader.load(temp_dataset)
        assert len(items) == 2
        assert all(isinstance(i, GoldenItem) for i in items)

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        loader = DatasetLoader(datasets_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent.json")

    def test_save_and_reload(self, sample_items: list[GoldenItem], tmp_path: Path) -> None:
        loader = DatasetLoader(datasets_dir=tmp_path)
        out = tmp_path / "saved.json"
        loader.save_golden(sample_items, out)
        reloaded = loader._read_and_validate(out)
        assert len(reloaded) == len(sample_items)
        assert reloaded[0].question == sample_items[0].question

    def test_skip_malformed_entries(self, tmp_path: Path) -> None:
        bad_data = [{"question": "Good Q", "ground_truth": "Good A"}, {"invalid": True}]
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad_data), encoding="utf-8")
        loader = DatasetLoader(datasets_dir=tmp_path)
        items = loader._read_and_validate(path)
        assert len(items) == 1   # malformed entry skipped


# ---------------------------------------------------------------------------
# Heuristic scoring function tests
# ---------------------------------------------------------------------------


class TestHeuristicFunctions:
    def test_faithfulness_perfect_match(self) -> None:
        score = _heuristic_faithfulness("tomato price", ["tomato price is ₹28"])
        assert score > 0.5

    def test_faithfulness_no_context(self) -> None:
        assert _heuristic_faithfulness("anything", []) == 0.0

    def test_faithfulness_empty_answer(self) -> None:
        assert _heuristic_faithfulness("", ["some context"]) == 1.0

    def test_answer_relevancy_full_overlap(self) -> None:
        score = _heuristic_answer_relevancy("tomato price today", "tomato price today")
        assert score >= 1.0

    def test_context_precision_none_relevant(self) -> None:
        score = _heuristic_context_precision("tomato price", ["unrelated text here"])
        assert isinstance(score, float)

    def test_context_recall_all_in_context(self) -> None:
        score = _heuristic_context_recall("tomato", ["I love tomato farming"])
        assert score == 1.0


# ---------------------------------------------------------------------------
# RAGASEvaluator tests
# ---------------------------------------------------------------------------


class TestRAGASEvaluator:
    def test_init_no_ragas(self) -> None:
        """Evaluator should initialise cleanly without ragas installed."""
        ev = create_ragas_evaluator(use_ragas=False)
        assert ev._ragas_available is False

    @pytest.mark.asyncio
    async def test_run_evaluation_stub_mode(
        self, temp_dataset: Path, tmp_path: Path
    ) -> None:
        """Evaluation with no pipeline → returns results using stub answers."""
        loader = DatasetLoader(datasets_dir=tmp_path)
        ev = RAGASEvaluator(rag_pipeline=None, loader=loader, use_ragas=False)
        results = await ev.run_evaluation(temp_dataset)

        assert isinstance(results, EvalResults)
        assert 0.0 <= results.faithfulness <= 1.0
        assert 0.0 <= results.answer_relevancy <= 1.0
        assert len(results.per_question) == 2

    @pytest.mark.asyncio
    async def test_run_evaluation_with_pipeline(
        self, temp_dataset: Path, tmp_path: Path, sample_item: GoldenItem
    ) -> None:
        """Evaluation with mock pipeline returns scores for pipeline output."""
        mock_pipeline = AsyncMock()
        mock_pipeline.query = AsyncMock(
            return_value=RAGResponse(
                answer="Tomato price is ₹28/kg in Bangalore.",
                contexts=["Tomato prices: ₹28/kg modal at Bangalore APMC."],
            )
        )
        loader = DatasetLoader(datasets_dir=tmp_path)
        ev = RAGASEvaluator(rag_pipeline=mock_pipeline, loader=loader, use_ragas=False)
        results = await ev.run_evaluation(temp_dataset)

        assert len(results.per_question) == 2
        assert mock_pipeline.query.call_count == 2

    @pytest.mark.asyncio
    async def test_pipeline_failure_falls_back_gracefully(
        self, temp_dataset: Path, tmp_path: Path
    ) -> None:
        """A broken pipeline should not crash the evaluator."""
        mock_pipeline = AsyncMock()
        mock_pipeline.query = AsyncMock(side_effect=RuntimeError("pipeline down"))
        loader = DatasetLoader(datasets_dir=tmp_path)
        ev = RAGASEvaluator(rag_pipeline=mock_pipeline, loader=loader, use_ragas=False)
        results = await ev.run_evaluation(temp_dataset)   # must not raise
        assert len(results.per_question) == 2

    @pytest.mark.asyncio
    async def test_empty_dataset_raises(self, tmp_path: Path) -> None:
        """Passing an empty dataset should raise ValueError."""
        empty = tmp_path / "empty.json"
        empty.write_text("[]", encoding="utf-8")
        loader = DatasetLoader(datasets_dir=tmp_path)
        ev = RAGASEvaluator(rag_pipeline=None, loader=loader, use_ragas=False)
        with pytest.raises(ValueError):
            await ev.run_evaluation(empty)


# ---------------------------------------------------------------------------
# ReportGenerator tests
# ---------------------------------------------------------------------------


class TestReportGenerator:
    def test_generate_creates_markdown(
        self, eval_results: EvalResults, tmp_path: Path
    ) -> None:
        gen = ReportGenerator()
        out = gen.generate(eval_results, tmp_path / "report.md")
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "# CropFresh RAG Evaluation Report" in content
        assert "Faithfulness" in content
        assert "Worst Performing Questions" in content

    def test_generate_creates_json_metadata(
        self, eval_results: EvalResults, tmp_path: Path
    ) -> None:
        gen = ReportGenerator()
        gen.generate(eval_results, tmp_path / "report.md")
        meta_path = tmp_path / "report.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert "faithfulness" in meta
        assert "all_targets_met" in meta
        assert meta["num_questions"] == len(eval_results.per_question)

    def test_target_status_displayed_correctly(
        self, eval_results: EvalResults, tmp_path: Path
    ) -> None:
        gen = ReportGenerator()
        out = gen.generate(eval_results, tmp_path / "report.md")
        content = out.read_text(encoding="utf-8")
        # All metrics above target → all ✅
        assert "✅" in content

    def test_report_creates_parent_dirs(
        self, eval_results: EvalResults, tmp_path: Path
    ) -> None:
        nested = tmp_path / "a" / "b" / "c" / "report.md"
        gen = ReportGenerator()
        gen.generate(eval_results, nested)
        assert nested.exists()


# ---------------------------------------------------------------------------
# EvalRunner tests
# ---------------------------------------------------------------------------


class TestEvalRunner:
    @pytest.mark.asyncio
    async def test_run_dataset_produces_report(
        self, temp_dataset: Path, tmp_path: Path
    ) -> None:
        loader = DatasetLoader(datasets_dir=tmp_path)
        evaluator = RAGASEvaluator(rag_pipeline=None, loader=loader, use_ragas=False)
        generator = ReportGenerator()
        runner = EvalRunner(evaluator=evaluator, loader=loader, generator=generator)

        report_path = tmp_path / "out.md"
        results = await runner.run_dataset(temp_dataset, report_path)

        assert isinstance(results, EvalResults)
        assert report_path.exists()
