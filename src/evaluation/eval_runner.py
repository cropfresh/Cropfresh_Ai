"""
Batch Evaluation Runner — CLI entry-point for running the full evaluation.

Responsibility: orchestrate DatasetLoader → RAGASEvaluator → ReportGenerator.

Usage (module):
    python -m src.evaluation.eval_runner \\
        --dataset datasets/agronomy_qa.json \\
        --output reports/eval_report.md

Usage (code):
    runner = EvalRunner()
    asyncio.run(runner.run_all(output_dir=Path("reports")))
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from loguru import logger

from src.evaluation.dataset_loader import DatasetLoader, _DOMAIN_FILES
from src.evaluation.models import EvalResults
from src.evaluation.ragas_evaluator import RAGASEvaluator, create_ragas_evaluator
from src.evaluation.report_generator import ReportGenerator


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class EvalRunner:
    """
    High-level coordinator: evaluates all domain datasets and produces
    individual per-domain reports plus a combined full-suite report.
    """

    def __init__(
        self,
        evaluator: RAGASEvaluator | None = None,
        loader: DatasetLoader | None = None,
        generator: ReportGenerator | None = None,
    ) -> None:
        self.evaluator = evaluator or create_ragas_evaluator()
        self.loader = loader or DatasetLoader()
        self.generator = generator or ReportGenerator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_dataset(
        self,
        dataset_path: str | Path,
        output_path: Path | str = Path("eval_report.md"),
    ) -> EvalResults:
        """Evaluate a single dataset file and write the report."""
        logger.info(f"Evaluating dataset: {dataset_path}")
        results = await self.evaluator.run_evaluation(dataset_path)
        self.generator.generate(results, output_path)
        self._log_summary(results)
        return results

    async def run_all(self, output_dir: Path = Path("reports")) -> dict[str, EvalResults]:
        """
        Evaluate every domain dataset and the combined golden dataset.
        Returns a mapping of {filename_stem: EvalResults}.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results_map: dict[str, EvalResults] = {}

        for fname in _DOMAIN_FILES:
            stem = Path(fname).stem
            try:
                results = await self.run_dataset(
                    fname, output_dir / f"{stem}_report.md"
                )
                results_map[stem] = results
            except FileNotFoundError:
                logger.warning(f"Skipping missing dataset: {fname}")

        # Combined evaluation on the full golden dataset
        golden_path = Path(__file__).parent / "golden_dataset.json"
        if golden_path.exists():
            results = await self.run_dataset(
                golden_path, output_dir / "golden_report.md"
            )
            results_map["golden"] = results

        return results_map

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_summary(results: EvalResults) -> None:
        targets = results.meets_targets()
        passed = sum(targets.values())
        total = len(targets)
        logger.info(
            f"Evaluation complete | overall={results.overall_score:.3f} "
            f"targets_met={passed}/{total}"
        )
        for metric, ok in targets.items():
            score = getattr(results, metric)
            indicator = "✅" if ok else "❌"
            logger.info(f"  {indicator} {metric}: {score:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CropFresh RAGAS Evaluation Runner")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to a single dataset JSON (default: run all domain datasets)",
    )
    parser.add_argument(
        "--output",
        default="reports/eval_report.md",
        help="Output path for the report (default: reports/eval_report.md)",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Output directory when running all datasets (default: reports)",
    )
    return parser


async def _main() -> None:
    args = _build_parser().parse_args()
    runner = EvalRunner()

    if args.dataset:
        await runner.run_dataset(args.dataset, Path(args.output))
    else:
        await runner.run_all(Path(args.output_dir))


if __name__ == "__main__":
    asyncio.run(_main())
