"""Runner for the Sprint 09 multilingual voice benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from loguru import logger

from src.evaluation.voice_benchmark_models import (
    VoiceBenchmarkArtifact,
    VoiceBenchmarkEntry,
    VoiceBenchmarkObservation,
    VoiceBenchmarkReport,
)
from src.shared.voice_semantic import evaluate_semantic_flush

DEFAULT_DATASET_PATH = Path(__file__).parent / "datasets" / "voice_multilingual_benchmark.json"
DEFAULT_RUBRIC_PATH = Path("docs/features/voice-benchmarking.md")


class VoiceBenchmarkRunner:
    """Evaluate fixed voice utterances against the semantic endpointing contract."""

    def __init__(
        self,
        dataset_path: Path = DEFAULT_DATASET_PATH,
        *,
        timeout_ms: int = 150,
        max_hold_ms: int = 800,
    ) -> None:
        self.dataset_path = dataset_path
        self.timeout_ms = timeout_ms
        self.max_hold_ms = max_hold_ms

    def load_dataset(self) -> list[VoiceBenchmarkEntry]:
        """Load and validate the fixed multilingual utterance set."""
        raw_entries = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        return [VoiceBenchmarkEntry.model_validate(entry) for entry in raw_entries]

    def load_observations(self, path: Path | None) -> dict[str, VoiceBenchmarkObservation]:
        """Load optional runtime observations keyed by benchmark id."""
        if path is None or not path.exists():
            return {}

        raw_entries = json.loads(path.read_text(encoding="utf-8"))
        observations = [
            VoiceBenchmarkObservation.model_validate(entry) for entry in raw_entries
        ]
        return {entry.benchmark_id: entry for entry in observations}

    async def run(
        self,
        output_dir: Path,
        *,
        observations_path: Path | None = None,
    ) -> VoiceBenchmarkReport:
        """Evaluate the benchmark set and write markdown + JSON artifacts."""
        entries = self.load_dataset()
        observations = self.load_observations(observations_path)
        artifacts = [await self._evaluate_entry(entry, observations.get(entry.id)) for entry in entries]
        report = VoiceBenchmarkReport(
            dataset_path=str(self.dataset_path),
            rubric_path=str(DEFAULT_RUBRIC_PATH),
            total_cases=len(artifacts),
            matched_cases=sum(1 for artifact in artifacts if artifact.matched),
            languages=sorted({artifact.language for artifact in artifacts}),
            artifacts=artifacts,
        )
        self._write_outputs(report, output_dir)
        logger.info(
            "Voice benchmark complete | matched={}/{}",
            report.matched_cases,
            report.total_cases,
        )
        return report

    async def _evaluate_entry(
        self,
        entry: VoiceBenchmarkEntry,
        observation: VoiceBenchmarkObservation | None,
    ) -> VoiceBenchmarkArtifact:
        decision = await evaluate_semantic_flush(
            transcript=entry.prompt_text,
            language=entry.language,
            llm_provider=None,
            enabled=True,
            timeout_ms=self.timeout_ms,
            max_hold_ms=self.max_hold_ms,
        )
        actual_endpointing = "flush" if decision.should_flush else "hold_then_flush"
        return VoiceBenchmarkArtifact(
            benchmark_id=entry.id,
            language=entry.language,
            prompt_text=entry.prompt_text,
            expected_endpointing=entry.expected_endpointing,
            actual_endpointing=actual_endpointing,
            matched=entry.expected_endpointing == actual_endpointing,
            decision_reason=decision.reason,
            semantic_hold_ms=decision.semantic_hold_ms,
            first_audio_ms=observation.first_audio_ms if observation else None,
            bargein_reaction_ms=observation.bargein_reaction_ms if observation else None,
            interruption_recovery_ms=(
                observation.interruption_recovery_ms if observation else None
            ),
            timing_targets_met=self._timing_targets_met(observation),
            expected_notes=entry.expected_notes,
            observation_notes=observation.notes if observation else None,
        )

    def _write_outputs(self, report: VoiceBenchmarkReport, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "voice_benchmark_report.json"
        markdown_path = output_dir / "voice_benchmark_report.md"
        json_path.write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        markdown_path.write_text(self._build_markdown(report), encoding="utf-8")

    def _build_markdown(self, report: VoiceBenchmarkReport) -> str:
        lines = [
            "# Voice Benchmark Report",
            "",
            f"- Generated at: `{report.generated_at.isoformat()}`",
            f"- Dataset: `{report.dataset_path}`",
            f"- Rubric: `{report.rubric_path}`",
            f"- Endpointing matches: `{report.matched_cases}/{report.total_cases}`",
            "",
            "| Benchmark | Lang | Expected | Actual | Match | First audio | Barge-in | Reason |",
            "|-----------|------|----------|--------|-------|-------------|----------|--------|",
        ]
        for artifact in report.artifacts:
            lines.append(
                "| {id} | {lang} | {expected} | {actual} | {matched} | {first_audio} | {bargein} | {reason} |".format(
                    id=artifact.benchmark_id,
                    lang=artifact.language,
                    expected=artifact.expected_endpointing,
                    actual=artifact.actual_endpointing,
                    matched="yes" if artifact.matched else "no",
                    first_audio=artifact.first_audio_ms if artifact.first_audio_ms is not None else "-",
                    bargein=artifact.bargein_reaction_ms if artifact.bargein_reaction_ms is not None else "-",
                    reason=artifact.decision_reason,
                )
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _timing_targets_met(
        observation: VoiceBenchmarkObservation | None,
    ) -> bool | None:
        if observation is None:
            return None

        checks: list[bool] = []
        if observation.first_audio_ms is not None:
            checks.append(observation.first_audio_ms <= 1200)
        if observation.bargein_reaction_ms is not None:
            checks.append(observation.bargein_reaction_ms <= 150)
        return all(checks) if checks else None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CropFresh voice benchmark runner")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output-dir", default="reports/voice")
    parser.add_argument("--observations", default=None)
    return parser


async def _main() -> None:
    args = _build_parser().parse_args()
    runner = VoiceBenchmarkRunner(dataset_path=Path(args.dataset))
    await runner.run(
        Path(args.output_dir),
        observations_path=Path(args.observations) if args.observations else None,
    )


if __name__ == "__main__":
    asyncio.run(_main())
