"""Focused tests for the Sprint 09 voice benchmark runner."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest

from src.evaluation.voice_benchmark_runner import DEFAULT_DATASET_PATH, VoiceBenchmarkRunner

ROOT = Path(__file__).resolve().parents[2]


def test_voice_benchmark_dataset_covers_all_sprint_09_languages() -> None:
    """The fixed utterance set should stay locked to the four Sprint 09 languages."""
    runner = VoiceBenchmarkRunner(dataset_path=DEFAULT_DATASET_PATH)

    entries = runner.load_dataset()

    assert {entry.language for entry in entries} == {"kn", "hi", "te", "ta"}
    assert len(entries) >= 8


@pytest.mark.asyncio
async def test_voice_benchmark_runner_writes_artifacts_and_merges_observations() -> None:
    """The runner should emit JSON and markdown artifacts with merged timing observations."""
    tmp_path = ROOT / ".pytest-artifacts" / f"voice-benchmark-{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        observations_path = tmp_path / "observations.json"
        observations_path.write_text(
            json.dumps(
                [
                    {
                        "benchmark_id": "hi_code_mix_flush_01",
                        "first_audio_ms": 980,
                        "bargein_reaction_ms": 120,
                        "interruption_recovery_ms": 240,
                        "notes": "Observed on fallback websocket path.",
                    }
                ]
            ),
            encoding="utf-8",
        )

        runner = VoiceBenchmarkRunner(dataset_path=DEFAULT_DATASET_PATH)
        report = await runner.run(tmp_path, observations_path=observations_path)

        json_report = json.loads((tmp_path / "voice_benchmark_report.json").read_text(encoding="utf-8"))
        markdown_report = (tmp_path / "voice_benchmark_report.md").read_text(encoding="utf-8")
        observed_artifact = next(
            artifact for artifact in report.artifacts if artifact.benchmark_id == "hi_code_mix_flush_01"
        )

        assert report.total_cases >= 8
        assert json_report["matched_cases"] <= json_report["total_cases"]
        assert "voice_benchmark_report.md" not in markdown_report
        assert "Voice Benchmark Report" in markdown_report
        assert observed_artifact.first_audio_ms == 980
        assert observed_artifact.bargein_reaction_ms == 120
        assert observed_artifact.timing_targets_met is True
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
