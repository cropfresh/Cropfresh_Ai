"""CLI to export a canonical quality manifest into training-ready layouts."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.agents.quality_assessment.training.dataset_exporter import (
    export_quality_dataset_from_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CropFresh quality dataset layouts")
    parser.add_argument("--manifest", required=True, type=Path, help="Path to canonical JSON or JSONL manifest")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write exported dataset artifacts")
    args = parser.parse_args()
    summary = export_quality_dataset_from_manifest(args.manifest, args.output_dir)
    print(summary.to_dict())


if __name__ == "__main__":
    main()
