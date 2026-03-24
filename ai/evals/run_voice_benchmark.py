"""Compatibility entrypoint for the Sprint 09 voice benchmark runner."""

from __future__ import annotations

import asyncio

from src.evaluation.voice_benchmark_runner import _main

if __name__ == "__main__":
    asyncio.run(_main())
