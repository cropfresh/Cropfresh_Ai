"""Compatibility entrypoint for the CI evaluation workflow."""

from __future__ import annotations

import asyncio

from src.evaluation.eval_runner import _main

if __name__ == "__main__":
    asyncio.run(_main())
