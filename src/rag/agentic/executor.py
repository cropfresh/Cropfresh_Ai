"""Execution helpers for the agentic retrieval planner."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from src.rag.agentic.models import RetrievalPlan, ToolCall


async def execute_retrieval_plan(
    plan: RetrievalPlan,
    call_tool: Callable[[ToolCall], Awaitable[list[Any]]],
) -> list[Any]:
    """Execute a retrieval plan while respecting parallelization hints."""
    all_docs: list[Any] = []

    parallel_calls = [step for step in plan.plan if step.can_parallelize]
    sequential_calls = sorted(
        [step for step in plan.plan if not step.can_parallelize],
        key=lambda step: step.priority,
    )

    if parallel_calls:
        results = await asyncio.gather(
            *[call_tool(step) for step in parallel_calls],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, list):
                all_docs.extend(result)
            elif isinstance(result, Exception):
                logger.warning("Parallel tool failed: {}", result)

    for step in sequential_calls:
        try:
            result = await call_tool(step)
            if isinstance(result, list):
                all_docs.extend(result)
        except Exception as exc:
            logger.warning("Sequential tool {} failed: {}", step.tool_name, exc)

    logger.info("Agentic executor collected {} documents", len(all_docs))
    return all_docs
