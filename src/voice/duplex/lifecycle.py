"""
Duplex Turn Lifecycle
=====================
Helpers for finalizing duplex turns and updating conversation state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from .models import PipelineState
from .timing import TurnTiming

if TYPE_CHECKING:
    from .pipeline import DuplexPipeline


def update_history(
    pipeline: "DuplexPipeline",
    text: str,
    response_text: str,
) -> None:
    """Persist the latest user/assistant exchange in memory."""
    pipeline._conversation_history.append({"role": "user", "content": text})
    pipeline._conversation_history.append(
        {"role": "assistant", "content": response_text}
    )
    if len(pipeline._conversation_history) > 20:
        pipeline._conversation_history = pipeline._conversation_history[-20:]


async def emit_interrupted(
    pipeline: "DuplexPipeline",
    timing: TurnTiming,
) -> None:
    """Emit a single interrupted event for the current turn."""
    if timing.interrupted_at is not None:
        return

    timing.mark_interrupted()
    await pipeline._emit(PipelineState.INTERRUPTED, timing=timing.snapshot())


async def finish_turn(
    pipeline: "DuplexPipeline",
    timing: TurnTiming,
    *,
    interrupted: bool = False,
    chunk_count: int = 0,
    error: str | None = None,
) -> None:
    """Finalize timings and emit the final idle event."""
    if interrupted:
        await emit_interrupted(pipeline, timing)

    timing.mark_completed()
    snapshot = timing.snapshot()
    pipeline._last_turn_timing = snapshot

    logger.info(
        "[DuplexPipeline] Complete: {} audio chunks, {}ms total",
        chunk_count,
        round(snapshot.get("total_ms") or 0),
    )

    payload = {"latency_ms": snapshot.get("total_ms"), "timing": snapshot}
    if error:
        payload["error"] = error
    await pipeline._emit(PipelineState.IDLE, **payload)
