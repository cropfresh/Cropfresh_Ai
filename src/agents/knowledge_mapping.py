from __future__ import annotations

import re
import time
from typing import Any

from src.agents.knowledge_models import BenchmarkSourceDetail


def build_source_details(documents: list[Any]) -> list[BenchmarkSourceDetail]:
    """Extract source freshness details from graph documents."""
    details: list[BenchmarkSourceDetail] = []
    now = time.time()
    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        source = getattr(document, "source", "") or str(metadata.get("source", ""))
        timestamp = metadata.get("as_of", "")
        raw_timestamp = metadata.get("timestamp")
        is_fresh = None
        if isinstance(raw_timestamp, (int, float)):
            is_fresh = (now - raw_timestamp) <= 7 * 24 * 3600
        details.append(
            BenchmarkSourceDetail(
                source=source or "unknown",
                title=str(metadata.get("title", metadata.get("market", source or "unknown"))),
                timestamp=str(timestamp),
                is_fresh=is_fresh,
                metadata=metadata,
            )
        )
    return details


def extract_citations(answer: str) -> list[str]:
    """Extract inline citation markers from the generated answer."""
    return re.findall(r"\[\d+\]", answer or "")
