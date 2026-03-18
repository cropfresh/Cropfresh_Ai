"""Cached loaders for structured Kannada prompt data."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).with_name("data")


@lru_cache(maxsize=1)
def load_dialect_lexicon_entries() -> list[dict[str, Any]]:
    """Load reusable Kannada dialect lexicon entries from JSONL."""
    return _load_jsonl("dialect_lexicon.jsonl")


@lru_cache(maxsize=1)
def load_domain_context_entries() -> list[dict[str, Any]]:
    """Load domain-specific Kannada context entries from JSONL."""
    return _load_jsonl("domain_context.jsonl")


def _load_jsonl(filename: str) -> list[dict[str, Any]]:
    path = DATA_DIR / filename
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entries.append(json.loads(line))
    return entries
