"""Lazy package exports for the ``ai.rag`` namespace."""

from __future__ import annotations

from ai.rag.export_map import AI_RAG_ALL, resolve_ai_rag_export

__all__ = AI_RAG_ALL


def __getattr__(name: str):
    value = resolve_ai_rag_export(name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
