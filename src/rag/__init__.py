"""Lazy package exports for the ``src.rag`` namespace."""

from __future__ import annotations

from src.rag.export_map import SRC_RAG_ALL, resolve_src_rag_export

__all__ = SRC_RAG_ALL


def __getattr__(name: str):
    value = resolve_src_rag_export(name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
