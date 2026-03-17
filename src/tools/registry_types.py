"""Annotation helpers for tool-registry schema generation."""

from __future__ import annotations

from typing import Any, get_args, get_origin


def get_annotation_name(annotation: Any) -> str:
    """Normalize modern Python annotations to simple schema-friendly names."""
    if annotation is Any:
        return "any"

    origin = get_origin(annotation)
    if origin is None:
        return getattr(annotation, "__name__", str(annotation))

    if origin in {list, set, tuple}:
        return "list"
    if origin is dict:
        return "dict"

    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if len(args) == 1:
        return get_annotation_name(args[0])
    return getattr(origin, "__name__", str(origin))
