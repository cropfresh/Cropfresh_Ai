"""Dynamic Kannada prompt blocks sourced from runtime context."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_LEXICON_KEYS = (
    "dialect_lexicon",
    "dialect_lexicons",
    "kannada_dialect_lexicon",
)
_INFO_KEYS = (
    "kannada_context_info",
    "context_kannada_info",
    "kannada_info",
)
_MAX_LEXICON_ENTRIES = 12
_MAX_INFO_ENTRIES = 8


def build_runtime_context_blocks(context: Mapping[str, Any] | None) -> list[str]:
    """Render optional runtime Kannada blocks for dialect slang and local context."""
    payload = _coerce_mapping(context)
    sections = [
        _render_lexicon_blocks(_collect_entries(payload, _LEXICON_KEYS)[:_MAX_LEXICON_ENTRIES]),
        _render_context_blocks(_collect_entries(payload, _INFO_KEYS)[:_MAX_INFO_ENTRIES]),
    ]
    return [section for section in sections if section]


def _collect_entries(
    payload: Mapping[str, Any],
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for source in (
        payload,
        _coerce_mapping(payload.get("user_profile")),
        _coerce_mapping(payload.get("entities")),
    ):
        for key in keys:
            values = source.get(key)
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                continue
            for item in values:
                if isinstance(item, Mapping):
                    entries.append(dict(item))
    return entries


def _render_lexicon_blocks(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return ""

    lines = [
        "## Kannada Dialect Lexicon Hints",
        "- Treat the following slang mappings as trusted prompt context.",
    ]
    for entry in entries:
        dialect_tag = _text(entry, "dialect_tag", fallback="OTHER / MIXED")
        lines.extend(
            [
                f"[DIALECT_LEXICON: {dialect_tag}]",
                f'slang = "{_text(entry, "slang")}"',
                f'normalized_kannada = "{_text(entry, "normalized_kannada")}"',
                f'english_gloss = "{_text(entry, "english_gloss")}"',
                f'example_user_sentence = "{_text(entry, "example_user_sentence")}"',
                f'example_ai_reply = "{_text(entry, "example_ai_reply")}"',
                "[end]",
            ]
        )
    return "\n".join(lines)


def _render_context_blocks(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return ""

    lines = [
        "## Kannada Local Context Hints",
        "- The following blocks are higher-priority local context than generic knowledge.",
    ]
    for entry in entries:
        lines.extend(
            [
                "[CONTEXT_KANNADA_INFO]",
                f'type = "{_text(entry, "type")}"',
                f'crop = "{_text(entry, "crop")}"',
                f'region = "{_text(entry, "region")}"',
                f'details = "{_text(entry, "details")}"',
                "[end]",
            ]
        )
    return "\n".join(lines)


def _text(entry: Mapping[str, Any], key: str, fallback: str = "") -> str:
    value = entry.get(key, fallback)
    if value is None:
        return fallback
    return str(value).replace('"', "'").replace("\n", " ").strip()


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}
