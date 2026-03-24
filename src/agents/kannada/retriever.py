"""Retrieve query-relevant Kannada prompt snippets from structured data."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from src.agents.kannada.data_loader import (
    load_dialect_lexicon_entries,
    load_domain_context_entries,
)
from src.agents.kannada.dialect_context import (
    extract_location_signal,
    resolve_dialect_bucket,
)
from src.agents.kannada.domain_resolution import resolve_domain_name

_WORD_RE = re.compile(r"[a-zA-Z\u0C80-\u0CFF']+")


def enrich_runtime_context(
    domain_name: str | None,
    context: Mapping[str, Any] | None = None,
    query: str = "",
    max_lexicon: int = 4,
    max_context_entries: int = 4,
) -> dict[str, Any]:
    """Merge retrieved Kannada entries into the current runtime context."""
    payload = deepcopy(dict(context or {}))
    existing_lexicon = _collect_existing_entries(payload, ("dialect_lexicon",))
    existing_info = _collect_existing_entries(
        payload,
        ("kannada_context_info", "context_kannada_info", "kannada_info"),
    )
    payload["dialect_lexicon"] = _merge_entries(
        existing_lexicon,
        retrieve_dialect_lexicon(query, payload, max_items=max_lexicon),
        key_fields=("dialect_tag", "slang"),
    )
    payload["kannada_context_info"] = _merge_entries(
        existing_info,
        retrieve_domain_context(
            domain_name,
            query=query,
            context=payload,
            max_items=max_context_entries,
        ),
        key_fields=("type", "crop", "region", "details"),
    )
    return payload


def retrieve_dialect_lexicon(
    query: str,
    context: Mapping[str, Any] | None = None,
    max_items: int = 4,
) -> list[dict[str, Any]]:
    """Retrieve dialect lexicon entries relevant to the current query."""
    if not query.strip():
        return []

    payload = _coerce_mapping(context)
    bucket = resolve_dialect_bucket(payload)
    location = extract_location_signal(payload).lower()
    query_lower = query.lower()
    tokens = _tokenize(query)
    scored: list[tuple[int, dict[str, Any]]] = []

    for entry in load_dialect_lexicon_entries():
        aliases = [entry.get("slang", ""), *entry.get("aliases", [])]
        alias_hit = any(
            isinstance(alias, str) and alias.lower() in query_lower for alias in aliases if alias
        )
        keyword_hit = any(
            isinstance(keyword, str) and keyword.lower() in tokens
            for keyword in entry.get("keywords", [])
        )
        district_hit = any(
            isinstance(district, str) and district.lower() in location
            for district in entry.get("districts", [])
        )
        bucket_hit = bucket and entry.get("dialect_tag") == bucket
        if not (alias_hit or keyword_hit or district_hit):
            continue

        score = 0
        if alias_hit:
            score += 8
        if keyword_hit:
            score += 2
        if district_hit:
            score += 2
        if bucket_hit:
            score += 3
        scored.append((score, _normalize_entry(entry)))

    return _top_entries(scored, max_items)


def retrieve_domain_context(
    domain_name: str | None,
    query: str = "",
    context: Mapping[str, Any] | None = None,
    max_items: int = 4,
) -> list[dict[str, Any]]:
    """Retrieve domain-specific Kannada guidance for the current turn."""
    payload = _coerce_mapping(context)
    domain = resolve_domain_name(domain_name)
    query_lower = query.lower()
    tokens = _tokenize(query)
    location = extract_location_signal(payload).lower()
    crops = _collect_crop_terms(payload, query_lower)
    if not (query_lower.strip() or crops or location):
        return []
    scored: list[tuple[int, dict[str, Any]]] = []

    for entry in load_domain_context_entries():
        entry_domain = resolve_domain_name(str(entry.get("domain", "")))
        crop = str(entry.get("crop", "")).lower()
        crop_hit = bool(crop and crop in crops)
        district_hit = any(
            isinstance(district, str) and district.lower() in location
            for district in entry.get("districts", [])
        )
        keyword_hit = any(
            isinstance(keyword, str) and keyword.lower() in tokens
            for keyword in entry.get("keywords", [])
        )

        score = 0
        if entry_domain == domain:
            score += 6
        elif entry_domain == "general":
            score += 1
        if crop and not (crop_hit or keyword_hit):
            continue
        if crop_hit:
            score += 4
        if district_hit:
            score += 3
        if keyword_hit:
            score += 2
        if score <= 1:
            continue
        scored.append((score, _normalize_entry(entry)))

    return _top_entries(scored, max_items)


def _collect_existing_entries(
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
            value = source.get(key)
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                continue
            for item in value:
                if isinstance(item, Mapping):
                    entries.append(dict(item))
    return entries


def _merge_entries(
    existing: list[dict[str, Any]],
    retrieved: list[dict[str, Any]],
    key_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for entry in [*existing, *retrieved]:
        key = tuple(str(entry.get(field, "")).strip().lower() for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        merged.append(entry)
    return merged


def _collect_crop_terms(payload: Mapping[str, Any], query_lower: str) -> set[str]:
    terms = set(_tokenize(query_lower))
    profile = _coerce_mapping(payload.get("user_profile"))
    entities = _coerce_mapping(payload.get("entities"))
    for item in profile.get("crops", []):
        if isinstance(item, str):
            terms.add(item.lower())
    for key in ("crop", "commodity"):
        value = entities.get(key) or payload.get(key)
        if isinstance(value, str) and value.strip():
            terms.add(value.lower())
    return terms


def _top_entries(
    scored: list[tuple[int, dict[str, Any]]],
    max_items: int,
) -> list[dict[str, Any]]:
    ordered = sorted(scored, key=lambda item: item[0], reverse=True)
    return [entry for _, entry in ordered[:max_items]]


def _normalize_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in dict(entry).items() if value not in ("", None, [])}


def _tokenize(text: str) -> set[str]:
    return {match.group(0).lower() for match in _WORD_RE.finditer(text) if len(match.group(0)) > 2}


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}
