"""ASGI entrypoint for the Sprint 08 VAD service."""

from __future__ import annotations

from .api import create_app

app = create_app()
