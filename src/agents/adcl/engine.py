"""Backward-compatible exports for the canonical ADCL service."""

from __future__ import annotations

from typing import Any

from src.agents.adcl.factory import get_adcl_service
from src.agents.adcl.service import ADCLService

ADCLAgent = ADCLService


def get_adcl_agent(
    db: Any | None = None,
    price_agent: Any | None = None,
    llm: Any | None = None,
) -> ADCLAgent:
    """Return the shared ADCL service for older direct-call sites and tests."""
    del price_agent
    return get_adcl_service(db=db, llm=llm)


__all__ = ["ADCLAgent", "ADCLService", "get_adcl_agent", "get_adcl_service"]
