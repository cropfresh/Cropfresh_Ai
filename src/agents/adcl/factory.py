"""Shared factory for the app-wide ADCL service singleton."""

from __future__ import annotations

from typing import Any

from src.agents.adcl.service import ADCLService

_service: ADCLService | None = None


def get_adcl_service(
    db: Any | None = None,
    rate_service: Any | None = None,
    llm: Any | None = None,
    imd_client: Any | None = None,
    enam_client: Any | None = None,
    enable_enam: bool | None = None,
) -> ADCLService:
    """Return a shared ADCL service instance and upgrade dependencies when provided."""
    global _service
    if _service is None:
        _service = ADCLService(
            db=db,
            rate_service=rate_service,
            llm=llm,
            imd_client=imd_client,
            enam_client=enam_client,
            enable_enam=bool(enable_enam),
        )
        return _service

    _service.update_dependencies(
        db=db,
        rate_service=rate_service,
        llm=llm,
        imd_client=imd_client,
        enam_client=enam_client,
        enable_enam=enable_enam,
    )
    return _service
