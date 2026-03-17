"""REST routes for the canonical ADCL service."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(prefix="/adcl", tags=["adcl"])


@router.get("/weekly")
async def get_weekly_adcl_report(
    request: Request,
    district: str = Query(..., min_length=2),
    force_live: bool = Query(default=False),
    farmer_id: str | None = Query(default=None),
    language: str | None = Query(default=None),
) -> dict[str, object]:
    """Return the canonical district-scoped weekly ADCL payload."""
    service = getattr(request.app.state, "adcl_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="ADCL service is not initialized")

    report = await service.generate_weekly_report(
        district=district,
        force_live=force_live,
        farmer_id=farmer_id,
        language=language,
    )
    return report.to_dict()
