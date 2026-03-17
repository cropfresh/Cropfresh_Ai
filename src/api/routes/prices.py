"""FastAPI routes for price aggregation and the shared rate hub."""

from __future__ import annotations

from datetime import date as date_type
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.api.services.price_aggregator import AggregatedPriceResult
from src.rates.connectors import PENDING_SOURCES
from src.rates.factory import get_rate_service
from src.rates.models import MultiSourceRateResult
from src.rates.query_builder import normalize_rate_query
from src.rates.settings import get_agmarknet_api_key

router = APIRouter(prefix="/prices", tags=["prices"])


class RateQueryRequest(BaseModel):
    """HTTP request body for multi-source rate queries."""

    rate_kinds: list[str] = Field(default_factory=list, min_length=1)
    commodity: str | None = None
    state: str = "Karnataka"
    district: str | None = None
    market: str | None = None
    date: date_type | None = None
    include_reference: bool = True
    force_live: bool = False
    comparison_depth: str = "all_sources"


@router.get("/latest", response_model=AggregatedPriceResult)
async def get_latest_price(
    commodity: str,
    market: str,
    target_date: Optional[date_type] = None,
):
    """Get the latest aggregated price for a given commodity and market."""
    target = target_date or date_type.today()
    return {
        "commodity": commodity,
        "market": market,
        "target_date": target,
        "min_price": 2000,
        "max_price": 2500,
        "modal_price": 2250,
        "average_price": 2300,
        "median_price": 2250,
        "unit": "INR/Quintal",
        "record_count": 2,
        "sources_used": ["agmarknet", "data.gov.in"],
        "evidence_records": [
            {"source": "agmarknet", "modal_price": 2200, "date": target.isoformat()},
            {"source": "data.gov.in", "modal_price": 2300, "date": target.isoformat()},
        ],
        "caveats": [],
    }


@router.get("/history")
async def get_price_history(
    commodity: str,
    market: str,
    from_date: date_type,
    to_date: date_type,
):
    """Get historical aggregated prices for a given date range."""
    if from_date > to_date:
        raise HTTPException(status_code=400, detail="from_date must be before to_date")
    return {"message": "History endpoint mocked", "commodity": commodity, "market": market}


@router.get("/summary")
async def get_state_summary(
    commodity: str,
    state: str,
    target_date: Optional[date_type] = None,
):
    """Provide a summary of prices across multiple markets in a state."""
    return {"message": "Summary endpoint mocked", "commodity": commodity, "state": state}


@router.post("/query", response_model=MultiSourceRateResult)
async def query_multi_source_rates(
    payload: RateQueryRequest,
    request: Request,
) -> MultiSourceRateResult:
    """Query the official-first multi-source Karnataka rate hub."""
    service = await get_rate_service(
        redis_client=getattr(request.app.state, "redis", None),
        llm_provider=getattr(request.app.state, "llm", None),
        agmarknet_api_key=get_agmarknet_api_key(),
    )
    query = normalize_rate_query(**payload.model_dump(mode="python"))
    return await service.query(query)


@router.get("/source-health")
async def get_price_source_health(request: Request) -> dict[str, object]:
    """Report connector health and pending-source metadata."""
    service = await get_rate_service(
        redis_client=getattr(request.app.state, "redis", None),
        llm_provider=getattr(request.app.state, "llm", None),
        agmarknet_api_key=get_agmarknet_api_key(),
    )
    return {
        "sources": [snapshot.model_dump(mode="json") for snapshot in service.get_source_health()],
        "pending_sources": [source.model_dump(mode="json") for source in PENDING_SOURCES],
    }
