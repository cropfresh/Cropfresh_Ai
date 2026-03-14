"""
CropFresh AI — Data & Scraper API Routes
==========================================
REST endpoints for managing data sources, scraping,
and AI Kosha dataset access.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.config import get_settings
from src.scrapers.agri_scrapers import AgriculturalDataAPI, DataSource
from src.scrapers.aikosha_client import AIKoshaCategory, AIKoshaClient

router = APIRouter(prefix="/data", tags=["data"])


# ============================================================================
# Request/Response Models
# ============================================================================


class ScrapeRequest(BaseModel):
    """Request to trigger a scrape job."""
    source: str = Field(
        ..., description="Data source (agmarknet, enam, imd)"
    )
    commodity: Optional[str] = Field(
        None, description="Commodity name (e.g., 'Tomato')"
    )
    state: Optional[str] = Field(
        None, description="State filter (e.g., 'Karnataka')"
    )


class ScrapeResponse(BaseModel):
    """Response from a scrape job."""
    source: str
    url: str
    record_count: int
    data: list[dict]
    duration_ms: float
    from_cache: bool
    error: Optional[str] = None
    success: bool


class DataSourceInfo(BaseModel):
    """Info about a data source."""
    name: str
    status: str
    total_requests: int
    successful_requests: int
    success_rate: float
    avg_response_ms: float
    last_error: Optional[str] = None


# ============================================================================
# Singleton instances (initialized lazily)
# ============================================================================

_agri_api: Optional[AgriculturalDataAPI] = None
_aikosha: Optional[AIKoshaClient] = None


def get_agri_api() -> AgriculturalDataAPI:
    """Get or create AgriculturalDataAPI singleton."""
    global _agri_api
    if _agri_api is None:
        _agri_api = AgriculturalDataAPI()
    return _agri_api


def get_aikosha_client() -> AIKoshaClient:
    """Get or create AIKoshaClient singleton."""
    global _aikosha
    if _aikosha is None:
        settings = get_settings()
        _aikosha = AIKoshaClient(
            api_key=settings.aikosha_api_key,
            base_url=settings.aikosha_base_url,
        )
    return _aikosha


# ============================================================================
# Scraping Endpoints
# ============================================================================


@router.post("/scrape", response_model=ScrapeResponse)
async def trigger_scrape(request: ScrapeRequest):
    """
    Trigger a scrape job for a specific data source.

    Supports: agmarknet, enam, imd
    """
    api = get_agri_api()

    if request.source in ("agmarknet", "enam"):
        if not request.commodity:
            raise HTTPException(
                status_code=400,
                detail="commodity is required for price scraping",
            )
        result = await api.get_mandi_prices(
            commodity=request.commodity,
            state=request.state,
            source=request.source,
        )
    elif request.source == "imd":
        result = await api.get_weather(
            state=request.state or "Karnataka",
            district=None,
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown source: {request.source}. Available: agmarknet, enam, imd",
        )

    return ScrapeResponse(
        source=result.source,
        url=result.url,
        record_count=result.record_count,
        data=result.data,
        duration_ms=result.duration_ms,
        from_cache=result.from_cache,
        error=result.error,
        success=result.success,
    )


@router.get("/sources")
async def list_data_sources():
    """List all available data sources with health status."""
    api = get_agri_api()
    health = api.get_all_health()

    aikosha = get_aikosha_client()
    aikosha_health = await aikosha.health_check()

    return {
        "sources": {
            **health,
            "ai_kosha": aikosha_health,
        },
        "available_sources": [s.value for s in DataSource],
    }


@router.get("/prices")
async def get_prices(
    commodity: str = Query(..., description="Commodity name"),
    state: Optional[str] = Query(None, description="State filter"),
    source: str = Query("agmarknet", description="Data source"),
):
    """Get commodity prices from the specified source."""
    api = get_agri_api()
    result = await api.get_mandi_prices(
        commodity=commodity, state=state, source=source
    )

    return {
        "commodity": commodity,
        "state": state,
        "source": result.source,
        "record_count": result.record_count,
        "prices": result.data,
        "from_cache": result.from_cache,
        "duration_ms": result.duration_ms,
    }


@router.get("/weather")
async def get_weather(
    state: str = Query("Karnataka", description="State name"),
    district: Optional[str] = Query(None, description="District name"),
):
    """Get weather forecast data."""
    api = get_agri_api()
    result = await api.get_weather(state=state, district=district)

    return {
        "state": state,
        "district": district,
        "record_count": result.record_count,
        "weather": result.data,
        "duration_ms": result.duration_ms,
    }


# ============================================================================
# AI Kosha Endpoints
# ============================================================================


@router.get("/aikosha/datasets")
async def search_aikosha_datasets(
    query: str = Query("", description="Search query"),
    category: Optional[str] = Query(None, description="Dataset category"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
):
    """Search AI Kosha datasets."""
    client = get_aikosha_client()

    # Map string category to enum
    cat_enum = None
    if category:
        try:
            cat_enum = AIKoshaCategory(category)
        except ValueError:
            # Try matching by key name
            cat_map = {c.name.lower(): c for c in AIKoshaCategory}
            cat_enum = cat_map.get(category.lower())

    result = await client.search_datasets(
        query=query, category=cat_enum, page=page, per_page=per_page
    )

    return {
        "total_results": result.total_results,
        "page": result.page,
        "per_page": result.per_page,
        "datasets": [d.model_dump() for d in result.datasets],
    }


@router.get("/aikosha/dataset/{dataset_id}")
async def get_aikosha_dataset(dataset_id: str):
    """Get details of a specific AI Kosha dataset."""
    client = get_aikosha_client()
    dataset = await client.get_dataset(dataset_id)

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return dataset.model_dump()


@router.get("/aikosha/categories")
async def list_aikosha_categories():
    """List available AI Kosha dataset categories."""
    return {
        "categories": [
            {"key": c.name, "value": c.value}
            for c in AIKoshaCategory
        ]
    }
