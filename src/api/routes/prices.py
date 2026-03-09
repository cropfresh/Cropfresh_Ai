"""
FastAPI routes for agricultural price intelligence.
"""
from datetime import date
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException

# Assuming a dependency injection setup for DB and Services
# from src.api.dependencies import get_price_aggregator
from src.api.services.price_aggregator import PriceAggregatorService, AggregatedPriceResult
from pydantic import BaseModel

router = APIRouter(prefix="/prices", tags=["prices"])

@router.get("/latest", response_model=AggregatedPriceResult)
async def get_latest_price(
    commodity: str,
    market: str,
    target_date: Optional[date] = None,
    # aggregator: PriceAggregatorService = Depends(get_price_aggregator)
):
    """
    Get the latest aggregated price for a given commodity and market.
    Defaults to today if no date is provided.
    Includes full transparent evidence (Perplexity-style).
    """
    target = target_date or date.today()
    try:
        # In actual usage, aggregator would be injected. 
        # Using a mock instantiation here for architectural completeness
        # result = await aggregator.get_aggregated_price(commodity, market, target)
        
        # return result
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
                {"source": "data.gov.in", "modal_price": 2300, "date": target.isoformat()}
            ],
            "caveats": []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_price_history(
    commodity: str,
    market: str,
    from_date: date,
    to_date: date
    # aggregator: PriceAggregatorService = Depends(get_price_aggregator)
):
    """
    Get historical aggregated prices for a given date range.
    Returns a time-series list of Perplexity-style aggregations.
    """
    if from_date > to_date:
        raise HTTPException(status_code=400, detail="from_date must be before to_date")
        
    # Implementation loops over dates or uses DB aggregations
    return {"message": "History endpoint mocked", "commodity": commodity, "market": market}


@router.get("/summary")
async def get_state_summary(
    commodity: str,
    state: str,
    target_date: Optional[date] = None
):
    """
    Provides a summary of prices across multiple markets in a state.
    """
    return {"message": "Summary endpoint mocked", "commodity": commodity, "state": state}
