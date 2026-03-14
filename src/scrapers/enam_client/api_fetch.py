"""
eNAM API Fetching Logic
"""

import httpx
from datetime import datetime
from typing import Optional
from loguru import logger

from .models import MandiPrice
from .constants import COMMODITY_CODES, STATE_CODES, APISETU_BASE


async def fetch_live_prices(
    api_key: str,
    commodity: str,
    state: str,
    district: Optional[str],
    market: Optional[str],
    limit: int,
) -> list[MandiPrice]:
    """Fetch prices from eNAM API."""
    commodity_code = COMMODITY_CODES.get(commodity.lower(), commodity.upper())
    state_code = STATE_CODES.get(state.lower(), state.upper()[:2])
    
    headers = {
        "X-Api-Key": api_key,
        "Accept": "application/json",
    }
    
    params = {
        "commodity": commodity_code,
        "state": state_code,
        "limit": limit,
    }
    
    if district:
        params["district"] = district
    if market:
        params["market"] = market
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{APISETU_BASE}/commodity/prices",
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()
            
            return parse_api_response(data, commodity, state)
            
    except httpx.HTTPError as e:
        logger.warning(f"eNAM API error: {e}, falling back to Agmarknet")
        return await fallback_to_agmarknet(commodity, state, district, limit)
    except Exception as e:
        logger.error(f"Unexpected error fetching prices: {e}")
        from .mock_data import get_mock_prices_data
        return get_mock_prices_data(commodity, state, district, limit)


def parse_api_response(
    data: dict,
    commodity: str,
    state: str,
) -> list[MandiPrice]:
    """Parse eNAM API response into MandiPrice objects."""
    prices = []
    
    for record in data.get("data", data.get("records", [])):
        try:
            prices.append(MandiPrice(
                commodity=record.get("commodity", commodity),
                variety=record.get("variety", ""),
                state=record.get("state", state),
                district=record.get("district", ""),
                market=record.get("market", ""),
                min_price=float(record.get("minPrice", record.get("min_price", 0))),
                max_price=float(record.get("maxPrice", record.get("max_price", 0))),
                modal_price=float(record.get("modalPrice", record.get("modal_price", 0))),
                arrival_qty=float(record.get("arrivals", 0)),
                traded_qty=float(record.get("traded", 0)),
                price_date=datetime.fromisoformat(record.get("date", datetime.now().isoformat())),
                source="enam_api",
            ))
        except Exception as e:
            logger.debug(f"Error parsing record: {e}")
            continue
    
    return prices


async def fallback_to_agmarknet(
    commodity: str,
    state: str,
    district: Optional[str],
    limit: int,
) -> list[MandiPrice]:
    """Fallback to Agmarknet if eNAM fails."""
    # Assuming agmarknet is a separate tool available in the codebase
    from src.tools.agmarknet import AgmarknetTool
    
    agmarknet = AgmarknetTool()
    agmarknet_prices = await agmarknet.get_prices(
        commodity=commodity,
        state=state,
        district=district,
        limit=limit,
    )
    
    # Convert to MandiPrice format
    return [
        MandiPrice(
            commodity=p.commodity,
            variety=p.variety,
            state=p.state,
            district=p.district,
            market=p.market,
            min_price=p.min_price,
            max_price=p.max_price,
            modal_price=p.modal_price,
            price_date=p.date,
            source="agmarknet_fallback",
        )
        for p in agmarknet_prices
    ]
