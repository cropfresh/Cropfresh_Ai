"""
Aggregates and synthesizes price data from multiple sources.
"""
from typing import List, Dict, Any, Optional
from datetime import date
from pydantic import BaseModel
import statistics
from loguru import logger
from src.db.models.price_records import NormalizedPriceRecord
from src.db.repositories.price_repository import PriceRepository


class AggregatedPriceResult(BaseModel):
    """Result of combining multiple price sources."""
    commodity: str
    market: str
    target_date: date
    
    # Computed metrics
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    modal_price: Optional[float] = None
    median_price: Optional[float] = None
    average_price: Optional[float] = None
    
    unit: Optional[str] = None
    
    # Transparency / Evidence
    record_count: int
    sources_used: List[str]
    evidence_records: List[Dict[str, Any]]
    caveats: List[str]


class PriceAggregatorService:
    """
    Perplexity-inspired ranking and aggregation engine.
    Given a query, it retrieves raw records, filters/ranks them,
    and returns synthesized statistics with transparent evidence.
    """
    
    # Ranking logic: lower score is better (primary govt sources preferred)
    SOURCE_RANKING = {
        "agmarknet": 1,
        "data.gov.in": 2,
        "enam": 3,
        "ceda": 4,
        "community": 5,
        "unknown": 99
    }
    
    def __init__(self, repository: PriceRepository):
        self.repo = repository

    async def get_aggregated_price(
        self, 
        commodity: str, 
        market: str, 
        target_date: date,
        window_days: int = 3
    ) -> AggregatedPriceResult:
        """
        Fetch, rank, and aggregate prices for a specific target.
        Falls back to a time window (e.g. up to 3 days prior) if exact date is missing.
        """
        # 1. Retrieval
        from datetime import timedelta
        start_date = target_date - timedelta(days=window_days)
        
        records = await self.repo.get_prices(commodity, market, start_date, target_date)
        
        caveats = []
        if not records:
            return AggregatedPriceResult(
                commodity=commodity, market=market, target_date=target_date,
                record_count=0, sources_used=[], evidence_records=[],
                caveats=[f"No data found for {commodity} in {market} between {start_date} and {target_date}"]
            )
            
        # 2. Filtering & Ranking (Prefer primary sources, discard outliers if needed)
        # Sort by best source first, then newest date
        ranked_records = sorted(
            records,
            key=lambda r: (self.SOURCE_RANKING.get(r.source.lower(), 99), -r.price_date.toordinal())
        )
        
        # We might only want to use the best available source's data if it's high quality, 
        # or combine all if we want an average across sources.
        # For this design, let's aggregate *all* valid records found in the window 
        # but expose the sources cleanly.
        
        valid_min = [r.min_price for r in ranked_records if r.min_price is not None]
        valid_max = [r.max_price for r in ranked_records if r.max_price is not None]
        valid_modal = [r.modal_price for r in ranked_records if r.modal_price is not None]
        
        # 3. Aggregation (Synthesis)
        # Compute central metrics
        agg_min = min(valid_min) if valid_min else None
        agg_max = max(valid_max) if valid_max else None
        
        # Modal price can be the literal provided modal from govt, or calculated median of modals
        agg_modal = statistics.median(valid_modal) if valid_modal else None
        agg_avg = statistics.mean(valid_modal) if valid_modal else None
        
        # Unit consensus (just take the first most reliable one)
        resolved_unit = next((r.unit for r in ranked_records if r.unit), "Unknown")
        
        sources_used = list(set([r.source for r in ranked_records]))
        
        if len(sources_used) == 1:
            caveats.append(f"Only 1 source available ({sources_used[0]}).")
            
        oldest_data_used = min([r.price_date for r in ranked_records])
        if (target_date - oldest_data_used).days > 1:
            caveats.append(f"Data includes records up to {(target_date - oldest_data_used).days} days old.")

        # Serialize evidence for transparency
        evidence = [
            {
                "id": r.id,
                "source": r.source,
                "date": r.price_date.isoformat(),
                "modal_price": r.modal_price,
                "raw_record_id": r.raw_record_id
            }
            for r in ranked_records
        ]
            
        return AggregatedPriceResult(
            commodity=commodity,
            market=market,
            target_date=target_date,
            min_price=agg_min,
            max_price=agg_max,
            modal_price=agg_modal,
            average_price=agg_avg,
            median_price=agg_modal, # Simplified mapping for now
            unit=resolved_unit,
            record_count=len(ranked_records),
            sources_used=sources_used,
            evidence_records=evidence,
            caveats=caveats
        )
