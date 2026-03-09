import pytest
from datetime import date
from src.db.models.price_records import NormalizedPriceRecord
from src.api.services.price_aggregator import PriceAggregatorService

class MockPriceRepository:
    async def get_prices(self, commodity: str, market: str, start_date: date, end_date: date):
        # Return mock records for testing aggregation logic
        return [
            NormalizedPriceRecord(
                commodity="Tomato", market="Kolar", price_date=date(2023, 10, 1),
                source="agmarknet", min_price=1000, max_price=1500, modal_price=1200, unit="INR/Quintal"
            ),
            NormalizedPriceRecord(
                commodity="Tomato", market="Kolar", price_date=date(2023, 10, 1),
                source="unknown_source", min_price=800, max_price=1200, modal_price=1000, unit="INR/Quintal"
            )
        ]

@pytest.mark.asyncio
async def test_price_aggregator():
    repo = MockPriceRepository()
    service = PriceAggregatorService(repository=repo)
    
    result = await service.get_aggregated_price("Tomato", "Kolar", date(2023, 10, 1))
    
    assert result.commodity == "Tomato"
    # Should use both records for aggregation stats in this simplistic test
    assert result.min_price == 800
    assert result.max_price == 1500
    # Median of 1200 and 1000 is 1100
    assert result.modal_price == 1100
    assert result.record_count == 2
    assert "agmarknet" in result.sources_used
