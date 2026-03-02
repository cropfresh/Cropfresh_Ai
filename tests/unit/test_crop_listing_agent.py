"""
Unit tests for Crop Listing Agent — entity extraction and basic routing.
"""

import pytest
from unittest.mock import AsyncMock
from src.agents.crop_listing.agent import CropListingAgent

@pytest.fixture
def mock_service():
    service = AsyncMock()
    service.create_listing_from_dict.return_value = {
        "id": "list-123",
        "asking_price_per_kg": 25.0
    }
    service.get_farmer_listings.return_value = [
        {"commodity": "Tomato", "quantity_kg": 100, "asking_price_per_kg": 20.0, "grade": "A"}
    ]
    service.cancel_listing.return_value = True
    service.update_listing.return_value = True
    return service

class TestCropListingExtraction:
    def test_extract_commodity_aliases(self):
        agent = CropListingAgent()
        assert agent._extract_commodity("I have some tamatar") == "Tomato"
        assert agent._extract_commodity("selling pyaaz") == "Onion"
        assert agent._extract_commodity("I have 100kg of nothing") is None

    def test_extract_quantity(self):
        agent = CropListingAgent()
        assert agent._extract_quantity("selling 150 kg") == 150.0
        assert agent._extract_quantity("I have 2.5 kgs") == 2.5
        assert agent._extract_quantity("selling 5 quintals") == 500.0

    def test_extract_price(self):
        agent = CropListingAgent()
        assert agent._extract_price("at rs 25/kg") == 25.0
        assert agent._extract_price("price 15 per kg") == 15.0
        assert agent._extract_price("₹30") == 30.0

class TestCropListingRouting:
    @pytest.mark.asyncio
    async def test_create_listing_intent(self, mock_service):
        agent = CropListingAgent(listing_service=mock_service)
        response = await agent.process(
            "sell 100kg tomato at rs 20", 
            context={"farmer_id": "f-1"}
        )
        assert "has been created" in response.content
        assert "list-123" in response.content

    @pytest.mark.asyncio
    async def test_my_listings_intent(self, mock_service):
        agent = CropListingAgent(listing_service=mock_service)
        response = await agent.process("show my listings", context={"farmer_id": "f-1"})
        assert "Tomato" in response.content
        mock_service.get_farmer_listings.assert_called_with("f-1")

    @pytest.mark.asyncio
    async def test_cancel_listing_intent(self, mock_service):
        agent = CropListingAgent(listing_service=mock_service)
        response = await agent.process("cancel listing", context={"listing_id": "L1"})
        assert "cancelled" in response.content
        mock_service.cancel_listing.assert_called_with("L1")

    @pytest.mark.asyncio
    async def test_update_price_intent(self, mock_service):
        agent = CropListingAgent(listing_service=mock_service)
        response = await agent.process(
            "update price to rs 30", 
            context={"listing_id": "L1"}
        )
        assert "updated" in response.content
        mock_service.update_listing.assert_called_with("L1", {"asking_price_per_kg": 30.0})

class TestCropListingEdgeCases:
    @pytest.mark.asyncio
    async def test_missing_commodity(self):
        agent = CropListingAgent()
        r = await agent.process("sell 100kg")
        assert "Which crop" in r.content

    @pytest.mark.asyncio
    async def test_missing_quantity(self):
        agent = CropListingAgent()
        r = await agent.process("sell tomato")
        assert "How many kg" in r.content

    @pytest.mark.asyncio
    async def test_execute_api_create(self, mock_service):
        agent = CropListingAgent(listing_service=mock_service)
        result = await agent.execute({"action": "create", "farmer_id": "f-1"})
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_api_cancel(self, mock_service):
        agent = CropListingAgent(listing_service=mock_service)
        result = await agent.execute({"action": "cancel", "listing_id": "123"})
        assert result["success"] is True
