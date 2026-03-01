"""
Unit Tests — Listing Service & Crop Listing Agent
==================================================
Tests for:
1. ListingService.create_listing — auto-price, expiry, QR code, ADCL tag
2. ListingService.search_listings — filter + pagination
3. ListingService.get_listing / update_listing / cancel_listing
4. ListingService.get_farmer_listings
5. ListingService.attach_grade — HITL flag logic
6. ListingService.expire_stale_listings — background job
7. CropListingAgent.process — create, my_listings, cancel, update_price
8. CropListingAgent.execute — structured dict interface
9. _get_shelf_life helper — commodity mapping
10. Voice agent create_listing intent creates real DB record (AC6)
"""

# * TEST MODULE: Listing Service + CropListingAgent
# NOTE: All external dependencies (DB, pricing agent, quality agent) are mocked

from datetime import date, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.crop_listing.agent import CropListingAgent
from src.api.services.listing_service import (
    CreateListingRequest,
    GradeAttachRequest,
    ListingService,
    UpdateListingRequest,
    get_listing_service,
    SHELF_LIFE_DAYS,
    GRADE_ORDER,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def mock_db():
    """Mock AuroraPostgresClient with listing CRUD methods."""
    db = AsyncMock()
    db.create_listing = AsyncMock(return_value="listing-uuid-001")
    db.get_listing = AsyncMock(return_value=None)
    db.search_listings = AsyncMock(return_value=[])
    db.update_listing = AsyncMock(return_value={"id": "listing-uuid-001", "status": "active"})
    db.expire_stale_listings = AsyncMock(return_value=3)
    return db


@pytest.fixture
def mock_pricing_agent():
    """Mock agent with predict() returning a PricePrediction-like object."""
    agent = AsyncMock()
    prediction = MagicMock()
    prediction.current_price = 2500.0    # ₹/quintal — service converts to /kg
    agent.predict = AsyncMock(return_value=prediction)
    return agent


@pytest.fixture
def mock_quality_agent():
    """Mock QualityAssessmentAgent."""
    agent = AsyncMock()
    agent.assess = AsyncMock(return_value={"hitl_required": False, "grade": "A"})
    return agent


@pytest.fixture
def mock_adcl_agent():
    """Mock ADCL agent that marks tomato as in-demand."""
    agent = AsyncMock()
    agent.get_weekly_demand = AsyncMock(return_value={
        "crops": [{"crop": "Tomato", "demand_score": 0.9}]
    })
    return agent


@pytest.fixture
def service(mock_db, mock_pricing_agent, mock_quality_agent, mock_adcl_agent):
    """Fully wired ListingService with all mock dependencies."""
    return ListingService(
        db=mock_db,
        pricing_agent=mock_pricing_agent,
        quality_agent=mock_quality_agent,
        adcl_agent=mock_adcl_agent,
    )


@pytest.fixture
def bare_service():
    """ListingService with no dependencies (all None)."""
    return ListingService()


def _make_request(**kwargs) -> CreateListingRequest:
    defaults = {
        "farmer_id": "farmer-001",
        "commodity": "Tomato",
        "quantity_kg": 100.0,
        "asking_price_per_kg": 25.0,
    }
    defaults.update(kwargs)
    return CreateListingRequest(**defaults)


def _mock_listing_row(**overrides) -> dict:
    base = {
        "id": "listing-uuid-001",
        "farmer_id": "farmer-001",
        "commodity": "Tomato",
        "quantity_kg": 100.0,
        "asking_price_per_kg": 25.0,
        "grade": "Unverified",
        "status": "active",
        "hitl_required": False,
        "adcl_tagged": False,
    }
    base.update(overrides)
    return base


# ═══════════════════════════════════════════════════════════════
# Shelf Life Tests
# ═══════════════════════════════════════════════════════════════

class TestShelfLife:
    """Tests for commodity shelf-life lookup."""

    def test_tomato_shelf_life_is_7_days(self):
        assert ListingService._get_shelf_life("Tomato") == 7

    def test_onion_shelf_life_is_60_days(self):
        assert ListingService._get_shelf_life("Onion") == 60

    def test_potato_shelf_life_is_90_days(self):
        assert ListingService._get_shelf_life("Potato") == 90

    def test_case_insensitive_lookup(self):
        assert ListingService._get_shelf_life("TOMATO") == 7
        assert ListingService._get_shelf_life("tomato") == 7

    def test_unknown_crop_returns_default(self):
        assert ListingService._get_shelf_life("dragon fruit") == SHELF_LIFE_DAYS["default"]

    def test_all_listed_commodities_have_shelf_life(self):
        for crop in ["beans", "okra", "carrot", "cauliflower", "cucumber", "chilli"]:
            assert ListingService._get_shelf_life(crop) > 0


# ═══════════════════════════════════════════════════════════════
# CreateListing Tests
# ═══════════════════════════════════════════════════════════════

class TestCreateListing:
    """Tests for listing creation with auto-enrichment."""

    @pytest.mark.asyncio
    async def test_create_listing_returns_listing_response(self, service):
        """create_listing returns a ListingResponse with id, commodity, price."""
        req = _make_request()
        result = await service.create_listing(req)
        assert result.id == "listing-uuid-001"
        assert result.commodity == "Tomato"
        assert result.asking_price_per_kg == 25.0

    @pytest.mark.asyncio
    async def test_auto_price_suggested_when_none_given(self, service):
        """When no price provided, pricing agent's value is used as suggested_price."""
        req = _make_request(asking_price_per_kg=None)
        result = await service.create_listing(req)
        # * pricing agent returns 2500.0 quintal → ₹25/kg
        assert result.suggested_price == 25.0
        assert result.asking_price_per_kg == 25.0

    @pytest.mark.asyncio
    async def test_expiry_set_from_shelf_life(self, service):
        """expires_at is set ~7 days ahead for Tomato."""
        req = _make_request()
        result = await service.create_listing(req)
        assert result.expires_at is not None
        delta = result.expires_at - datetime.utcnow()
        assert 6 <= delta.days <= 8

    @pytest.mark.asyncio
    async def test_batch_qr_code_generated(self, service):
        """A unique batch QR code is created for every listing."""
        req = _make_request()
        result = await service.create_listing(req)
        assert result.batch_qr_code is not None
        assert result.batch_qr_code.startswith("CF-TOM-")

    @pytest.mark.asyncio
    async def test_adcl_tag_set_for_in_demand_crop(self, service):
        """Tomato is in the ADCL demand list → adcl_tagged=True."""
        req = _make_request(commodity="Tomato")
        result = await service.create_listing(req)
        assert result.adcl_tagged is True

    @pytest.mark.asyncio
    async def test_adcl_tag_false_for_non_demand_crop(self, service):
        """Potato is NOT in mock ADCL list → adcl_tagged=False."""
        req = _make_request(commodity="Potato")
        result = await service.create_listing(req)
        assert result.adcl_tagged is False

    @pytest.mark.asyncio
    async def test_quality_assessment_triggered_with_photos(self, service, mock_quality_agent):
        """Quality agent is called when photos list is non-empty."""
        req = _make_request(photos=["s3://bucket/photo1.jpg"])
        await service.create_listing(req)
        mock_quality_agent.assess.assert_called_once()

    @pytest.mark.asyncio
    async def test_quality_assessment_not_triggered_without_photos(self, service, mock_quality_agent):
        """Quality agent is NOT called when no photos provided."""
        req = _make_request(photos=None)
        await service.create_listing(req)
        mock_quality_agent.assess.assert_not_called()

    @pytest.mark.asyncio
    async def test_hitl_required_when_quality_agent_unavailable_but_photos_given(self):
        """When photos given but no quality agent, hitl_required defaults to True."""
        svc = ListingService(db=AsyncMock(create_listing=AsyncMock(return_value="id")))
        req = _make_request(photos=["s3://bucket/p.jpg"])
        result = await svc.create_listing(req)
        assert result.hitl_required is True

    @pytest.mark.asyncio
    async def test_create_uses_fallback_price_when_pricing_agent_unavailable(self):
        """Uses ₹25/kg hard fallback when pricing agent is absent and no price given."""
        svc = ListingService(db=AsyncMock(create_listing=AsyncMock(return_value="id")))
        req = _make_request(asking_price_per_kg=None)
        result = await svc.create_listing(req)
        assert result.asking_price_per_kg == 25.0

    @pytest.mark.asyncio
    async def test_create_listing_from_dict_works(self, service):
        """create_listing_from_dict wraps create_listing and returns a dict."""
        data = {
            "farmer_id": "farmer-001",
            "commodity": "Onion",
            "quantity_kg": 200.0,
            "asking_price_per_kg": 18.0,
        }
        result = await service.create_listing_from_dict(data)
        assert isinstance(result, dict)
        assert result["commodity"] == "Onion"


# ═══════════════════════════════════════════════════════════════
# SearchListings Tests
# ═══════════════════════════════════════════════════════════════

class TestSearchListings:
    """Tests for paginated search with filters."""

    @pytest.mark.asyncio
    async def test_search_returns_paginated_result(self, service, mock_db):
        """search_listings returns PaginatedListings with items and pagination."""
        mock_db.search_listings.return_value = [_mock_listing_row()]
        result = await service.search_listings({"commodity": "Tomato"})
        assert len(result.items) == 1
        assert result.limit == 20
        assert result.has_more is False

    @pytest.mark.asyncio
    async def test_search_empty_returns_empty_page(self, service, mock_db):
        """Empty DB results return empty PaginatedListings."""
        mock_db.search_listings.return_value = []
        result = await service.search_listings()
        assert result.items == []
        assert result.has_more is False

    @pytest.mark.asyncio
    async def test_has_more_true_when_extra_row_returned(self, service, mock_db):
        """has_more is True when DB returns limit+1 rows."""
        rows = [_mock_listing_row(id=f"id-{i}") for i in range(21)]
        mock_db.search_listings.return_value = rows
        result = await service.search_listings(limit=20)
        assert result.has_more is True
        assert len(result.items) == 20

    @pytest.mark.asyncio
    async def test_min_grade_filter_removes_lower_grades(self, service, mock_db):
        """min_grade='A' filters out grade B and Unverified rows."""
        rows = [
            _mock_listing_row(id="1", grade="A+"),
            _mock_listing_row(id="2", grade="A"),
            _mock_listing_row(id="3", grade="B"),
            _mock_listing_row(id="4", grade="Unverified"),
        ]
        mock_db.search_listings.return_value = rows
        result = await service.search_listings({"min_grade": "A"})
        grades = [item.grade for item in result.items]
        assert "B" not in grades
        assert "Unverified" not in grades
        assert "A" in grades

    @pytest.mark.asyncio
    async def test_search_no_db_returns_empty(self, bare_service):
        """Service without DB returns empty list gracefully."""
        result = await bare_service.search_listings()
        assert result.items == []


# ═══════════════════════════════════════════════════════════════
# Get / Update / Cancel Tests
# ═══════════════════════════════════════════════════════════════

class TestGetUpdateCancel:
    """Tests for single-record operations."""

    @pytest.mark.asyncio
    async def test_get_listing_returns_response(self, service, mock_db):
        """get_listing returns ListingResponse when row found."""
        mock_db.get_listing = AsyncMock(return_value=_mock_listing_row())
        result = await service.get_listing("listing-uuid-001")
        assert result is not None
        assert result.id == "listing-uuid-001"

    @pytest.mark.asyncio
    async def test_get_listing_returns_none_when_missing(self, service, mock_db):
        """get_listing returns None when no DB row."""
        mock_db.get_listing = AsyncMock(return_value=None)
        result = await service.get_listing("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_listing_returns_response(self, service, mock_db):
        """update_listing returns updated ListingResponse."""
        mock_db.update_listing.return_value = _mock_listing_row(asking_price_per_kg=30.0)
        result = await service.update_listing("listing-uuid-001", {"asking_price_per_kg": 30.0})
        assert result is not None
        assert result.asking_price_per_kg == 30.0

    @pytest.mark.asyncio
    async def test_cancel_listing_returns_true(self, service, mock_db):
        """cancel_listing returns True when DB update succeeds."""
        mock_db.update_listing.return_value = {"id": "listing-uuid-001", "status": "cancelled"}
        result = await service.cancel_listing("listing-uuid-001")
        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_listing_returns_false_without_db(self, bare_service):
        """cancel_listing returns False when no DB configured."""
        result = await bare_service.cancel_listing("any-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_farmer_listings_returns_list(self, service, mock_db):
        """get_farmer_listings returns list of ListingResponse."""
        mock_db.search_listings.return_value = [
            _mock_listing_row(id="1"),
            _mock_listing_row(id="2"),
        ]
        result = await service.get_farmer_listings("farmer-001")
        assert len(result) == 2


# ═══════════════════════════════════════════════════════════════
# Grade Attachment Tests
# ═══════════════════════════════════════════════════════════════

class TestAttachGrade:
    """Tests for quality grade attachment with HITL logic."""

    @pytest.mark.asyncio
    async def test_attach_grade_a_sets_hitl_false_for_high_confidence(self, service, mock_db):
        """Grade A with confidence 0.85 → hitl_required=False."""
        mock_db.update_listing.return_value = _mock_listing_row(
            grade="A", cv_confidence=0.85, hitl_required=False
        )
        req = GradeAttachRequest(grade="A", cv_confidence=0.85)
        result = await service.attach_grade("listing-uuid-001", req)
        # * Verify update was called with hitl_required=False
        call_updates = mock_db.update_listing.call_args[0][1]
        assert call_updates["hitl_required"] is False

    @pytest.mark.asyncio
    async def test_attach_grade_low_confidence_sets_hitl_true(self, service, mock_db):
        """Grade B with confidence 0.55 → hitl_required=True."""
        mock_db.update_listing.return_value = _mock_listing_row(grade="B", hitl_required=True)
        req = GradeAttachRequest(grade="B", cv_confidence=0.55)
        result = await service.attach_grade("listing-uuid-001", req)
        call_updates = mock_db.update_listing.call_args[0][1]
        assert call_updates["hitl_required"] is True

    @pytest.mark.asyncio
    async def test_attach_grade_a_plus_always_sets_hitl_true(self, service, mock_db):
        """Grade A+ always requires HITL regardless of confidence."""
        mock_db.update_listing.return_value = _mock_listing_row(grade="A+", hitl_required=True)
        req = GradeAttachRequest(grade="A+", cv_confidence=0.99)
        result = await service.attach_grade("listing-uuid-001", req)
        call_updates = mock_db.update_listing.call_args[0][1]
        assert call_updates["hitl_required"] is True

    @pytest.mark.asyncio
    async def test_attach_grade_returns_none_when_listing_not_found(self, service, mock_db):
        """attach_grade returns None when listing doesn't exist."""
        mock_db.update_listing.return_value = None
        req = GradeAttachRequest(grade="A")
        result = await service.attach_grade("nonexistent", req)
        assert result is None


# ═══════════════════════════════════════════════════════════════
# Expire Stale Listings Tests
# ═══════════════════════════════════════════════════════════════

class TestExpireStaleListings:
    """Tests for background expiry job."""

    @pytest.mark.asyncio
    async def test_expire_stale_listings_returns_count(self, service, mock_db):
        """expire_stale_listings returns count from DB."""
        result = await service.expire_stale_listings()
        assert result == 3

    @pytest.mark.asyncio
    async def test_expire_stale_listings_returns_zero_without_db(self, bare_service):
        """expire_stale_listings returns 0 when no DB configured."""
        result = await bare_service.expire_stale_listings()
        assert result == 0


# ═══════════════════════════════════════════════════════════════
# CropListingAgent Tests
# ═══════════════════════════════════════════════════════════════

class TestCropListingAgent:
    """Tests for agent natural language interface."""

    def _make_agent(self, svc=None):
        return CropListingAgent(listing_service=svc)

    @pytest.mark.asyncio
    async def test_process_creates_listing_from_natural_language(self):
        """Agent parses 'sell 200 kg tomatoes at ₹25 per kg' and calls service."""
        mock_svc = AsyncMock()
        mock_svc.create_listing_from_dict = AsyncMock(return_value={
            "id": "new-id",
            "commodity": "Tomato",
            "quantity_kg": 200.0,
            "asking_price_per_kg": 25.0,
        })
        agent = self._make_agent(mock_svc)
        response = await agent.process(
            "I want to sell 200 kg tomatoes at ₹25 per kg",
            context={"farmer_id": "farmer-001"},
        )
        assert "Tomato" in response.content or "tomato" in response.content.lower()
        mock_svc.create_listing_from_dict.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_asks_for_commodity_when_missing(self):
        """Agent asks clarifying question when no commodity detected."""
        agent = self._make_agent()
        response = await agent.process("I want to sell 100 kg at Rs 20")
        assert "crop" in response.content.lower() or "which" in response.content.lower()

    @pytest.mark.asyncio
    async def test_process_asks_for_quantity_when_missing(self):
        """Agent asks for quantity when only commodity present."""
        agent = self._make_agent()
        response = await agent.process("I want to sell tomatoes", context={"farmer_id": "f1"})
        assert "kg" in response.content.lower() or "many" in response.content.lower()

    @pytest.mark.asyncio
    async def test_process_my_listings_returns_summary(self):
        """'my listings' query triggers get_farmer_listings and formats response."""
        mock_svc = AsyncMock()
        mock_svc.get_farmer_listings = AsyncMock(return_value=[
            MagicMock(commodity="Tomato", quantity_kg=100.0,
                      asking_price_per_kg=25.0, grade="A"),
        ])
        agent = self._make_agent(mock_svc)
        response = await agent.process(
            "show my listings", context={"farmer_id": "farmer-001"}
        )
        assert "Tomato" in response.content or "listing" in response.content.lower()

    @pytest.mark.asyncio
    async def test_process_my_listings_empty_message_when_no_listings(self):
        """Graceful 'no active listings' message when list is empty."""
        mock_svc = AsyncMock()
        mock_svc.get_farmer_listings = AsyncMock(return_value=[])
        agent = self._make_agent(mock_svc)
        response = await agent.process("my listings", context={"farmer_id": "f1"})
        assert "no" in response.content.lower() or "active" in response.content.lower()

    @pytest.mark.asyncio
    async def test_process_cancel_listing(self):
        """Cancel intent with listing_id cancels the listing."""
        mock_svc = AsyncMock()
        mock_svc.cancel_listing = AsyncMock(return_value=True)
        agent = self._make_agent(mock_svc)
        response = await agent.process(
            "cancel my listing",
            context={"farmer_id": "f1", "listing_id": "listing-001"},
        )
        assert "cancel" in response.content.lower()
        mock_svc.cancel_listing.assert_called_once_with("listing-001")

    @pytest.mark.asyncio
    async def test_process_update_price(self):
        """Price update intent updates the listing price."""
        mock_svc = AsyncMock()
        mock_svc.update_listing = AsyncMock(return_value={
            "id": "listing-001", "asking_price_per_kg": 30.0
        })
        agent = self._make_agent(mock_svc)
        response = await agent.process(
            "update price to Rs 30 per kg",
            context={"farmer_id": "f1", "listing_id": "listing-001"},
        )
        assert "30" in response.content or "price" in response.content.lower()

    @pytest.mark.asyncio
    async def test_process_handles_service_exception_gracefully(self):
        """Agent returns error message when service raises exception."""
        mock_svc = AsyncMock()
        mock_svc.create_listing_from_dict = AsyncMock(side_effect=RuntimeError("DB down"))
        agent = self._make_agent(mock_svc)
        response = await agent.process(
            "sell 100 kg tomatoes at 25",
            context={"farmer_id": "f1"},
        )
        assert response.error is not None or "sorry" in response.content.lower()


class TestCropListingAgentExecute:
    """Tests for structured execute() interface."""

    @pytest.mark.asyncio
    async def test_execute_create_returns_success(self):
        """execute with action=create returns success + listing data."""
        mock_svc = AsyncMock()
        mock_svc.create_listing_from_dict = AsyncMock(return_value={
            "id": "exec-id", "commodity": "Onion", "asking_price_per_kg": 18.0
        })
        agent = CropListingAgent(listing_service=mock_svc)
        result = await agent.execute({
            "action": "create",
            "farmer_id": "f1",
            "commodity": "Onion",
            "quantity_kg": 50.0,
        })
        assert result["success"] is True
        assert result["data"]["id"] == "exec-id"

    @pytest.mark.asyncio
    async def test_execute_cancel_returns_success(self):
        """execute with action=cancel calls cancel_listing."""
        mock_svc = AsyncMock()
        mock_svc.cancel_listing = AsyncMock(return_value=True)
        agent = CropListingAgent(listing_service=mock_svc)
        result = await agent.execute({"action": "cancel", "listing_id": "id-1"})
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_unknown_action_returns_error(self):
        """execute with unknown action returns success=False."""
        agent = CropListingAgent(listing_service=AsyncMock())
        result = await agent.execute({"action": "explode"})
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_execute_without_service_returns_error(self):
        """execute returns error when listing_service is None."""
        agent = CropListingAgent(listing_service=None)
        result = await agent.execute({"action": "create"})
        assert result["success"] is False
        assert "ListingService not configured" in result["error"]


# ═══════════════════════════════════════════════════════════════
# AC6: Voice Agent create_listing creates real DB record
# ═══════════════════════════════════════════════════════════════

class TestVoiceListingIntegration:
    """
    AC6: Voice agent create_listing intent creates real DB record.

    Tests that CropListingAgent (used by VoiceAgent) correctly
    delegates to ListingService which calls DB.create_listing.
    """

    @pytest.mark.asyncio
    async def test_voice_create_listing_calls_db_create(self):
        """End-to-end: voice query → agent → service → DB.create_listing called."""
        # * Arrange: mock DB
        mock_db = AsyncMock()
        mock_db.create_listing = AsyncMock(return_value="voice-listing-123")

        svc = ListingService(db=mock_db)
        agent = CropListingAgent(listing_service=svc)

        # * Act: simulate voice query
        response = await agent.process(
            "mujhe 150 kg tamatar bechna hai Rs 22 per kg mein",
            context={"farmer_id": "farmer-voice-001"},
        )

        # * Assert: DB was called with correct commodity
        mock_db.create_listing.assert_called_once()
        call_data = mock_db.create_listing.call_args[0][0]
        assert call_data["commodity"] == "Tomato"
        assert call_data["quantity_kg"] == 150.0
        assert call_data["farmer_id"] == "farmer-voice-001"

    @pytest.mark.asyncio
    async def test_voice_create_listing_response_contains_listing_id(self):
        """Voice response confirms listing creation with the ID."""
        mock_db = AsyncMock()
        mock_db.create_listing = AsyncMock(return_value="voice-listing-XYZ")

        svc = ListingService(db=mock_db)
        agent = CropListingAgent(listing_service=svc)

        response = await agent.process(
            "sell 100 kg tomatoes at 25",
            context={"farmer_id": "farmer-001"},
        )

        assert "voice-listing-XYZ" in response.content


# ═══════════════════════════════════════════════════════════════
# Factory Test
# ═══════════════════════════════════════════════════════════════

def test_get_listing_service_factory():
    """get_listing_service returns a ListingService with injected deps."""
    mock_db = MagicMock()
    svc = get_listing_service(db=mock_db)
    assert isinstance(svc, ListingService)
    assert svc.db is mock_db

def test_grade_order_constants():
    """GRADE_ORDER maps A+ > A > B > C > Unverified."""
    assert GRADE_ORDER["A+"] > GRADE_ORDER["A"] > GRADE_ORDER["B"] > GRADE_ORDER["C"]
    assert GRADE_ORDER["C"] > GRADE_ORDER["Unverified"]
