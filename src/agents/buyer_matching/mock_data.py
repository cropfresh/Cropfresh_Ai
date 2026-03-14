"""
Buyer Matching Mock Data
========================
Provides synthetic fallback data for demonstration or isolated testing.
"""

from datetime import datetime, timedelta

from .models import BuyerProfile, ListingProfile


class BuyerMatchingMockDataMixin:
    """Mixin for generating synthetic listings and buyers."""

    def _get_mock_listing_and_buyers(self, listing_id: str) -> tuple[ListingProfile, list[BuyerProfile]]:
        listing = ListingProfile(
            listing_id=listing_id,
            farmer_id="farmer-test-001",
            commodity="Tomato",
            variety="Hybrid",
            quantity_kg=200,
            asking_price_per_kg=24.0,
            grade="A",
            pickup_lat=13.13,
            pickup_lon=78.15,
            district="Kolar",
            reliability_score=0.86,
        )
        buyers = [
            BuyerProfile(
                buyer_id="buyer-near",
                name="Kolar Retail Hub",
                type="retailer",
                district="Kolar",
                delivery_lat=13.14,
                delivery_lon=78.16,
                preferred_grades=["A", "B"],
                max_price_per_kg=30.0,
                demand_commodities=["Tomato", "Onion"],
                demand_quantity_kg=300,
                order_history=[
                    {"commodity": "Tomato", "date": datetime.now().isoformat()},
                    {"commodity": "Tomato", "date": (datetime.now() - timedelta(days=14)).isoformat()},
                ],
            ),
            BuyerProfile(
                buyer_id="buyer-mid",
                name="Bangalore Fresh Stores",
                type="wholesaler",
                district="Bangalore",
                delivery_lat=12.98,
                delivery_lon=77.60,
                preferred_grades=["A"],
                max_price_per_kg=25.0,
                demand_commodities=["Tomato"],
                demand_quantity_kg=120,
                order_history=[{"commodity": "Tomato", "date": (datetime.now() - timedelta(days=21)).isoformat()}],
            ),
        ]
        return listing, buyers

    def _get_mock_listings_for_commodity(self, commodity: str) -> list[ListingProfile]:
        return [
            ListingProfile(
                listing_id="listing-a",
                farmer_id="farmer-a",
                commodity=commodity,
                variety="Hybrid",
                quantity_kg=180,
                asking_price_per_kg=23.0,
                grade="A",
                pickup_lat=13.13,
                pickup_lon=78.15,
                district="Kolar",
                reliability_score=0.9,
            ),
            ListingProfile(
                listing_id="listing-b",
                farmer_id="farmer-b",
                commodity=commodity,
                variety="Local",
                quantity_kg=240,
                asking_price_per_kg=27.0,
                grade="B",
                pickup_lat=13.05,
                pickup_lon=77.95,
                district="Bangalore Rural",
                reliability_score=0.75,
            ),
        ]
