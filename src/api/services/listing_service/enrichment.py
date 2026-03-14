"""
Listing Enrichment Mixin
========================
Handles listing augmentations: shelf life, ADCL demand tags, AI pricing, and AI quality checks.
"""

from typing import Any, Optional

from loguru import logger

from .constants import SHELF_LIFE_DAYS


class ListingEnrichmentMixin:
    """Mixin handling external ML agent integrations for listings."""

    pricing_agent: Optional[Any]
    quality_agent: Optional[Any]
    adcl_agent: Optional[Any]

    async def _suggest_price(self, commodity: str) -> Optional[float]:
        """Fetch price recommendation from pricing/prediction agent."""
        if not self.pricing_agent:
            return None
        try:
            if hasattr(self.pricing_agent, "predict"):
                prediction = await self.pricing_agent.predict(commodity=commodity)
                return round(prediction.current_price / 100, 2)   # quintal → kg
            if hasattr(self.pricing_agent, "get_recommendation"):
                rec = await self.pricing_agent.get_recommendation(commodity=commodity)
                return rec.get("recommended_price_per_kg")
        except Exception as exc:
            logger.warning(f"Price suggestion failed for {commodity}: {exc}")
        return None

    async def _check_adcl_tag(self, commodity: str) -> bool:
        """Check if commodity appears in current ADCL weekly demand list."""
        if not self.adcl_agent:
            return False
        try:
            result = await self.adcl_agent.get_weekly_demand()
            crops = result.get("crops", [])
            return any(
                c.get("crop", "").lower() == commodity.lower()
                for c in crops
            )
        except Exception as exc:
            logger.warning(f"ADCL tag check failed: {exc}")
        return False

    async def _trigger_quality_assessment(
        self, listing_data: dict, photos: list[str]
    ) -> bool:
        """
        Trigger quality assessment for a listing with photos.

        Returns:
            hitl_required flag from assessment result.
        """
        if not self.quality_agent:
            return True     # Default to HITL when no agent available

        try:
            result = await self.quality_agent.assess(
                photos=photos,
                commodity=listing_data.get("commodity", ""),
            )
            return result.get("hitl_required", True)
        except Exception as exc:
            logger.warning(f"Quality assessment trigger failed: {exc}")
            return True

    @staticmethod
    def _get_shelf_life(commodity: str) -> int:
        """Return shelf life in days for a commodity (case-insensitive)."""
        return SHELF_LIFE_DAYS.get(commodity.lower(), SHELF_LIFE_DAYS["default"])
