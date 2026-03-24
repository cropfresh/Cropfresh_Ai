"""
Crop Listing Agent
==================
Natural language interface for produce listing management.

Parses farmer voice/text input into structured listing operations,
delegates business logic to ListingService, and returns
formatted voice-friendly responses.

Supported intents:
- create_listing  → parse commodity, qty, price → create DB record
- my_listings     → fetch farmer's active listings
- cancel_listing  → soft-delete a listing by ID or commodity
- update_price    → update asking price on an active listing
"""

# * CROP LISTING AGENT MODULE
# NOTE: This agent wraps ListingService in a BaseAgent-compatible interface

import re
from typing import Any, Optional

from loguru import logger

from src.agents.base_agent import AgentConfig, AgentResponse, BaseAgent
from src.memory.state_manager import AgentExecutionState

# * Known commodity aliases for entity extraction
COMMODITY_ALIASES: dict[str, str] = {
    "tomato": "Tomato", "tamatar": "Tomato", "tomatoes": "Tomato",
    "onion": "Onion", "pyaaz": "Onion", "eerulli": "Onion",
    "potato": "Potato", "aloo": "Potato", "alugedde": "Potato",
    "beans": "Beans", "sabz": "Beans", "hurali": "Beans",
    "okra": "Okra", "bhindi": "Okra", "bendekai": "Okra",
    "carrot": "Carrot", "gajar": "Carrot", "gajjari": "Carrot",
    "cauliflower": "Cauliflower", "gobhi": "Cauliflower",
    "cucumber": "Cucumber", "kheera": "Cucumber",
    "chilli": "Chilli", "mirchi": "Chilli", "menasinakai": "Chilli",
}


class CropListingAgent(BaseAgent):
    """
    Agent for creating and managing produce listings via natural language.

    Accepts text queries like:
      "I want to sell 200 kg tomatoes at ₹25 per kg"
      "Show my listings"
      "Cancel my onion listing"

    Delegates all persistence to ListingService.
    """

    def __init__(
        self,
        listing_service: Optional[Any] = None,
        llm: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        config = AgentConfig(
            name="crop_listing",
            description="Creates and manages produce listings from natural language input",
        )
        super().__init__(config=config, llm=llm, **kwargs)
        self.listing_service = listing_service

    # ─────────────────────────────────────────────────────────
    # BaseAgent contract
    # ─────────────────────────────────────────────────────────

    async def process(
        self,
        query: str,
        context: Optional[dict] = None,
        execution: Optional[AgentExecutionState] = None,
    ) -> AgentResponse:
        """
        Parse natural language and dispatch to listing operations.

        Args:
            query: Farmer's voice/text input.
            context: Optional dict; may include 'farmer_id', 'session_id'.
            execution: Optional execution state tracker.

        Returns:
            AgentResponse with listing result or error message.
        """
        ctx = context or {}
        query_lower = query.lower()

        try:
            # * Check cancel/update BEFORE "my listing" to avoid prefix collision
            if any(k in query_lower for k in ("cancel", "withdraw", "remove", "delete")):
                return await self._handle_cancel(query, ctx)

            if any(k in query_lower for k in ("update price", "change price", "new price", "price change")):
                return await self._handle_update_price(query, ctx)

            if any(k in query_lower for k in ("my listing", "show listing", "mere listing", "nanna listing")):
                return await self._handle_my_listings(ctx)

            # * Default: attempt listing creation from query text
            return await self._handle_create(query, ctx)

        except Exception as exc:
            logger.error(f"CropListingAgent.process error: {exc}")
            return AgentResponse(
                content="Sorry, I couldn't process your listing request. Please try again.",
                agent_name=self.name,
                confidence=0.0,
                error=str(exc),
            )

    def _get_system_prompt(self, context: Optional[dict] = None) -> str:
        return (
            "You are a helpful farming assistant. "
            "Help farmers create and manage produce listings clearly and accurately."
        )

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Structured API interface for orchestrators.

        Args:
            input_data: Dict with 'action' and relevant fields.
                Actions: 'create', 'search', 'get', 'cancel', 'update_price'

        Returns:
            Dict with 'success', 'data', and optional 'error'.
        """
        action = input_data.get("action", "create")

        if not self.listing_service:
            return {"success": False, "error": "ListingService not configured"}

        try:
            if action == "create":
                result = await self.listing_service.create_listing_from_dict(input_data)
                return {"success": True, "data": result}

            if action == "search":
                result = await self.listing_service.search_listings(input_data)
                return {"success": True, "data": result}

            if action == "get":
                result = await self.listing_service.get_listing(input_data["listing_id"])
                return {"success": bool(result), "data": result}

            if action == "cancel":
                ok = await self.listing_service.cancel_listing(input_data["listing_id"])
                return {"success": ok}

            if action == "update_price":
                result = await self.listing_service.update_listing(
                    input_data["listing_id"],
                    {"asking_price_per_kg": input_data["new_price"]},
                )
                return {"success": bool(result), "data": result}

            return {"success": False, "error": f"Unknown action: {action}"}

        except Exception as exc:
            logger.error(f"CropListingAgent.execute error: {exc}")
            return {"success": False, "error": str(exc)}

    # ─────────────────────────────────────────────────────────
    # Private handlers
    # ─────────────────────────────────────────────────────────

    async def _handle_create(self, query: str, ctx: dict) -> AgentResponse:
        """Parse query and create a new listing."""
        commodity = self._extract_commodity(query)
        quantity = self._extract_quantity(query)
        price = self._extract_price(query)
        farmer_id = ctx.get("farmer_id", "")

        if not commodity:
            return AgentResponse(
                content="Which crop would you like to list? Please mention the crop name.",
                agent_name=self.name,
                confidence=0.5,
            )

        if not quantity:
            return AgentResponse(
                content=f"How many kg of {commodity} would you like to sell?",
                agent_name=self.name,
                confidence=0.5,
            )

        if not farmer_id:
            return AgentResponse(
                content="Listing created in preview mode (no farmer ID in context).",
                agent_name=self.name,
                confidence=0.6,
                steps=[f"commodity={commodity}", f"qty={quantity}kg", f"price={price}"],
            )

        if not self.listing_service:
            return AgentResponse(
                content=(
                    f"Listing preview: {quantity}kg {commodity}"
                    + (f" at ₹{price}/kg" if price else " (price will be auto-suggested)")
                ),
                agent_name=self.name,
                confidence=0.7,
                steps=["listing_service_unavailable"],
            )

        listing_data = {
            "farmer_id": farmer_id,
            "commodity": commodity,
            "quantity_kg": quantity,
            "asking_price_per_kg": price,
        }
        result = await self.listing_service.create_listing_from_dict(listing_data)
        listing_id = result.get("id", "N/A")
        actual_price = result.get("asking_price_per_kg", price)
        suggested = result.get("suggested_price")

        price_text = f"₹{actual_price}/kg"
        if suggested and not price:
            price_text = f"₹{actual_price}/kg (auto-suggested)"

        return AgentResponse(
            content=(
                f"Your listing has been created! "
                f"{quantity}kg {commodity} at {price_text}. "
                f"Listing ID: {listing_id}"
            ),
            agent_name=self.name,
            confidence=0.95,
            steps=[f"listing_id={listing_id}", f"price={actual_price}"],
        )

    async def _handle_my_listings(self, ctx: dict) -> AgentResponse:
        """Return farmer's active listings summary."""
        farmer_id = ctx.get("farmer_id", "")
        if not farmer_id or not self.listing_service:
            return AgentResponse(
                content="You have no active listings currently.",
                agent_name=self.name,
                confidence=0.6,
            )
        listings = await self.listing_service.get_farmer_listings(farmer_id)
        if not listings:
            return AgentResponse(
                content="You have no active listings currently.",
                agent_name=self.name,
                confidence=0.9,
            )
        lines = [
            f"• {listing['commodity']} — {listing['quantity_kg']}kg @ ₹{listing['asking_price_per_kg']}/kg ({listing['grade']})"
            for listing in listings[:5]
        ]
        return AgentResponse(
            content="Your active listings:\n" + "\n".join(lines),
            agent_name=self.name,
            confidence=0.9,
        )

    async def _handle_cancel(self, query: str, ctx: dict) -> AgentResponse:
        """Cancel a listing by ID or commodity mention."""
        listing_id = ctx.get("listing_id")
        if not listing_id or not self.listing_service:
            return AgentResponse(
                content="Please provide the listing ID to cancel.",
                agent_name=self.name,
                confidence=0.5,
            )
        ok = await self.listing_service.cancel_listing(listing_id)
        if ok:
            return AgentResponse(
                content=f"Listing {listing_id} has been cancelled.",
                agent_name=self.name,
                confidence=0.95,
            )
        return AgentResponse(
            content="Could not find that listing to cancel.",
            agent_name=self.name,
            confidence=0.6,
        )

    async def _handle_update_price(self, query: str, ctx: dict) -> AgentResponse:
        """Update price on an existing listing."""
        new_price = self._extract_price(query)
        listing_id = ctx.get("listing_id")
        if not listing_id or not new_price or not self.listing_service:
            return AgentResponse(
                content="Please provide the listing ID and new price to update.",
                agent_name=self.name,
                confidence=0.5,
            )
        result = await self.listing_service.update_listing(
            listing_id, {"asking_price_per_kg": new_price}
        )
        if result:
            return AgentResponse(
                content=f"Price updated to ₹{new_price}/kg for listing {listing_id}.",
                agent_name=self.name,
                confidence=0.95,
            )
        return AgentResponse(
            content="Could not update the listing price.",
            agent_name=self.name,
            confidence=0.6,
        )

    # ─────────────────────────────────────────────────────────
    # Entity extraction helpers
    # ─────────────────────────────────────────────────────────

    def _extract_commodity(self, text: str) -> Optional[str]:
        """Extract crop name from query text."""
        text_lower = text.lower()
        for alias, canonical in COMMODITY_ALIASES.items():
            if alias in text_lower:
                return canonical
        return None

    def _extract_quantity(self, text: str) -> Optional[float]:
        """Extract kg quantity from query text."""
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:kg|kgs|kilo|kilogram)", text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:quintal|quintals|q)", text, re.IGNORECASE)
        if match:
            return float(match.group(1)) * 100.0
        return None

    def _extract_price(self, text: str) -> Optional[float]:
        """Extract price per kg from query text."""
        match = re.search(
            r"(?:rs\.?|₹|inr|price|at)\s*(\d+(?:\.\d+)?)\s*(?:/kg|per kg)?",
            text, re.IGNORECASE
        )
        if match:
            return float(match.group(1))
        return None
