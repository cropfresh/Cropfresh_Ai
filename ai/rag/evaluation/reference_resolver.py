from __future__ import annotations

from datetime import datetime, timezone

from ai.rag.evaluation.models import GoldenEntry, ResolvedReference


class ReferenceResolver:
    """Resolve benchmark references for static and live entries."""

    async def resolve(self, entry: GoldenEntry) -> ResolvedReference:
        if entry.mode == "static":
            return ResolvedReference(
                ground_truth=entry.ground_truth,
                contexts=entry.contexts,
                freshness_ok=True,
                metadata={"mode": "static"},
            )

        if entry.reference_resolver == "agmarknet_price":
            return await self._resolve_agmarknet_price(entry)
        if entry.reference_resolver == "agmarknet_price_compare":
            return await self._resolve_agmarknet_compare(entry)

        raise ValueError(f"Unsupported reference resolver: {entry.reference_resolver}")

    async def _resolve_agmarknet_price(self, entry: GoldenEntry) -> ResolvedReference:
        from src.tools.agmarknet import AgmarknetTool

        params = entry.resolver_params
        tool = AgmarknetTool()
        prices = await tool.get_prices(
            commodity=str(params.get("commodity", "Tomato")).title(),
            state=str(params.get("state", "Karnataka")),
            district=params.get("district"),
        )
        if not prices:
            prices = tool.get_mock_prices(
                commodity=str(params.get("commodity", "Tomato")).title(),
                district=str(params.get("district", "Kolar")),
            )
        latest = prices[0]
        freshness_ok = (datetime.now(timezone.utc) - latest.date.replace(tzinfo=timezone.utc)).days <= 7
        return ResolvedReference(
            ground_truth=(
                f"{latest.commodity} price in {latest.district} is approximately "
                f"Rs.{latest.modal_price:.0f}/quintal as of {latest.date.date()}."
            ),
            contexts=[
                (
                    f"{price.commodity} mandi price in {price.market}, {price.district}: "
                    f"modal Rs.{price.modal_price:.0f}/quintal on {price.date.date()}."
                )
                for price in prices[:3]
            ],
            freshness_ok=freshness_ok,
            metadata={"source": "agmarknet", "district": latest.district},
        )

    async def _resolve_agmarknet_compare(self, entry: GoldenEntry) -> ResolvedReference:
        from src.tools.agmarknet import AgmarknetTool

        params = entry.resolver_params
        commodity = str(params.get("commodity", "Onion")).title()
        districts = [str(district) for district in params.get("districts", [])]
        tool = AgmarknetTool()
        market_lines: list[str] = []
        contexts: list[str] = []
        freshness_flags: list[bool] = []
        for district in districts:
            prices = await tool.get_prices(commodity=commodity, state="Karnataka", district=district)
            if not prices:
                prices = tool.get_mock_prices(commodity=commodity, district=district)
            price = prices[0]
            freshness_flags.append(
                (datetime.now(timezone.utc) - price.date.replace(tzinfo=timezone.utc)).days <= 7
            )
            market_lines.append(f"{district} Rs.{price.modal_price:.0f}/quintal")
            contexts.append(
                f"{commodity} mandi price in {price.market}, {district}: modal Rs.{price.modal_price:.0f}/quintal."
            )
        return ResolvedReference(
            ground_truth=f"{commodity} prices: " + ", ".join(market_lines) + ".",
            contexts=contexts,
            freshness_ok=all(freshness_flags),
            metadata={"source": "agmarknet", "districts": districts},
        )
