"""Repository adapter for ADCL persistence and lookup."""

from __future__ import annotations

from datetime import date
from typing import Any

from src.agents.adcl.models import WeeklyReport


class ADCLRepository:
    """Wraps the DB client methods needed by the ADCL service."""

    def __init__(self, db: Any | None = None) -> None:
        self._db = db

    async def get_recent_orders(
        self,
        district: str,
        days: int = 90,
    ) -> list[dict[str, Any]]:
        if self._db and hasattr(self._db, "get_recent_orders"):
            return await self._db.get_recent_orders(district=district, days=days)
        return []

    async def get_price_history(
        self,
        commodity: str,
        district: str,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        if self._db and hasattr(self._db, "get_price_history"):
            return await self._db.get_price_history(
                commodity=commodity,
                district=district,
                days=days,
            )
        return []

    async def get_latest_report(
        self,
        district: str,
        week_start: date | None = None,
    ) -> WeeklyReport | None:
        if not (self._db and hasattr(self._db, "get_latest_adcl_report")):
            return None
        payload = await self._db.get_latest_adcl_report(
            district=district,
            week_start=week_start,
        )
        if not payload:
            return None
        return WeeklyReport.from_dict(payload)

    async def save_report(self, report: WeeklyReport) -> None:
        if self._db and hasattr(self._db, "insert_adcl_report"):
            await self._db.insert_adcl_report(report.to_dict())

    async def get_farmer_district(self, farmer_id: str) -> str | None:
        if not (self._db and hasattr(self._db, "get_farmer")):
            return None
        farmer = await self._db.get_farmer(farmer_id)
        if not farmer:
            return None
        district = farmer.get("district")
        return str(district) if district else None
