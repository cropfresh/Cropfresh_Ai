"""Canonical ADCL service used by API, listings, voice, and wrappers."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from loguru import logger

from src.agents.adcl.demand import aggregate_demand
from src.agents.adcl.live_context import fetch_enam_context, fetch_imd_context
from src.agents.adcl.models import WeeklyReport
from src.agents.adcl.price_runtime import build_price_signals
from src.agents.adcl.report_utils import (
    attach_crop_context,
    base_source_health,
    build_empty_report,
    report_metadata,
)
from src.agents.adcl.repository import ADCLRepository
from src.agents.adcl.scoring import score_and_label
from src.agents.adcl.summary import SummaryGenerator
from src.agents.adcl.time_utils import utc_now, utc_now_iso
from src.production.observability import trace_agent


class ADCLService:
    """District-first ADCL service with persistence, evidence, and compatibility shims."""

    def __init__(
        self,
        db: Any | None = None,
        rate_service: Any | None = None,
        llm: Any | None = None,
        imd_client: Any | None = None,
        enam_client: Any | None = None,
        enable_enam: bool = False,
    ) -> None:
        self.repository = ADCLRepository(db=db)
        self.rate_service = rate_service
        self.summary_generator = SummaryGenerator(llm=llm)
        self.imd_client = imd_client
        self.enam_client = enam_client
        self.enable_enam = enable_enam

    def update_dependencies(
        self,
        db: Any | None = None,
        rate_service: Any | None = None,
        llm: Any | None = None,
        imd_client: Any | None = None,
        enam_client: Any | None = None,
        enable_enam: bool | None = None,
    ) -> None:
        """Allow app startup to upgrade the shared singleton with real services."""
        if db is not None:
            self.repository = ADCLRepository(db=db)
        if rate_service is not None:
            self.rate_service = rate_service
        if llm is not None:
            self.summary_generator = SummaryGenerator(llm=llm)
        if imd_client is not None:
            self.imd_client = imd_client
        if enam_client is not None:
            self.enam_client = enam_client
        if enable_enam is not None:
            self.enable_enam = enable_enam

    @trace_agent("adcl_service")
    async def generate_weekly_report(
        self,
        district: str,
        force_live: bool = False,
        farmer_id: str | None = None,
        language: str | None = None,
    ) -> WeeklyReport:
        """Generate or fetch the weekly district report using live dependencies only."""
        district_name = district.strip() or "Bangalore"
        week_start = _week_start_for(utc_now().date())
        if not force_live:
            cached = await self.repository.get_latest_report(district_name, week_start)
            if cached is not None:
                return cached

        orders = await self.repository.get_recent_orders(district=district_name, days=90)
        source_health = base_source_health(len(orders))
        base_metadata = report_metadata(force_live, farmer_id, language, 0)
        if not orders:
            report = build_empty_report(
                district=district_name,
                week_start=week_start,
                source_health=source_health,
                metadata=base_metadata,
            )
            await self.repository.save_report(report)
            return report

        demand_records = aggregate_demand(orders)
        commodities = [record["commodity"] for record in demand_records]
        price_signals, rate_health, price_freshness = await build_price_signals(
            repository=self.repository,
            rate_service=self.rate_service,
            district=district_name,
            commodities=commodities,
            force_live=force_live,
        )
        imd_context, imd_health, imd_freshness = await fetch_imd_context(
            self.imd_client,
            district_name,
            commodities,
        )
        enam_context, enam_health, enam_freshness = await fetch_enam_context(
            self.enam_client,
            district_name,
            commodities,
            enabled=self.enable_enam,
        )

        source_health["rate_hub"] = rate_health
        source_health["imd"] = imd_health
        source_health["enam"] = enam_health
        crops = score_and_label(
            demand_records,
            {name: signal["predicted_price_per_kg"] for name, signal in price_signals.items()},
            current_month=utc_now().month,
        )
        attach_crop_context(crops, price_signals, source_health, imd_context, enam_context)
        summaries = await self.summary_generator.generate_async(crops)
        green_count = sum(1 for crop in crops if crop.green_label)
        report = WeeklyReport(
            week_start=week_start,
            district=district_name,
            crops=crops,
            summary_en=summaries.get("en", ""),
            summary_hi=summaries.get("hi", ""),
            summary_kn=summaries.get("kn", ""),
            freshness={
                "generated_at": utc_now_iso(),
                "price": price_freshness,
                "imd": imd_freshness,
                "enam": enam_freshness,
            },
            source_health=source_health,
            metadata=report_metadata(
                force_live=force_live,
                farmer_id=farmer_id,
                language=language,
                crop_count=len(crops),
                green_count=green_count,
            ),
        )
        await self.repository.save_report(report)
        logger.info(
            "ADCL weekly report generated district={} crops={} green={} orders={} rate_status={} imd_status={} enam_status={}",
            district_name,
            len(crops),
            green_count,
            len(orders),
            rate_health.get("status", "disabled"),
            imd_health.get("status", "disabled"),
            enam_health.get("status", "gated"),
        )
        return report

    async def get_weekly_demand(
        self,
        district: str = "Bangalore",
        force_live: bool = False,
    ) -> dict[str, Any]:
        """Return the canonical report payload used by REST and compatibility callers."""
        report = await self.generate_weekly_report(district=district, force_live=force_live)
        return report.to_dict()

    async def is_recommended_crop(
        self,
        commodity: str,
        district: str,
        week_start: date | None = None,
    ) -> bool:
        """Return whether the commodity is green-labelled for the target district."""
        report = await self.repository.get_latest_report(district, week_start)
        if report is None:
            report = await self.generate_weekly_report(district=district, force_live=False)
        return any(
            crop.green_label and crop.commodity.lower() == commodity.lower()
            for crop in report.crops
        )

    async def get_weekly_list(self, location: str = "Bangalore") -> list[str]:
        """Compatibility shim for older voice callers expecting a crop-name list."""
        report = await self.generate_weekly_report(district=location, force_live=False)
        return [crop.commodity for crop in report.crops[:5]]


def _week_start_for(day: date) -> date:
    return day - timedelta(days=day.weekday())
