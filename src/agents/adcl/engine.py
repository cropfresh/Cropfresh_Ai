"""
ADCL Agent — Engine / Orchestrator
=====================================
ADCLAgent orchestrates the weekly Adaptive Demand Crop List report:

  1. Fetch last 90 days of buyer orders (DB or mock)
  2. Aggregate demand by commodity (demand.py)
  3. Get price forecasts per commodity (price_agent or fallback)
  4. Apply seasonal fit + sow-season + green-label scoring (scoring.py)
  5. Generate multi-language summaries (summary.py)
  6. Persist report to DB (adcl_reports table) if DB available
  7. Return WeeklyReport

All external dependencies (db, price_agent, llm) are constructor-injected
and optional — the agent runs standalone for testing.
"""

# * ADCL ENGINE MODULE
# NOTE: Fully async; DB + price_agent + LLM are optional injections.

from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Any, Optional

from loguru import logger

from src.agents.adcl.demand import aggregate_demand
from src.agents.adcl.models import ADCLCrop, WeeklyReport
from src.agents.adcl.scoring import score_and_label
from src.agents.adcl.summary import SummaryGenerator


class ADCLAgent:
    """
    Adaptive Demand Crop List Agent.

    Generates weekly market intelligence for farmers:
    - Which crops to sow now (green-label based on sowing season)
    - Demand trends from buyer orders
    - Price forecasts
    - Seasonal fit (both harvest and sowing)
    - Multi-language summaries (en, hi, kn)

    All dependencies optional via constructor injection.
    """

    def __init__(
        self,
        db: Any | None = None,
        price_agent: Any | None = None,
        llm: Any | None = None,
    ) -> None:
        """
        Args:
            db          : Async DB client with get_recent_orders() and
                          insert_adcl_report() methods. None → uses mock data.
            price_agent : Agent with async predict(commodity) -> float method.
                          None → predicted_price_per_kg defaults to 0.0.
            llm         : LLM provider with async generate(messages) -> str.
                          None → template-based summaries are used.
        """
        self._db = db
        self._price_agent = price_agent
        self._summary_gen = SummaryGenerator(llm=llm)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_weekly_report(
        self,
        district: str = "Bangalore",
    ) -> WeeklyReport:
        """
        Generate the weekly ADCL report.

        Args:
            district : District name (for future geo-filtering). Not used
                       in filtering yet but stored for audit.

        Returns:
            WeeklyReport with ranked crops, green labels, and summaries.
        """
        logger.info("ADCL: generating weekly report for district={}", district)

        # Step 1 — Fetch orders
        orders = await self._get_orders()
        logger.debug("ADCL: fetched {} orders", len(orders))

        # Step 2 — Aggregate demand
        demand_records = aggregate_demand(orders)
        if not demand_records:
            logger.warning("ADCL: no demand records — returning empty report")
            return self._empty_report()

        # Step 3 — Price forecasts
        price_forecasts = await self._get_price_forecasts(demand_records)

        # Step 4 — Score + green-label (now uses demand_trend + sow_season_fit)
        current_month = datetime.now().month
        crops: list[ADCLCrop] = score_and_label(
            demand_records, price_forecasts, current_month
        )
        logger.info("ADCL: scored {} crops, {} green-labelled",
                    len(crops), sum(1 for c in crops if c.green_label))

        # Step 5 — Summaries
        summaries = await self._summary_gen.generate_async(crops)

        # Step 6 — Build report
        today = date.today()
        # Align to start of current week (Monday)
        week_start = today  # simplified; production would iso-week align

        report = WeeklyReport(
            week_start=week_start,
            crops=crops,
            summary_en=summaries.get("en", ""),
            summary_hi=summaries.get("hi", ""),
            summary_kn=summaries.get("kn", ""),
        )

        # Step 7 — Persist (best-effort)
        await self._persist(report)

        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_orders(self) -> list[dict[str, Any]]:
        """Fetch orders from DB or fall back to mock data."""
        if self._db is not None:
            try:
                return await self._db.get_recent_orders(days=90)
            except Exception as exc:
                logger.warning("ADCL: DB fetch failed ({}), using mock data", exc)

        return self._get_mock_orders()

    async def _get_price_forecasts(
        self,
        demand_records: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Fetch price forecasts for each commodity."""
        if self._price_agent is None:
            return {}

        forecasts: dict[str, float] = {}
        for rec in demand_records:
            commodity = rec["commodity"]
            try:
                price = await self._price_agent.predict(commodity)
                forecasts[commodity] = float(price)
            except Exception as exc:
                logger.debug("ADCL: price forecast failed for {}: {}", commodity, exc)
                forecasts[commodity] = 0.0

        return forecasts

    async def _persist(self, report: WeeklyReport) -> None:
        """Persist report to DB if available (best-effort)."""
        if self._db is None:
            return
        try:
            await self._db.insert_adcl_report(report.to_dict())
            logger.info("ADCL: report persisted to adcl_reports table")
        except Exception as exc:
            logger.warning("ADCL: persist failed: {}", exc)

    def _empty_report(self) -> WeeklyReport:
        """Return an empty report when no data is available."""
        return WeeklyReport(
            week_start=date.today(),
            crops=[],
            summary_en="No order data available for this week.",
            summary_hi="इस सप्ताह कोई ऑर्डर डेटा उपलब्ध नहीं है।",
            summary_kn="ಈ ವಾರ ಯಾವುದೇ ಆದೇಶ ಡೇಟಾ ಲಭ್ಯವಿಲ್ಲ.",
        )

    def _get_mock_orders(self) -> list[dict[str, Any]]:
        """
        Realistic mock order data for Bangalore district (last 90 days).

        Used when DB is not connected (testing / development).
        Includes 8 commodities with varied buyer counts and volumes.
        """
        today = date.today()

        def _days_ago(n: int) -> str:
            from datetime import timedelta
            return (today - timedelta(days=n)).isoformat()

        return [
            # Tomato — high volume, multiple buyers, rising demand
            {"commodity": "tomato", "quantity_kg": 500.0, "buyer_id": "b1", "created_at": _days_ago(5)},
            {"commodity": "tomato", "quantity_kg": 420.0, "buyer_id": "b2", "created_at": _days_ago(12)},
            {"commodity": "tomato", "quantity_kg": 380.0, "buyer_id": "b3", "created_at": _days_ago(20)},
            {"commodity": "tomato", "quantity_kg": 310.0, "buyer_id": "b4", "created_at": _days_ago(35)},
            {"commodity": "tomato", "quantity_kg": 280.0, "buyer_id": "b1", "created_at": _days_ago(60)},
            {"commodity": "tomato", "quantity_kg": 200.0, "buyer_id": "b5", "created_at": _days_ago(80)},
            # Onion — stable demand
            {"commodity": "onion", "quantity_kg": 600.0, "buyer_id": "b2", "created_at": _days_ago(8)},
            {"commodity": "onion", "quantity_kg": 550.0, "buyer_id": "b3", "created_at": _days_ago(25)},
            {"commodity": "onion", "quantity_kg": 580.0, "buyer_id": "b6", "created_at": _days_ago(55)},
            {"commodity": "onion", "quantity_kg": 520.0, "buyer_id": "b7", "created_at": _days_ago(75)},
            # Potato — moderate
            {"commodity": "potato", "quantity_kg": 400.0, "buyer_id": "b1", "created_at": _days_ago(10)},
            {"commodity": "potato", "quantity_kg": 350.0, "buyer_id": "b4", "created_at": _days_ago(40)},
            {"commodity": "potato", "quantity_kg": 300.0, "buyer_id": "b8", "created_at": _days_ago(70)},
            # Capsicum — low demand, falling
            {"commodity": "capsicum", "quantity_kg": 150.0, "buyer_id": "b2", "created_at": _days_ago(6)},
            {"commodity": "capsicum", "quantity_kg": 200.0, "buyer_id": "b9", "created_at": _days_ago(45)},
            {"commodity": "capsicum", "quantity_kg": 250.0, "buyer_id": "b3", "created_at": _days_ago(80)},
            # Okra — seasonal mismatch (off-season in March)
            {"commodity": "okra", "quantity_kg": 100.0, "buyer_id": "b5", "created_at": _days_ago(15)},
            {"commodity": "okra", "quantity_kg": 90.0,  "buyer_id": "b6", "created_at": _days_ago(50)},
            # Cabbage — rising
            {"commodity": "cabbage", "quantity_kg": 320.0, "buyer_id": "b4", "created_at": _days_ago(3)},
            {"commodity": "cabbage", "quantity_kg": 280.0, "buyer_id": "b7", "created_at": _days_ago(22)},
            {"commodity": "cabbage", "quantity_kg": 180.0, "buyer_id": "b2", "created_at": _days_ago(60)},
            # Beans
            {"commodity": "beans", "quantity_kg": 220.0, "buyer_id": "b1", "created_at": _days_ago(7)},
            {"commodity": "beans", "quantity_kg": 190.0, "buyer_id": "b8", "created_at": _days_ago(30)},
            {"commodity": "beans", "quantity_kg": 170.0, "buyer_id": "b3", "created_at": _days_ago(65)},
            # Brinjal
            {"commodity": "brinjal", "quantity_kg": 180.0, "buyer_id": "b6", "created_at": _days_ago(18)},
            {"commodity": "brinjal", "quantity_kg": 160.0, "buyer_id": "b9", "created_at": _days_ago(55)},
        ]


def get_adcl_agent(
    db: Any | None = None,
    price_agent: Any | None = None,
    llm: Any | None = None,
) -> ADCLAgent:
    """Factory for ADCLAgent."""
    return ADCLAgent(db=db, price_agent=price_agent, llm=llm)
