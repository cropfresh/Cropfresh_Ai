"""Pending-access sources that are tracked but not executed."""

from __future__ import annotations

from src.rates.enums import RateKind
from src.rates.models import PendingSource

PENDING_SOURCES: list[PendingSource] = [
    PendingSource(
        source="enam_official_api",
        rate_kind=RateKind.MANDI_WHOLESALE,
        reason="Requires official registration and credentials",
        source_url="https://enam.gov.in/",
    ),
    PendingSource(
        source="agriwatch",
        rate_kind=RateKind.MANDI_WHOLESALE,
        reason="Paid subscription required",
        source_url="https://oldwebsite.agriwatch.in/spot-market-prices.php",
    ),
    PendingSource(
        source="ncdex",
        rate_kind=RateKind.MANDI_WHOLESALE,
        reason="Exchange feed requires a dedicated market-data agreement",
        source_url="https://www.ncdex.com/",
    ),
    PendingSource(
        source="kisan_suvidha",
        rate_kind=RateKind.MANDI_WHOLESALE,
        reason="Mobile app endpoint is not public",
        source_url="https://services.india.gov.in/service/detail/check-prices-and-arrivals-of-agricultural-commodities",
    ),
    PendingSource(
        source="mandi_bhav_app",
        rate_kind=RateKind.MANDI_WHOLESALE,
        reason="Mobile app endpoint is not public",
        source_url="https://play.google.com/store/apps/details?id=com.livertigo.mandibhav&hl=en_IN",
    ),
    PendingSource(
        source="napanta_app",
        rate_kind=RateKind.MANDI_WHOLESALE,
        reason="App feed is not publicly documented",
        source_url="https://www.napanta.com/",
    ),
    PendingSource(
        source="ksamb_legacy",
        rate_kind=RateKind.MANDI_WHOLESALE,
        reason="Legacy portal retained for reference only",
        source_url="https://ksamb.com/",
    ),
    PendingSource(
        source="maratavahini_legacy",
        rate_kind=RateKind.MANDI_WHOLESALE,
        reason="Legacy portal retained for reference only",
        source_url="https://maratavahini.kar.nic.in/",
    ),
]
