"""Pydantic models for the multi-source rate hub."""

from __future__ import annotations

from datetime import date as date_type
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from src.rates.enums import AuthorityTier, ComparisonDepth, FetchMode, RateKind


class RateQuery(BaseModel):
    """Normalized rate query shared across API, tools, and scheduler."""

    rate_kinds: list[RateKind]
    commodity: Optional[str] = None
    state: str = "Karnataka"
    district: Optional[str] = None
    market: Optional[str] = None
    date: Optional[date_type] = None
    include_reference: bool = True
    force_live: bool = False
    comparison_depth: ComparisonDepth = ComparisonDepth.ALL_SOURCES

    @model_validator(mode="after")
    def validate_query(self) -> "RateQuery":
        requires_commodity = {
            RateKind.MANDI_WHOLESALE,
            RateKind.RETAIL_PRODUCE,
            RateKind.SUPPORT_PRICE,
        }
        if any(kind in requires_commodity for kind in self.rate_kinds) and not self.commodity:
            raise ValueError("commodity is required for mandi, retail, and support price queries")
        return self

    @property
    def target_date(self) -> date_type:
        return self.date or date_type.today()

    @property
    def location_label(self) -> str:
        if self.market:
            return self.market
        if self.district:
            return self.district
        return self.state


class NormalizedRateRecord(BaseModel):
    """Canonical record stored in the multi-source repository."""

    rate_kind: RateKind
    commodity: Optional[str] = None
    variety: Optional[str] = None
    state: str = "Karnataka"
    district: Optional[str] = None
    market: Optional[str] = None
    location_label: str
    price_date: date_type
    unit: str
    currency: str = "INR"
    price_value: Optional[float] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    modal_price: Optional[float] = None
    source: str
    authority_tier: AuthorityTier
    source_url: str
    freshness: str = "live"
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    raw_record_id: Optional[str] = None


class SourceQuote(BaseModel):
    """Response-facing view of a source quote."""

    rate_kind: RateKind
    source: str
    authority_tier: AuthorityTier
    commodity: Optional[str] = None
    location_label: str
    price_date: date_type
    unit: str
    currency: str = "INR"
    price_value: Optional[float] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    modal_price: Optional[float] = None
    freshness: str = "live"
    source_url: str
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    warnings: list[str] = Field(default_factory=list)


class CanonicalRate(BaseModel):
    """Primary answer selected for a rate kind/location group."""

    rate_kind: RateKind
    source: str
    authority_tier: AuthorityTier
    commodity: Optional[str] = None
    location_label: str
    price_date: date_type
    unit: str
    currency: str = "INR"
    price_value: Optional[float] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    modal_price: Optional[float] = None
    comparison_count: int = 0
    freshness: str = "live"


class SourceHealthSnapshot(BaseModel):
    """Health and circuit status for one source connector."""

    source: str
    status: str = "healthy"
    supports_live: bool = True
    fetch_mode: FetchMode = FetchMode.LIVE
    last_successful_fetch: Optional[datetime] = None
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    circuit_open_until: Optional[datetime] = None


class PendingSource(BaseModel):
    """Metadata-only source that is known but not executed."""

    source: str
    rate_kind: RateKind
    reason: str
    source_url: str


class MultiSourceRateResult(BaseModel):
    """Full response for API and tool consumers."""

    query_target: dict[str, object]
    canonical_rates: list[CanonicalRate] = Field(default_factory=list)
    comparison_quotes: list[SourceQuote] = Field(default_factory=list)
    source_health: list[SourceHealthSnapshot] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    pending_sources: list[PendingSource] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
