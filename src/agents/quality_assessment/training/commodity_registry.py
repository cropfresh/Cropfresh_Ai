"""Stable commodity registry for quality-grading datasets and models."""

from __future__ import annotations

from dataclasses import dataclass

UNKNOWN_COMMODITY = "unknown"
UNKNOWN_COMMODITY_ID = 0
LAUNCH_COHORT: tuple[str, ...] = ("tomato", "onion", "potato")
_COMMODITY_ORDER: tuple[str, ...] = (
    "tomato",
    "onion",
    "potato",
    "beans",
    "mango",
    "brinjal",
    "okra",
    "cabbage",
    "cauliflower",
    "capsicum",
    "chili",
    "cucumber",
    "peas",
    "carrot",
    "beetroot",
)


@dataclass(frozen=True, slots=True)
class CommoditySpec:
    """Stable commodity identifier exposed to training and inference code."""

    slug: str
    commodity_id: int
    launch_enabled: bool


COMMODITY_SPECS: dict[str, CommoditySpec] = {
    slug: CommoditySpec(
        slug=slug,
        commodity_id=index,
        launch_enabled=slug in LAUNCH_COHORT,
    )
    for index, slug in enumerate(_COMMODITY_ORDER, start=1)
}


def normalize_commodity(value: str | None) -> str:
    """Map free-form commodity names to a stable lowercase slug."""
    if not value:
        return UNKNOWN_COMMODITY
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def get_commodity_id(value: str | None) -> int:
    """Resolve a commodity name to a stable integer id."""
    spec = COMMODITY_SPECS.get(normalize_commodity(value))
    return spec.commodity_id if spec else UNKNOWN_COMMODITY_ID


def get_commodity_slug(commodity_id: int) -> str:
    """Resolve a commodity id back to its slug."""
    for spec in COMMODITY_SPECS.values():
        if spec.commodity_id == commodity_id:
            return spec.slug
    return UNKNOWN_COMMODITY


def launch_cohort_defaults() -> tuple[str, ...]:
    """Return the initial production cohort for grading rollout."""
    return LAUNCH_COHORT
