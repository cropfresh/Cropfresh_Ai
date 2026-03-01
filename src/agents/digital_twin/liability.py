"""
Digital Twin — Liability Matrix
================================
Determines the responsible party for produce quality disputes
using transit time, grade delta, and photo evidence.

CropFresh Liability Decision Tree
----------------------------------
Priority | Condition                              | Outcome
---------|----------------------------------------|---------------------------
  1      | No arrival photos submitted            | Buyer claim rejected (0%)
  2      | No quality degradation detected        | No liability (0%)
  3      | Quantity mismatch > 5%                 | Shared (farmer + hauler)
  4      | Grade drop + transit > 6h              | Hauler (cold-chain failure)
  5      | Grade drop + transit < 2h              | Farmer (pre-existing issue)
  6      | Grade drop + transit 2–6h              | Shared

Claim percentage is proportional to grade drop severity (20% per grade level)
plus a bonus for each new defect detected at arrival.
"""

# * LIABILITY MODULE
# NOTE: All thresholds are tunable constants — adjust based on pilot feedback.
# ! SECURITY: Liability recommendations are advisory. Final decisions require human review for amounts > ₹50,000.

from __future__ import annotations

from dataclasses import dataclass


# * ═══════════════════════════════════════════════════════════════
# * Tunable Thresholds
# * ═══════════════════════════════════════════════════════════════

# NOTE: Transit thresholds define the cold-chain responsibility window
LONG_TRANSIT_HOURS: float = 6.0      # Above this → hauler likely at fault
SHORT_TRANSIT_HOURS: float = 2.0     # Below this → pre-existing issue, farmer at fault

# NOTE: Quantity mismatch beyond this % triggers shared liability
QUANTITY_MISMATCH_THRESHOLD: float = 5.0

# * Per-grade-level claim rate (% of order value per grade drop)
CLAIM_RATE_PER_GRADE: float = 20.0

# * Additional claim % per new defect found at arrival
CLAIM_RATE_PER_DEFECT: float = 5.0

# * Maximum total claim percentage
MAX_CLAIM_PERCENT: float = 100.0

# * Reduction applied when farmer is blamed (short transit is ambiguous)
FARMER_CLAIM_REDUCTION: float = 0.70


# * ═══════════════════════════════════════════════════════════════
# * Result Type
# * ═══════════════════════════════════════════════════════════════

@dataclass
class LiabilityResult:
    """Output of the liability determination."""

    liable_party: str       # 'farmer' | 'hauler' | 'buyer' | 'shared' | 'none'
    claim_percent: float    # Recommended compensation % of order value [0, 100]
    reasoning: str          # Human-readable explanation for the recommendation


# * ═══════════════════════════════════════════════════════════════
# * Grade Helpers
# * ═══════════════════════════════════════════════════════════════

_GRADE_VALUES: dict[str, int] = {"A+": 4, "A": 3, "B": 2, "C": 1}


def _grade_drop_count(grade_departure: str, grade_arrival: str) -> int:
    """
    Return the number of grade levels dropped.

    Returns:
        Non-negative integer. 0 if no drop or improvement.
    """
    dep = _GRADE_VALUES.get(grade_departure, 2)
    arr = _GRADE_VALUES.get(grade_arrival, 2)
    return max(0, dep - arr)


def _compute_base_claim(grade_departure: str, grade_arrival: str, new_defects: list[str]) -> float:
    """
    Calculate base claim % from grade drop and new defect count.

    Args:
        grade_departure: Departure grade.
        grade_arrival:   Arrival grade.
        new_defects:     Defects new since departure.

    Returns:
        Base claim percentage [0, 100].
    """
    drop = _grade_drop_count(grade_departure, grade_arrival)
    grade_claim = drop * CLAIM_RATE_PER_GRADE
    defect_claim = min(CLAIM_RATE_PER_GRADE, len(new_defects) * CLAIM_RATE_PER_DEFECT)
    return min(MAX_CLAIM_PERCENT, grade_claim + defect_claim)


# * ═══════════════════════════════════════════════════════════════
# * Liability Matrix
# * ═══════════════════════════════════════════════════════════════

def determine_liability(
    grade_departure: str,
    grade_arrival: str,
    quality_delta: float,
    transit_hours: float,
    new_defects: list[str],
    has_arrival_photos: bool,
    quantity_mismatch_percent: float = 0.0,
) -> LiabilityResult:
    """
    Apply the CropFresh liability matrix to assign a responsible party.

    Args:
        grade_departure:           Departure grade (e.g. 'A').
        grade_arrival:             Arrival grade   (e.g. 'B').
        quality_delta:             Numeric quality delta from compute_grade_delta().
        transit_hours:             Hours between departure twin creation and arrival.
        new_defects:               Defects found at arrival not present at departure.
        has_arrival_photos:        Whether the buyer submitted arrival photos.
        quantity_mismatch_percent: Percentage quantity discrepancy (default 0).

    Returns:
        LiabilityResult with liable party, claim %, and reasoning string.
    """
    # * Rule 1 — No photos: claim rejected outright
    if not has_arrival_photos:
        return LiabilityResult(
            liable_party="none",
            claim_percent=0.0,
            reasoning=(
                "Claim rejected: buyer did not submit arrival photos. "
                "Photo evidence is mandatory to raise a quality dispute."
            ),
        )

    # * Rule 2 — Quantity mismatch > threshold: shared liability (before degradation check)
    # NOTE: Quantity discrepancy is independent of quality grade — checked first.
    if quantity_mismatch_percent > QUANTITY_MISMATCH_THRESHOLD:
        claim = min(MAX_CLAIM_PERCENT, quantity_mismatch_percent * 1.5)
        return LiabilityResult(
            liable_party="shared",
            claim_percent=round(claim, 1),
            reasoning=(
                f"Quantity mismatch of {quantity_mismatch_percent:.1f}% detected "
                f"(threshold: {QUANTITY_MISMATCH_THRESHOLD}%). "
                "Shared liability: farmer (weighment accuracy) "
                "and hauler (delivery verification)."
            ),
        )

    # * Rule 3 — No quality degradation: no liability
    no_grade_drop = quality_delta >= 0.0
    no_new_defects = not new_defects
    if no_grade_drop and no_new_defects:
        return LiabilityResult(
            liable_party="none",
            claim_percent=0.0,
            reasoning=(
                f"No quality degradation detected. "
                f"Departure grade {grade_departure} matches arrival grade {grade_arrival}."
            ),
        )

    base_claim = _compute_base_claim(grade_departure, grade_arrival, new_defects)

    # * Rule 4 — Grade drop + long transit: hauler at fault (cold-chain failure)
    if transit_hours > LONG_TRANSIT_HOURS:
        return LiabilityResult(
            liable_party="hauler",
            claim_percent=round(base_claim, 1),
            reasoning=(
                f"Grade dropped from {grade_departure} to {grade_arrival} "
                f"during {transit_hours:.1f}h transit "
                f"(threshold: >{LONG_TRANSIT_HOURS}h). "
                "Extended transit time indicates cold-chain failure — hauler is responsible."
            ),
        )

    # * Rule 5 — Grade drop + short transit: farmer at fault (pre-existing issue)
    if transit_hours < SHORT_TRANSIT_HOURS:
        farmer_claim = round(base_claim * FARMER_CLAIM_REDUCTION, 1)
        return LiabilityResult(
            liable_party="farmer",
            claim_percent=farmer_claim,
            reasoning=(
                f"Grade dropped from {grade_departure} to {grade_arrival} "
                f"with only {transit_hours:.1f}h transit "
                f"(threshold: <{SHORT_TRANSIT_HOURS}h). "
                "Rapid degradation suggests pre-existing quality issue at departure — "
                "farmer is responsible."
            ),
        )

    # * Rule 6 — Mid-range transit: shared liability between farmer and hauler
    shared_claim = round(base_claim * 0.60, 1)
    return LiabilityResult(
        liable_party="shared",
        claim_percent=shared_claim,
        reasoning=(
            f"Grade dropped from {grade_departure} to {grade_arrival} "
            f"during {transit_hours:.1f}h transit "
            f"({SHORT_TRANSIT_HOURS}–{LONG_TRANSIT_HOURS}h window). "
            "Ambiguous transit window — shared liability between farmer and hauler."
        ),
    )
