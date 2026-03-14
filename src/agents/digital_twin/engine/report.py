"""
Diff Report Mixin
=================
Handles execution of AI diff reporting comparing twins against arrival.
"""

from typing import Any

from loguru import logger

from src.agents.digital_twin.diff_analysis import (
    compute_grade_delta,
    compute_new_defects,
    compute_similarity,
)
from src.agents.digital_twin.liability import determine_liability
from src.agents.digital_twin.models import ArrivalData, DiffReport, DigitalTwin

from .utils import (
    build_explanation,
    compute_report_confidence,
    compute_transit_hours,
    estimate_arrival_grade,
    infer_arrival_defects,
)


class DiffReportMixin:
    """Mixin containing report generation orchestration."""

    # Instance vars expected to be bounded
    similarity_engine: Any

    async def generate_diff_report(
        self,
        departure_twin: DigitalTwin,
        arrival_data: ArrivalData,
    ) -> DiffReport:
        """
        Generate an AI-powered diff report for a departure twin vs. arrival state.
        """
        transit_hours = compute_transit_hours(
            departure_twin.created_at, arrival_data.arrived_at
        )

        grade_arrival = estimate_arrival_grade(
            departure_twin=departure_twin,
            arrival_photos=arrival_data.arrival_photos,
            transit_hours=transit_hours,
        )

        arrival_defects = infer_arrival_defects(grade_arrival, departure_twin.defect_types)

        substitution_flag = False
        if self.similarity_engine.available:
            batch_result = self.similarity_engine.compare_url_batches(
                departure_twin.all_photos(),
                arrival_data.arrival_photos,
            )
            similarity_score  = batch_result["similarity_score"]
            substitution_flag = batch_result["substitution_flag"]
            analysis_method   = "resnet50"
            logger.debug(
                "ResNet50 similarity: score={:.4f} min={:.4f} substitution={}",
                similarity_score, batch_result["min_score"], substitution_flag,
            )
        else:
            similarity_score, analysis_method = compute_similarity(
                departure_photos=departure_twin.all_photos(),
                arrival_photos=arrival_data.arrival_photos,
                grade_departure=departure_twin.grade,
                grade_arrival=grade_arrival,
                departure_defects=departure_twin.defect_types,
                arrival_defects=arrival_defects,
            )

        quality_delta = compute_grade_delta(departure_twin.grade, grade_arrival)
        new_defects = compute_new_defects(departure_twin.defect_types, arrival_defects)

        liability_result = determine_liability(
            grade_departure=departure_twin.grade,
            grade_arrival=grade_arrival,
            quality_delta=quality_delta,
            transit_hours=transit_hours,
            new_defects=new_defects,
            has_arrival_photos=bool(arrival_data.arrival_photos),
            substitution_flag=substitution_flag,
        )

        confidence = compute_report_confidence(
            similarity_score=similarity_score,
            has_photos=bool(arrival_data.arrival_photos),
            departure_confidence=departure_twin.confidence,
            analysis_method=analysis_method,
        )

        explanation = build_explanation(
            departure_twin=departure_twin,
            grade_arrival=grade_arrival,
            new_defects=new_defects,
            transit_hours=transit_hours,
            similarity_score=similarity_score,
            liability_result=liability_result,
        )

        diff = DiffReport(
            quality_delta=quality_delta,
            grade_departure=departure_twin.grade,
            grade_arrival=grade_arrival,
            new_defects=new_defects,
            similarity_score=similarity_score,
            transit_hours=transit_hours,
            liability=liability_result.liable_party,
            claim_percent=liability_result.claim_percent,
            confidence=confidence,
            explanation=explanation,
            analysis_method=analysis_method,
        )

        logger.info(
            "DiffReport: {} → {} | delta={:.3f} | sim={:.3f} | "
            "liability={} | claim={}% | conf={:.3f} | method={}",
            diff.grade_departure, diff.grade_arrival, diff.quality_delta,
            diff.similarity_score, diff.liability, diff.claim_percent,
            diff.confidence, diff.analysis_method,
        )
        return diff
