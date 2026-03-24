"""
Quality Assessment Agent (CV-QG)
================================
AI-powered produce grading with HITL fallback and digital twin linkage.
"""

from __future__ import annotations

import base64
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, Field

from src.agents.base_agent import AgentConfig, AgentResponse, BaseAgent
from src.agents.digital_twin.engine import DigitalTwinEngine, get_digital_twin_engine
from src.agents.digital_twin.models import DiffReport, DigitalTwin
from src.agents.quality_assessment.vision_models import CropVisionPipeline, QualityResult
from src.api.services.hitl_service import HITLNotificationService
from src.memory.state_manager import AgentExecutionState
from src.orchestrator.llm_provider import BaseLLMProvider

# * ═══════════════════════════════════════════════════════════════
# * DATA MODELS
# * ═══════════════════════════════════════════════════════════════

VALID_GRADES = ("A+", "A", "B", "C")

HITL_CONFIDENCE_THRESHOLD = 0.7


class GradeAssessment(BaseModel):
    """Result of a single produce quality assessment."""

    listing_id: str
    commodity: str
    grade: str
    confidence: float
    hitl_required: bool
    defects_detected: list[str] = Field(default_factory=list)
    defect_count: int = 0
    shelf_life_days: int = 0
    reasoning: str = ""
    assessment_id: str = ""
    assessed_at: datetime = Field(default_factory=datetime.now)


class QualityReport(BaseModel):
    """Full quality report that can be stored as a Digital Twin."""

    assessment: GradeAssessment
    image_count: int = 0
    method: str = "manual"  # "vision" | "rule_based" | "manual"
    digital_twin_linked: bool = False


class QualityAssessmentAgent(BaseAgent):
    """
    CV-QG Agent — grades produce and flags items for HITL review.

    Current implementation uses LLM analysis of user-provided
    descriptions (and optionally base64 images). Phase 2 will add
    YOLOv8 + ViT vision inference for <500ms automated grading.

    Usage:
        agent = QualityAssessmentAgent(llm=provider)
        await agent.initialize()
        report = await agent.assess(listing_id="abc", commodity="Tomato",
                                     description="Red, firm, no spots")
    """

    def __init__(
        self,
        llm: Optional[BaseLLMProvider] = None,
        twin_engine: Optional[DigitalTwinEngine] = None,
        hitl_service: Optional[HITLNotificationService] = None,
        **kwargs: Any,
    ):
        config = AgentConfig(
            name="quality_assessment",
            description="AI produce grading with HITL fallback for dispute-proof quality verification",
            max_retries=1,
            temperature=0.2,
            max_tokens=400,
            kb_categories=["agronomy"],
        )
        super().__init__(config=config, llm=llm, **kwargs)
        self.vision_pipeline = CropVisionPipeline()
        # NOTE: twin_engine injected for testability; defaults to in-memory engine
        self.twin_engine: DigitalTwinEngine = twin_engine or get_digital_twin_engine()
        # * FR9: HITL dispatch service — log-only mode when None
        self.hitl_service: Optional[HITLNotificationService] = hitl_service
        self._digital_twin_store: dict[str, QualityReport] = {}

    def _get_system_prompt(self, context: Optional[dict] = None) -> str:
        return """You are CropFresh's Quality Assessment Agent.
Given a description (and optionally photos) of harvested produce,
determine the quality grade and any defects.

Respond ONLY with a JSON object:
{
    "grade": "A+" | "A" | "B" | "C",
    "confidence": 0.0-1.0,
    "defects": ["defect_type", ...],
    "shelf_life_days": integer,
    "reasoning": "brief explanation"
}

Grade definitions:
  A+ — Export premium quality, no visible defects
  A  — Retail quality, minor cosmetic issues only
  B  — Wholesale quality, visible but acceptable defects
  C  — Processing quality, multiple defects"""

    async def process(
        self,
        query: str,
        context: Optional[dict] = None,
        execution: Optional[AgentExecutionState] = None,
    ) -> AgentResponse:
        """Process a natural-language quality question via the supervisor."""
        if not self.llm:
            return AgentResponse(
                content=self._rule_based_response(query),
                agent_name=self.name,
                confidence=0.6,
                steps=["rule_based_grading"],
            )

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": query},
        ]
        response_text = await self.generate_with_llm(messages, context=context)

        return AgentResponse(
            content=response_text,
            agent_name=self.name,
            confidence=0.8,
            steps=["llm_quality_analysis"],
        )

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        image_b64 = input_data.get("image_b64")
        report = await self.assess(
            listing_id=input_data.get("listing_id", f"lst-{uuid4().hex[:8]}"),
            commodity=input_data.get("commodity", "tomato"),
            description=input_data.get("description", ""),
            image_b64=image_b64,
            require_upgrade_review=bool(input_data.get("require_upgrade_review", False)),
        )
        return {
            "grade": report.assessment.grade,
            "confidence": report.assessment.confidence,
            "defects": report.assessment.defects_detected,
            "defect_count": report.assessment.defect_count,
            "hitl_required": report.assessment.hitl_required,
            "shelf_life_days": report.assessment.shelf_life_days,
            "assessment_id": report.assessment.assessment_id,
            "digital_twin_linked": report.digital_twin_linked,
            "message": self._format_result_message(report.assessment, report.assessment.commodity),
        }

    async def assess(
        self,
        listing_id: str,
        commodity: str,
        description: str = "",
        image_b64: Optional[str] = None,
        require_upgrade_review: bool = False,
    ) -> QualityReport:
        """
        Perform a quality assessment on a produce listing.

        Args:
            listing_id: The listing to assess
            commodity: Commodity name (e.g., "Tomato")
            description: Text description of produce condition
            image_b64: Optional base64-encoded photo

        Returns:
            QualityReport with grade, confidence, and HITL flag
        """
        image_data: Optional[bytes] = None
        if image_b64:
            try:
                image_data = base64.b64decode(image_b64)
            except Exception:
                image_data = None

        if image_data:
            quality_result = await self.vision_pipeline.assess_quality(
                image_data, commodity, description
            )
            image_count = 1
        else:
            quality_result = await self.vision_pipeline.assess_description(commodity, description)
            image_count = 0

        hitl_required = (
            quality_result.hitl_required
            or require_upgrade_review
            or quality_result.confidence < HITL_CONFIDENCE_THRESHOLD
            or quality_result.grade == "A+"
        )

        assessment_id = f"qa-{uuid4().hex[:12]}"
        assessment = GradeAssessment(
            listing_id=listing_id,
            commodity=commodity,
            grade=quality_result.grade if quality_result.grade in VALID_GRADES else "C",
            confidence=quality_result.confidence,
            hitl_required=hitl_required,
            defects_detected=quality_result.defects,
            defect_count=quality_result.defect_count,
            shelf_life_days=quality_result.shelf_life_days,
            reasoning=self._build_reasoning(quality_result, require_upgrade_review),
            assessment_id=assessment_id,
        )

        report = QualityReport(
            assessment=assessment,
            image_count=image_count,
            method=quality_result.assessment_mode,
            digital_twin_linked=True,
        )
        self._digital_twin_store[assessment_id] = report
        logger.info(
            "Quality assessed: {} grade={} conf={:.2f} hitl={} twin={}",
            commodity,
            assessment.grade,
            assessment.confidence,
            assessment.hitl_required,
            assessment_id,
        )

        # * FR9: dispatch HITL review notification when flag is set
        if assessment.hitl_required:
            await self._dispatch_hitl(
                listing_id=listing_id,
                confidence=assessment.confidence,
                grade=assessment.grade,
                defect_count=assessment.defect_count,
            )

        return report

    async def _dispatch_hitl(
        self,
        listing_id: str,
        confidence: float,
        grade: str,
        defect_count: int,
    ) -> None:
        """
        Dispatch HITL review notification (FR9).
        Delegates to HITLNotificationService when injected; logs only otherwise.
        """
        if self.hitl_service:
            await self.hitl_service.trigger_review(
                listing_id=listing_id,
                confidence=confidence,
                grade=grade,
                defect_count=defect_count,
            )
        else:
            logger.debug(
                "HITL required for listing {} (conf={:.2f} grade={} defects={}) — "
                "no HITLNotificationService injected; log-only mode",
                listing_id,
                confidence,
                grade,
                defect_count,
            )

    # ─────────────────────────────────────────────────────────
    # * Digital Twin Integration
    # ─────────────────────────────────────────────────────────

    async def create_departure_twin(
        self,
        listing_id: str,
        farmer_photos: list[str],
        agent_photos: list[str],
        quality_result: QualityResult,
        gps: tuple[float, float] = (0.0, 0.0),
    ) -> DigitalTwin:
        """
        Create an immutable departure twin snapshot for a listing.

        Delegates to DigitalTwinEngine.create_departure_twin().
        Stores twin_id in _digital_twin_store for fast lookup.

        Args:
            listing_id:     UUID of the crop listing.
            farmer_photos:  S3 URLs of farmer-submitted photos.
            agent_photos:   S3 URLs of field agent verification photos.
            quality_result: QualityResult from the assess() pipeline.
            gps:            (lat, lng) at farm gate.

        Returns:
            DigitalTwin with stable twin_id.
        """
        twin = await self.twin_engine.create_departure_twin(
            listing_id=listing_id,
            farmer_photos=farmer_photos,
            agent_photos=agent_photos,
            quality_result=quality_result,
            gps=gps,
        )
        logger.info(
            "Departure twin {} linked to listing {} (grade={})",
            twin.twin_id,
            listing_id,
            twin.grade,
        )
        return twin

    async def compare_twin(
        self,
        twin_id: str,
        arrival_photos: list[str],
        arrival_gps: tuple[float, float] = (0.0, 0.0),
    ) -> DiffReport:
        """
        Compare a departure twin against buyer arrival photos.

        Called by OrderService._trigger_twin_diff() during dispute resolution.
        Delegates to DigitalTwinEngine.compare_arrival().

        Args:
            twin_id:        Digital twin ID from create_departure_twin().
            arrival_photos: S3 URLs of buyer arrival photos.
            arrival_gps:    (lat, lng) at buyer delivery point.

        Returns:
            DiffReport with quality delta, liability, and explanation.

        Raises:
            ValueError: If departure twin not found.
        """
        logger.info(
            "Comparing twin {} against {} arrival photo(s)",
            twin_id,
            len(arrival_photos),
        )
        return await self.twin_engine.compare_arrival(
            twin_id=twin_id,
            arrival_photos=arrival_photos,
            arrival_gps=arrival_gps,
        )

    def _rule_based_response(self, query: str) -> str:
        """Quick text response for supervisor routing."""
        return (
            "Based on the description, I'd recommend getting a field agent "
            "to verify the produce quality. You can submit photos for AI "
            "grading, or describe the condition and I'll give a preliminary grade."
        )

    @staticmethod
    def _build_reasoning(result: QualityResult, require_upgrade_review: bool) -> str:
        reasons = [f"AI grade {result.grade} at confidence {result.confidence:.2f}"]
        if result.defects:
            reasons.append(f"Defects: {', '.join(result.defects)}")
        if require_upgrade_review:
            reasons.append("Farmer requested grade upgrade review")
        if result.hitl_required:
            reasons.append("HITL required by quality policy")
        return ". ".join(reasons)

    @staticmethod
    def _format_result_message(assessment: GradeAssessment, commodity: str) -> str:
        return (
            f"{commodity.title()} graded as {assessment.grade} "
            f"(confidence {assessment.confidence:.2f}). "
            f"Shelf life estimate: {assessment.shelf_life_days} days. "
            f"HITL required: {'yes' if assessment.hitl_required else 'no'}."
        )
