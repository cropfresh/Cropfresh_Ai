"""Vision assessment routes for the static testing lab and listing workflows."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from src.api.routers.vision_models import (
    GradeAttachPreview,
    VisionAssessRequest,
    VisionAssessResponse,
    VisionHealthResponse,
)

router = APIRouter(prefix="/vision", tags=["vision"])


def _quality_agent(request: Request):
    listing_service = getattr(request.app.state, "listing_service", None)
    if listing_service is not None and getattr(listing_service, "quality_agent", None) is not None:
        return listing_service.quality_agent
    supervisor = getattr(request.app.state, "supervisor", None)
    agents = getattr(supervisor, "_agents", {}) if supervisor is not None else {}
    return agents.get("quality_assessment_agent")


def _message(commodity: str, grade: str, defects: list[str]) -> str:
    if defects:
        return f"{commodity} graded {grade} with defects: {', '.join(defects)}."
    return f"{commodity} graded {grade} with no major visible defects detected."


@router.get("/health", response_model=VisionHealthResponse)
async def get_vision_health(request: Request) -> VisionHealthResponse:
    """Expose whether the shared quality agent is available and in full vision mode."""
    quality_agent = _quality_agent(request)
    if quality_agent is None:
        return VisionHealthResponse(
            service_ready=False,
            vision_ready=False,
            assessment_mode="unavailable",
            model_dir="models/vision/",
        )
    pipeline = getattr(quality_agent, "vision_pipeline", None)
    fallback_mode = bool(getattr(pipeline, "fallback_mode", True))
    return VisionHealthResponse(
        service_ready=True,
        vision_ready=not fallback_mode,
        assessment_mode="vision" if not fallback_mode else "rule_based",
        model_dir=getattr(pipeline, "model_dir", "models/vision/"),
    )


@router.post("/assess", response_model=VisionAssessResponse)
async def assess_vision(payload: VisionAssessRequest, request: Request) -> VisionAssessResponse:
    """Run the shared quality agent and return a UI-friendly vision contract."""
    quality_agent = _quality_agent(request)
    if quality_agent is None:
        raise HTTPException(status_code=503, detail="Quality assessment agent is not available")

    try:
        report = await quality_agent.assess(
            listing_id=payload.listing_id or "vision-lab-preview",
            commodity=payload.commodity,
            description=payload.description,
            image_b64=payload.image_b64,
            require_upgrade_review=payload.require_upgrade_review,
        )
    except Exception as exc:
        logger.error("POST /vision/assess error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    assessment = report.assessment
    defects = assessment.defects_detected
    confidence = round(float(assessment.confidence), 4)
    return VisionAssessResponse(
        listing_id=assessment.listing_id,
        commodity=assessment.commodity,
        grade=assessment.grade,
        confidence=confidence,
        defects=defects,
        defect_count=assessment.defect_count,
        hitl_required=assessment.hitl_required,
        shelf_life_days=assessment.shelf_life_days,
        assessment_id=assessment.assessment_id,
        digital_twin_linked=report.digital_twin_linked,
        assessment_mode=report.method,
        vision_ready=not bool(getattr(quality_agent.vision_pipeline, "fallback_mode", True)),
        message=_message(assessment.commodity, assessment.grade, defects),
        grade_attach_preview=GradeAttachPreview(
            grade=assessment.grade,
            cv_confidence=confidence,
            defect_types=defects,
        ),
    )
