"""
RAG API Routes
==============
REST API endpoints for the agentic RAG system.

Production changes (Pillar 2 & 3):
  - KnowledgeAgent pulled from request.app.state (lifespan DI — no global singleton)
  - Redis cache check before LLM call (cache key = sha256 of question+context)
  - Request-scoped logging with trace IDs
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, Field

from src.api.config import get_settings


router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class QueryRequest(BaseModel):
    """RAG query request."""
    question: str
    context: str = ""


class QueryResponse(BaseModel):
    """RAG query response."""
    answer: str
    sources: list[str] = []
    confidence: float
    query_type: str
    steps: list[str] = []
    cached: bool = False


class SearchRequest(BaseModel):
    """Semantic search request."""
    query: str
    top_k: int = 5
    category: Optional[str] = None


class SearchResult(BaseModel):
    """Single search result."""
    text: str
    source: str
    category: str
    score: float


class SearchResponse(BaseModel):
    """Search response."""
    results: list[SearchResult]
    query: str
    total: int


class IngestRequest(BaseModel):
    """Document ingestion request."""
    documents: list[dict] = Field(
        ...,
        description="List of documents with 'text', 'source', 'category' fields"
    )


class IngestResponse(BaseModel):
    """Ingestion response."""
    ingested: int
    message: str


class StatsResponse(BaseModel):
    """Knowledge base stats."""
    collection: str
    vectors_count: int
    points_count: int
    status: str


# =============================================================================
# Dependency: KnowledgeAgent from app.state (set by lifespan)
# =============================================================================

async def get_knowledge_agent(request: Request):
    """
    Retrieve the KnowledgeAgent instance from app.state.

    The agent is initialized once at application startup (lifespan) and
    shared across all requests — no per-request initialization cost.
    Raises 503 if the agent failed to initialize.
    """
    agent = getattr(request.app.state, "knowledge_agent", None)
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="Knowledge agent not available. Check Qdrant connection and logs.",
        )
    return agent


# =============================================================================
# Cache helpers
# =============================================================================

def _cache_key(question: str, context: str) -> str:
    """Deterministic cache key from query inputs."""
    payload = f"{question.strip().lower()}|{context.strip().lower()}"
    return "rag:" + hashlib.sha256(payload.encode()).hexdigest()


async def _try_cache_get(request: Request, key: str) -> Optional[dict]:
    """Try fetching from Redis; returns None if unavailable or miss."""
    redis = getattr(request.app.state, "redis", None)
    if not redis:
        return None
    try:
        raw = await redis.get(key)
        if raw:
            logger.debug("Cache HIT for key {}", key[:16])
            return json.loads(raw)
    except Exception as exc:
        logger.warning("Redis GET failed: {}", exc)
    return None


async def _try_cache_set(request: Request, key: str, value: dict, ttl: int = 300) -> None:
    """Try storing in Redis; silently skips if unavailable."""
    redis = getattr(request.app.state, "redis", None)
    if not redis:
        return
    try:
        await redis.setex(key, ttl, json.dumps(value))
        logger.debug("Cache SET key {} ttl={}s", key[:16], ttl)
    except Exception as exc:
        logger.warning("Redis SET failed: {}", exc)


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/rag/query", response_model=QueryResponse)
async def rag_query(
    request: Request,
    body: QueryRequest,
    agent=Depends(get_knowledge_agent),
):
    """
    Query the knowledge base using agentic RAG.

    Uses adaptive routing, document grading, and self-correction.
    Results are cached in Redis for 5 minutes to reduce LLM cost.
    """
    settings = get_settings()

    # ── Redis cache check ─────────────────────────
    cache_key = _cache_key(body.question, body.context)
    cached = await _try_cache_get(request, cache_key)
    if cached:
        return QueryResponse(**cached, cached=True)

    # ── LLM call ──────────────────────────────────
    response = await agent.answer(
        question=body.question,
        context=body.context,
    )

    result = QueryResponse(
        answer=response.answer,
        sources=response.sources,
        confidence=response.confidence,
        query_type=response.query_type,
        steps=response.steps,
        cached=False,
    )

    # ── Cache the result ──────────────────────────
    if settings.use_redis_cache:
        await _try_cache_set(
            request,
            cache_key,
            result.model_dump(exclude={"cached"}),
            ttl=300,  # 5 minutes
        )

    return result


@router.get("/rag/search", response_model=SearchResponse)
async def rag_search(
    request: Request,
    query: str,
    top_k: int = 5,
    category: Optional[str] = None,
    agent=Depends(get_knowledge_agent),
):
    """
    Semantic search without generation.

    Returns relevant documents based on similarity.
    """
    results = await agent.search(
        query=query,
        top_k=top_k,
        category=category,
    )

    return SearchResponse(
        results=[
            SearchResult(
                text=r["text"],
                source=r["source"],
                category=r["category"],
                score=r["score"] or 0.0,
            )
            for r in results
        ],
        query=query,
        total=len(results),
    )


@router.post("/rag/ingest", response_model=IngestResponse)
async def rag_ingest(
    body: IngestRequest,
    agent=Depends(get_knowledge_agent),
):
    """
    Ingest documents into the knowledge base.

    Each document should have 'text', and optionally 'source', 'category'.
    """
    for i, doc in enumerate(body.documents):
        if "text" not in doc:
            raise HTTPException(
                status_code=400,
                detail=f"Document {i} missing 'text' field",
            )

    count = await agent.ingest_documents(body.documents)

    return IngestResponse(
        ingested=count,
        message=f"Successfully ingested {count} documents",
    )


@router.get("/rag/stats", response_model=StatsResponse)
async def rag_stats(agent=Depends(get_knowledge_agent)):
    """Get knowledge base statistics."""
    stats = agent.get_stats()

    if "error" in stats:
        raise HTTPException(status_code=500, detail=stats["error"])

    return StatsResponse(
        collection=stats.get("collection", ""),
        vectors_count=stats.get("vectors_count", 0),
        points_count=stats.get("points_count", 0),
        status=stats.get("status", "unknown"),
    )


# =============================================================================
# Sprint 05 Test Endpoints (for static test UI)
# =============================================================================

class RouteTestRequest(BaseModel):
    """Request to test the adaptive query router."""
    query: str = Field(description="Query text to route")
    has_image: bool = Field(default=False, description="Simulate image attachment")


class RouteTestResponse(BaseModel):
    """Response from router test."""
    strategy: str
    confidence: float
    reason: str
    estimated_cost_inr: float
    requires_live_data: bool
    requires_image: bool
    pre_filter_matched: bool
    entities: dict


class NormalizeTestRequest(BaseModel):
    """Request to test AgriEmbeddingWrapper term normalization."""
    text: str = Field(description="Text to normalize (may contain Hindi/Kannada terms)")


class NormalizeTestResponse(BaseModel):
    """Response from normalization test."""
    original: str
    normalized: str
    term_map_size: int
    terms_expanded: list[str]


@router.post("/rag/route", response_model=RouteTestResponse)
async def test_router(request: RouteTestRequest):
    """
    Test the Adaptive Query Router (Sprint 05).

    Routes the query using the 8-strategy router with USE_ADAPTIVE_ROUTER forced=true.
    Uses rule-based pre-filter only (no LLM cost).
    """
    import os
    orig = os.environ.get("USE_ADAPTIVE_ROUTER", "false")
    os.environ["USE_ADAPTIVE_ROUTER"] = "true"

    try:
        from ai.rag.query_analyzer import AdaptiveQueryRouter

        router_instance = AdaptiveQueryRouter(llm=None)
        decision = await router_instance.route(
            query=request.query,
            has_image=request.has_image,
        )

        return RouteTestResponse(
            strategy=decision.strategy.value,
            confidence=decision.confidence,
            reason=decision.reason,
            estimated_cost_inr=decision.estimated_cost_inr,
            requires_live_data=decision.requires_live_data,
            requires_image=decision.requires_image,
            pre_filter_matched=decision.pre_filter_matched,
            entities=decision.entities,
        )
    finally:
        os.environ["USE_ADAPTIVE_ROUTER"] = orig


@router.post("/rag/normalize", response_model=NormalizeTestResponse)
async def test_normalize(request: NormalizeTestRequest):
    """
    Test the AgriEmbeddingWrapper bilingual normalization (Sprint 05).

    Returns the expanded English text with all Hindi/Kannada terms replaced.
    No model loading required — pure text transformation.
    """
    from ai.rag.agri_embeddings import AgriEmbeddingWrapper

    wrapper = AgriEmbeddingWrapper.__new__(AgriEmbeddingWrapper)
    wrapper.enable_term_normalization = True

    normalized = wrapper._normalize_terms(request.text)

    terms_expanded = [
        f"{term} → {AgriEmbeddingWrapper.TERM_MAP[term]}"
        for term in sorted(AgriEmbeddingWrapper.TERM_MAP.keys(), key=len, reverse=True)
        if term in request.text.lower()
    ]

    return NormalizeTestResponse(
        original=request.text,
        normalized=normalized,
        term_map_size=len(AgriEmbeddingWrapper.TERM_MAP),
        terms_expanded=terms_expanded,
    )
