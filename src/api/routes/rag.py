"""
RAG API Routes
==============
REST API endpoints for the agentic RAG system.
"""

from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.config import get_settings


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
# Global Agent Instance
# =============================================================================

_knowledge_agent = None


async def get_knowledge_agent():
    """Get or create knowledge agent instance."""
    global _knowledge_agent
    
    if _knowledge_agent is None:
        from src.agents.knowledge_agent import KnowledgeAgent
        from src.orchestrator.llm_provider import create_llm_provider
        
        settings = get_settings()
        
        # Create LLM if API key configured
        llm = None
        if settings.groq_api_key:
            llm = create_llm_provider(
                provider=settings.llm_provider,
                api_key=settings.groq_api_key,
                model=settings.llm_model,
            )
        
        _knowledge_agent = KnowledgeAgent(
            llm=llm,
            qdrant_host=settings.qdrant_host,
            qdrant_port=settings.qdrant_port,
        )
        await _knowledge_agent.initialize()
    
    return _knowledge_agent


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/rag/query", response_model=QueryResponse)
async def rag_query(request: QueryRequest):
    """
    Query the knowledge base using agentic RAG.
    
    Uses adaptive routing, document grading, and self-correction.
    """
    agent = await get_knowledge_agent()
    
    response = await agent.answer(
        question=request.question,
        context=request.context,
    )
    
    return QueryResponse(
        answer=response.answer,
        sources=response.sources,
        confidence=response.confidence,
        query_type=response.query_type,
        steps=response.steps,
    )


@router.get("/rag/search", response_model=SearchResponse)
async def rag_search(
    query: str,
    top_k: int = 5,
    category: Optional[str] = None,
):
    """
    Semantic search without generation.
    
    Returns relevant documents based on similarity.
    """
    agent = await get_knowledge_agent()
    
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
async def rag_ingest(request: IngestRequest):
    """
    Ingest documents into the knowledge base.
    
    Each document should have 'text', and optionally 'source', 'category'.
    """
    agent = await get_knowledge_agent()
    
    # Validate documents
    for i, doc in enumerate(request.documents):
        if "text" not in doc:
            raise HTTPException(
                status_code=400,
                detail=f"Document {i} missing 'text' field"
            )
    
    count = await agent.ingest_documents(request.documents)
    
    return IngestResponse(
        ingested=count,
        message=f"Successfully ingested {count} documents",
    )


@router.get("/rag/stats", response_model=StatsResponse)
async def rag_stats():
    """Get knowledge base statistics."""
    agent = await get_knowledge_agent()
    stats = agent.get_stats()
    
    if "error" in stats:
        raise HTTPException(status_code=500, detail=stats["error"])
    
    return StatsResponse(
        collection=stats.get("collection", ""),
        vectors_count=stats.get("vectors_count", 0),
        points_count=stats.get("points_count", 0),
        status=stats.get("status", "unknown"),
    )
