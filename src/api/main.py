"""
CropFresh AI Service - FastAPI Application
==========================================
Main entry point for the AI service API.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    settings = get_settings()
    logger.info("🌾 CropFresh AI Service starting...")
    logger.info(f"   LLM Provider: {settings.llm_provider}")
    logger.info(f"   Debug mode: {settings.debug}")
    
    # Startup
    yield
    
    # Shutdown
    logger.info("🌾 CropFresh AI Service shutting down...")


app = FastAPI(
    title="CropFresh AI Service",
    description="AI-powered agricultural marketplace backend",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "cropfresh-ai",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/health/ready")
async def readiness():
    """Readiness check endpoint."""
    settings = get_settings()
    
    checks = {
        "llm_configured": bool(settings.groq_api_key),
        "qdrant_configured": bool(settings.qdrant_host),
    }
    
    all_ready = all(checks.values())
    
    return {
        "ready": all_ready,
        "checks": checks,
    }


# API routes
from src.api.routes import rag
app.include_router(rag.router, prefix="/api/v1", tags=["rag"])

# Chat API routes (Advanced Multi-Agent System)
from src.api.routes import chat
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])

# Voice API routes
from src.api.rest import voice as voice_rest
app.include_router(voice_rest.router, tags=["voice"])

# WebSocket routes
from src.api import websocket as voice_ws
app.include_router(voice_ws.router, tags=["websocket"])


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
