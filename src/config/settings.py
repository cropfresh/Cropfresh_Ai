"""
CropFresh AI Configuration Management
=====================================
Centralized settings using Pydantic Settings.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ═══════════════════════════════════════════════════════════════
    # LLM Provider
    # ═══════════════════════════════════════════════════════════════
    llm_provider: Literal["groq", "together", "vllm"] = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    
    groq_api_key: str = ""
    together_api_key: str = ""
    vllm_base_url: str = "http://localhost:8000/v1"

    # ═══════════════════════════════════════════════════════════════
    # Vector Database
    # ═══════════════════════════════════════════════════════════════
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str = ""
    qdrant_collection: str = "agri_knowledge"

    # ═══════════════════════════════════════════════════════════════
    # Embeddings
    # ═══════════════════════════════════════════════════════════════
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: Literal["cpu", "cuda"] = "cpu"

    # ═══════════════════════════════════════════════════════════════
    # External APIs
    # ═══════════════════════════════════════════════════════════════
    agmarknet_api_key: str = ""
    weather_api_key: str = ""
    google_maps_api_key: str = ""

    # ═══════════════════════════════════════════════════════════════
    # Redis
    # ═══════════════════════════════════════════════════════════════
    redis_url: str = "redis://localhost:6379/0"

    # ═══════════════════════════════════════════════════════════════
    # Supabase (PostgreSQL)
    # ═══════════════════════════════════════════════════════════════
    supabase_url: str = ""
    supabase_key: str = ""

    # ═══════════════════════════════════════════════════════════════
    # Neo4j (Graph Database)
    # ═══════════════════════════════════════════════════════════════
    neo4j_uri: str = ""
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # ═══════════════════════════════════════════════════════════════
    # Database (Legacy - use Supabase instead)
    # ═══════════════════════════════════════════════════════════════
    database_url: str = "postgresql://user:password@localhost:5432/cropfresh"

    # ═══════════════════════════════════════════════════════════════
    # API Settings
    # ═══════════════════════════════════════════════════════════════
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    debug: bool = True
    log_level: str = "INFO"

    # ═══════════════════════════════════════════════════════════════
    # Model Paths
    # ═══════════════════════════════════════════════════════════════
    yolo_model_path: str = "models/yolov11m.pt"
    whisper_model_size: str = "large-v3-turbo"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
