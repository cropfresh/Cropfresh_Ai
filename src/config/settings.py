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
    llm_provider: Literal["bedrock", "groq", "together", "vllm"] = "bedrock"
    llm_model: str = "claude-sonnet-4"

    # Amazon Bedrock
    aws_region: str = "ap-south-1"
    aws_profile: str = ""
    # ! SECURITY: AWS credentials via env vars (AWS_ACCESS_KEY_ID,
    # ! AWS_SECRET_ACCESS_KEY) or IAM role — never hardcode.
    bedrock_router_model: str = "claude-haiku"

    # Groq (dev / speed-critical tasks)
    groq_api_key: str = ""
    together_api_key: str = ""
    vllm_base_url: str = "http://localhost:8000/v1"

    # ═══════════════════════════════════════════════════════════════
    # Vector Database
    # ═══════════════════════════════════════════════════════════════
    vector_db_provider: Literal["pgvector", "qdrant"] = "pgvector"
    # Qdrant (dev/legacy fallback)
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
    # Amazon Aurora PostgreSQL (replaces Supabase + Qdrant vectors)
    # ═══════════════════════════════════════════════════════════════
    pg_host: str = ""       # Aurora endpoint
    pg_database: str = "cropfresh"
    pg_port: int = 5432
    pg_user: str = "cropfresh_app"
    pg_password: str = ""   # Empty = use IAM auth
    pg_use_iam_auth: bool = False  # True in production
    # Legacy Supabase (kept for migration period)
    supabase_url: str = ""
    supabase_key: str = ""

    # ═══════════════════════════════════════════════════════════════
    # Neo4j (Graph Database) — kept as-is
    # ═══════════════════════════════════════════════════════════════
    neo4j_uri: str = ""
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # ═══════════════════════════════════════════════════════════════
    # AI Kosha (India's AI Dataset Platform)
    # ═══════════════════════════════════════════════════════════════
    aikosha_api_key: str = ""
    aikosha_base_url: str = "https://indiaai.gov.in/api/v1"

    # ═══════════════════════════════════════════════════════════════
    # Scraping Configuration
    # ═══════════════════════════════════════════════════════════════
    scraping_rate_limit: int = 30  # requests per minute (global)
    scraping_cache_ttl: int = 300  # cache TTL in seconds (5 min)

    # ═══════════════════════════════════════════════════════════════
    # API Settings
    # ═══════════════════════════════════════════════════════════════
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    debug: bool = True
    log_level: str = "INFO"
    api_key: str = ""  # API key for authentication (empty = disabled)
    environment: str = "development"  # development, staging, production

    @property
    def has_llm_configured(self) -> bool:
        """Check if any LLM provider has valid credentials."""
        if self.llm_provider == "bedrock":
            return True  # Bedrock uses IAM roles / env vars — boto3 validates at runtime
        if self.llm_provider == "groq":
            return bool(self.groq_api_key)
        if self.llm_provider == "together":
            return bool(self.together_api_key)
        if self.llm_provider == "vllm":
            return bool(self.vllm_base_url)
        return False

    # ═══════════════════════════════════════════════════════════════
    # Model Paths
    # ═══════════════════════════════════════════════════════════════
    yolo_model_path: str = "models/yolov11m.pt"
    whisper_model_size: str = "large-v3-turbo"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
