"""
CropFresh AI Configuration Management
=====================================
Centralized settings using Pydantic Settings.
"""

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
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
    debug: bool = False  # IMPORTANT: must be False in production
    log_level: str = "INFO"

    # Deployment environment: development | staging | production
    environment: str = "development"

    # ═══════════════════════════════════════════════════════════════
    # Security
    # ═══════════════════════════════════════════════════════════════
    # API key for X-API-Key header authentication.
    # Leave empty in development (auth is skipped automatically).
    api_key: str = ""

    # Comma-separated list of allowed CORS origins.
    # Example: https://app.cropfresh.in,https://admin.cropfresh.in
    # Use "*" for development only — never in production.
    allowed_origins: str = "*"

    # ═══════════════════════════════════════════════════════════════
    # Observability
    # ═══════════════════════════════════════════════════════════════
    # OTLP endpoint for OpenTelemetry trace export.
    # Example: http://otel-collector:4317 (gRPC)
    otel_endpoint: str = ""

    # LangSmith tracing (for agent evaluation pipeline)
    langsmith_api_key: str = ""
    langsmith_project: str = "cropfresh-ai"

    # ═══════════════════════════════════════════════════════════════
    # AI Kosha (India AI dataset platform)
    # ═══════════════════════════════════════════════════════════════
    aikosha_api_key: str = ""
    aikosha_base_url: str = "https://indiaai.gov.in/api/v1"

    # ═══════════════════════════════════════════════════════════════
    # Feature Flags
    # ═══════════════════════════════════════════════════════════════
    use_adaptive_router: bool = True  # Enable 8-strategy adaptive query router
    use_redis_cache: bool = True      # Enable Redis response cache
    enable_reranker: bool = True      # Enable Cohere/cross-encoder reranker

    # Scraping
    scraping_rate_limit: int = 30
    scraping_cache_ttl: int = 300

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v.lower() not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v.lower()

    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse allowed_origins CSV into a list."""
        if self.allowed_origins == "*":
            return ["*"]
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    # ═══════════════════════════════════════════════════════════════
    # Model Paths
    # ═══════════════════════════════════════════════════════════════
    yolo_model_path: str = "models/yolov11m.pt"
    whisper_model_size: str = "large-v3-turbo"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
