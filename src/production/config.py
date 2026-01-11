"""
Production Configuration
========================
Environment-based configuration for production deployment.

Features:
- Environment-based settings
- Secret management
- Feature flags
- Configuration validation
"""

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field, SecretStr


class Environment(str, Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    api_key: Optional[SecretStr] = None
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout_sec: int = 30


class DatabaseConfig(BaseModel):
    """Database configuration."""
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[SecretStr] = None
    
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: Optional[SecretStr] = None


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    enabled: bool = True
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10


class CacheConfig(BaseModel):
    """Cache configuration."""
    enabled: bool = True
    ttl_seconds: int = 300
    max_entries: int = 1000


class ObservabilityConfig(BaseModel):
    """Observability configuration."""
    enabled: bool = True
    otlp_endpoint: Optional[str] = None
    log_level: str = "INFO"
    trace_sample_rate: float = 0.1


class FeatureFlags(BaseModel):
    """Feature flags."""
    enable_research_agent: bool = True
    enable_web_scraping: bool = True
    enable_voice: bool = False
    enable_vision: bool = False
    enable_autonomous_tasks: bool = False


class ProductionConfig(BaseModel):
    """Complete production configuration."""
    environment: Environment = Environment.DEVELOPMENT
    service_name: str = "cropfresh-ai"
    version: str = "1.0.0"
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    
    # Component configs
    llm: LLMConfig = Field(default_factory=LLMConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    features: FeatureFlags = Field(default_factory=FeatureFlags)


def load_config(env_file: Optional[Path] = None) -> ProductionConfig:
    """
    Load configuration from environment.
    
    Args:
        env_file: Optional .env file path
        
    Returns:
        ProductionConfig with all settings
    """
    # Load .env file if provided
    if env_file and env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            logger.info("Loaded environment from {}", env_file)
        except ImportError:
            pass
    
    # Determine environment
    env_str = os.getenv("ENVIRONMENT", "development").lower()
    environment = Environment(env_str) if env_str in [e.value for e in Environment] else Environment.DEVELOPMENT
    
    # Build config from environment
    config = ProductionConfig(
        environment=environment,
        service_name=os.getenv("SERVICE_NAME", "cropfresh-ai"),
        version=os.getenv("VERSION", "1.0.0"),
        
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", "8000")),
        cors_origins=os.getenv("CORS_ORIGINS", "*").split(","),
        
        llm=LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "groq"),
            model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            api_key=SecretStr(os.getenv("GROQ_API_KEY", "")) if os.getenv("GROQ_API_KEY") else None,
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4000")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        ),
        
        database=DatabaseConfig(
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            qdrant_api_key=SecretStr(os.getenv("QDRANT_API_KEY", "")) if os.getenv("QDRANT_API_KEY") else None,
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=SecretStr(os.getenv("NEO4J_PASSWORD", "")) if os.getenv("NEO4J_PASSWORD") else None,
        ),
        
        rate_limit=RateLimitConfig(
            enabled=os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
            requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "60")),
            requests_per_hour=int(os.getenv("RATE_LIMIT_RPH", "1000")),
        ),
        
        cache=CacheConfig(
            enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
            ttl_seconds=int(os.getenv("CACHE_TTL", "300")),
        ),
        
        observability=ObservabilityConfig(
            enabled=os.getenv("OBSERVABILITY_ENABLED", "true").lower() == "true",
            otlp_endpoint=os.getenv("OTLP_ENDPOINT"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            trace_sample_rate=float(os.getenv("TRACE_SAMPLE_RATE", "0.1")),
        ),
        
        features=FeatureFlags(
            enable_research_agent=os.getenv("FEATURE_RESEARCH", "true").lower() == "true",
            enable_web_scraping=os.getenv("FEATURE_SCRAPING", "true").lower() == "true",
            enable_voice=os.getenv("FEATURE_VOICE", "false").lower() == "true",
            enable_vision=os.getenv("FEATURE_VISION", "false").lower() == "true",
            enable_autonomous_tasks=os.getenv("FEATURE_AUTONOMOUS", "false").lower() == "true",
        ),
    )
    
    logger.info("Loaded {} configuration", environment.value)
    return config


# Global config instance
_config: Optional[ProductionConfig] = None


def get_config() -> ProductionConfig:
    """Get or create global config."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
