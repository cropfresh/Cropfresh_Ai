"""Configuration for the Sprint 08 VAD service."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class VadServiceSettings(BaseSettings):
    """Settings shared by the FastAPI and gRPC surfaces."""

    host: str = "0.0.0.0"
    port: int = 8101
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50061
    service_version: str = "0.1.0"
    log_level: str = "INFO"

    sample_rate: int = 16000
    frame_samples: int = 512
    rms_threshold: float = 0.015
    speech_onset_threshold: float = 0.5
    speech_offset_threshold: float = 0.35
    min_speech_ms: int = 250
    silence_padding_ms: int = 300

    model_path: str | None = None
    enable_grpc: bool = True

    model_config = SettingsConfigDict(
        env_prefix="VAD_SERVICE_",
        env_file=".env",
        extra="ignore",
    )

    @property
    def frame_duration_ms(self) -> float:
        return self.frame_samples * 1000 / self.sample_rate
