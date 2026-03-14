"""
Google AMED Package
===================
Exposes the single unified client representing Google's Agricultural Landscape API.
"""

from typing import Optional

from .client import GoogleAMEDClient
from .models import (
    CropMonitoringData,
    CropStage,
    CropType,
    FieldBoundary,
    HealthStatus,
    RegionalCropStats,
    SeasonInfo,
)


# Singleton instance
_amed_client: Optional[GoogleAMEDClient] = None

def get_amed_client(
    api_key: str = "",
    project_id: str = "",
    use_mock: bool = True,
) -> GoogleAMEDClient:
    """
    Get or create singleton AMED client instance.
    
    Args:
        api_key: GCP API key
        project_id: GCP project ID
        use_mock: Use mock data
        
    Returns:
        GoogleAMEDClient instance
    """
    global _amed_client
    
    if _amed_client is None:
        _amed_client = GoogleAMEDClient(
            api_key=api_key,
            project_id=project_id,
            use_mock=use_mock,
        )
    
    return _amed_client


__all__ = [
    "GoogleAMEDClient",
    "get_amed_client",
    "CropMonitoringData",
    "CropStage",
    "CropType",
    "FieldBoundary",
    "HealthStatus",
    "RegionalCropStats",
    "SeasonInfo",
]
