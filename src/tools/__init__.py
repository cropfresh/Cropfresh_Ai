"""
Tools module for Advanced Agentic RAG System.

Provides:
- Tool Registry for dynamic tool management
- Agmarknet API for market prices
- eNAM API for live mandi prices
- IMD Weather API for agricultural weather
- Google AMED for crop monitoring
- Real-Time Data Manager for unified access
- Calculator for AISP and yield estimates
- Web Search for real-time information

Author: CropFresh AI Team
Version: 3.0.0
"""

from src.tools.registry import (
    ToolDefinition,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    get_tool_registry,
)
from src.tools.agmarknet import AgmarknetPrice, AgmarknetTool

# New real-time data clients
from src.tools.enam_client import ENAMClient, MandiPrice, PriceTrend, get_enam_client
from src.tools.imd_weather import (
    IMDWeatherClient,
    CurrentWeather,
    WeatherForecast,
    AgroAdvisory,
    get_imd_client,
)
from src.tools.google_amed import (
    GoogleAMEDClient,
    CropMonitoringData,
    SeasonInfo,
    get_amed_client,
)
from src.tools.realtime_data import (
    RealTimeDataManager,
    RealTimeData,
    DataFreshness,
    get_realtime_data_manager,
)

# Import tools to trigger auto-registration
from src.tools import weather
from src.tools import calculator
from src.tools import web_search

__all__ = [
    # Registry
    "ToolRegistry",
    "ToolDefinition",
    "ToolParameter",
    "ToolResult",
    "get_tool_registry",
    # Agmarknet
    "AgmarknetTool",
    "AgmarknetPrice",
    # eNAM (NEW)
    "ENAMClient",
    "MandiPrice",
    "PriceTrend",
    "get_enam_client",
    # IMD Weather (NEW)
    "IMDWeatherClient",
    "CurrentWeather",
    "WeatherForecast",
    "AgroAdvisory",
    "get_imd_client",
    # Google AMED (NEW)
    "GoogleAMEDClient",
    "CropMonitoringData",
    "SeasonInfo",
    "get_amed_client",
    # Real-Time Data Manager (NEW)
    "RealTimeDataManager",
    "RealTimeData",
    "DataFreshness",
    "get_realtime_data_manager",
]

