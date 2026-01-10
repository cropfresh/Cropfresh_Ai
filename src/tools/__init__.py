"""
Tools module for Advanced Agentic RAG System.

Provides:
- Tool Registry for dynamic tool management
- Agmarknet API for market prices
- Weather API for agricultural planning
- Calculator for AISP and yield estimates
- Web Search for real-time information

Author: CropFresh AI Team
Version: 2.0.0
"""

from src.tools.registry import (
    ToolDefinition,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    get_tool_registry,
)
from src.tools.agmarknet import AgmarknetPrice, AgmarknetTool

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
]
