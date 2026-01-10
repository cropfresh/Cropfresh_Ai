"""
Tool Registry
=============
Dynamic tool registration and management for agentic RAG system.

Provides:
- Tool registration and discovery
- Tool schema generation for LLM
- Safe tool execution with validation
- Async tool support

Author: CropFresh AI Team
Version: 2.0.0
"""

import asyncio
import inspect
from typing import Any, Callable, Optional, get_type_hints

from loguru import logger
from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """Single tool parameter definition."""
    
    name: str
    type: str
    description: str
    required: bool = True
    default: Optional[Any] = None


class ToolDefinition(BaseModel):
    """Complete tool definition for LLM."""
    
    name: str
    description: str
    parameters: list[ToolParameter] = Field(default_factory=list)
    returns: str = "str"
    category: str = "general"
    is_async: bool = False


class ToolResult(BaseModel):
    """Result of tool execution."""
    
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0


class ToolRegistry:
    """
    Dynamic tool registry for agentic system.
    
    Features:
    - Register tools with metadata
    - Generate OpenAI-compatible tool schemas
    - Execute tools safely with error handling
    - Categorize tools by domain
    
    Usage:
        registry = ToolRegistry()
        
        @registry.register("weather", "Get weather forecast")
        async def get_weather(location: str, days: int = 7) -> dict:
            ...
        
        # Get tools for LLM
        tools = registry.get_tools_for_llm()
        
        # Execute tool
        result = await registry.execute("get_weather", location="Kolar", days=3)
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._tools: dict[str, Callable] = {}
        self._definitions: dict[str, ToolDefinition] = {}
        
        logger.info("ToolRegistry initialized")
    
    def register(
        self,
        name: Optional[str] = None,
        description: str = "",
        category: str = "general",
    ):
        """
        Decorator to register a tool.
        
        Args:
            name: Tool name (defaults to function name)
            description: Tool description
            category: Tool category (agronomy, commerce, platform, etc.)
            
        Usage:
            @registry.register("get_price", "Get current market price")
            async def get_price(commodity: str, location: str) -> dict:
                ...
        """
        def decorator(func: Callable):
            tool_name = name or func.__name__
            
            # Extract parameters from function signature
            params = []
            sig = inspect.signature(func)
            hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
            
            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue
                
                param_type = hints.get(param_name, Any).__name__ if param_name in hints else "any"
                
                # Get default value
                has_default = param.default is not inspect.Parameter.empty
                default_val = param.default if has_default else None
                
                params.append(ToolParameter(
                    name=param_name,
                    type=param_type,
                    description=f"Parameter: {param_name}",
                    required=not has_default,
                    default=default_val,
                ))
            
            # Determine return type
            return_type = hints.get("return", Any)
            return_type_name = return_type.__name__ if hasattr(return_type, "__name__") else "any"
            
            # Create definition
            definition = ToolDefinition(
                name=tool_name,
                description=description or func.__doc__ or f"Tool: {tool_name}",
                parameters=params,
                returns=return_type_name,
                category=category,
                is_async=asyncio.iscoroutinefunction(func),
            )
            
            # Store
            self._tools[tool_name] = func
            self._definitions[tool_name] = definition
            
            logger.debug(f"Registered tool: {tool_name} (category: {category})")
            
            return func
        
        return decorator
    
    def add_tool(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: str = "",
        category: str = "general",
    ) -> None:
        """
        Programmatically add a tool.
        
        Args:
            func: Tool function
            name: Tool name
            description: Tool description
            category: Tool category
        """
        decorator = self.register(name, description, category)
        decorator(func)
    
    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_definition(self, name: str) -> Optional[ToolDefinition]:
        """Get tool definition by name."""
        return self._definitions.get(name)
    
    def list_tools(self, category: Optional[str] = None) -> list[str]:
        """
        List available tool names.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of tool names
        """
        if category:
            return [
                name for name, defn in self._definitions.items()
                if defn.category == category
            ]
        return list(self._tools.keys())
    
    def get_tools_for_llm(
        self,
        categories: Optional[list[str]] = None,
        format: str = "openai",
    ) -> list[dict]:
        """
        Generate tool schemas for LLM.
        
        Args:
            categories: Optional category filter
            format: Schema format ('openai', 'anthropic')
            
        Returns:
            List of tool schemas
        """
        schemas = []
        
        for name, defn in self._definitions.items():
            if categories and defn.category not in categories:
                continue
            
            if format == "openai":
                # OpenAI function calling format
                properties = {}
                required = []
                
                for param in defn.parameters:
                    properties[param.name] = {
                        "type": self._map_type(param.type),
                        "description": param.description,
                    }
                    if param.required:
                        required.append(param.name)
                
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": defn.name,
                        "description": defn.description,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                })
            
            elif format == "anthropic":
                # Anthropic tool format
                input_schema = {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }
                
                for param in defn.parameters:
                    input_schema["properties"][param.name] = {
                        "type": self._map_type(param.type),
                        "description": param.description,
                    }
                    if param.required:
                        input_schema["required"].append(param.name)
                
                schemas.append({
                    "name": defn.name,
                    "description": defn.description,
                    "input_schema": input_schema,
                })
        
        return schemas
    
    def _map_type(self, py_type: str) -> str:
        """Map Python type to JSON schema type."""
        mapping = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
            "NoneType": "null",
        }
        return mapping.get(py_type, "string")
    
    async def execute(
        self,
        tool_name: str,
        **kwargs,
    ) -> ToolResult:
        """
        Execute a tool safely.
        
        Args:
            tool_name: Name of tool to execute
            **kwargs: Tool arguments
            
        Returns:
            ToolResult with success/error
        """
        import time
        
        start = time.time()
        
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool not found: {tool_name}",
            )
        
        definition = self._definitions.get(tool_name)
        
        try:
            # Validate required parameters
            if definition:
                for param in definition.parameters:
                    if param.required and param.name not in kwargs:
                        return ToolResult(
                            tool_name=tool_name,
                            success=False,
                            error=f"Missing required parameter: {param.name}",
                        )
            
            # Execute
            if definition and definition.is_async:
                result = await tool(**kwargs)
            else:
                result = tool(**kwargs)
            
            execution_time = (time.time() - start) * 1000
            
            logger.debug(f"Tool {tool_name} executed in {execution_time:.1f}ms")
            
            return ToolResult(
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time_ms=execution_time,
            )
            
        except Exception as e:
            execution_time = (time.time() - start) * 1000
            logger.error(f"Tool {tool_name} failed: {e}")
            
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
            )
    
    def get_tool_prompt(self, categories: Optional[list[str]] = None) -> str:
        """
        Generate a text description of available tools for LLM prompt.
        
        Args:
            categories: Optional category filter
            
        Returns:
            Formatted tool descriptions
        """
        lines = ["Available tools:"]
        
        for name, defn in self._definitions.items():
            if categories and defn.category not in categories:
                continue
            
            params = ", ".join([
                f"{p.name}: {p.type}" + ("" if p.required else f" = {p.default}")
                for p in defn.parameters
            ])
            
            lines.append(f"\n- **{defn.name}**({params})")
            lines.append(f"  {defn.description}")
        
        return "\n".join(lines)


# Global tool registry
_global_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get or create global tool registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry
