"""
Base Agent Tool Mixin
=====================
Tool execution capabilities for Base Agent.
"""

from typing import Optional
from loguru import logger

from src.memory.state_manager import AgentExecutionState
from src.tools.registry import ToolResult


class ToolMixin:
    """Mixin providing tool execution and formatting capabilities."""

    async def use_tool(
        self,
        tool_name: str,
        execution: Optional[AgentExecutionState] = None,
        **kwargs,
    ) -> ToolResult:
        """Execute a tool safely."""
        if not self.tools:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error="No tool registry available",
            )
        
        logger.debug(f"Agent '{self.name}' using tool: {tool_name}")
        
        result = await self.tools.execute(tool_name, **kwargs)
        
        # Track in execution state
        if execution and self.state_manager:
            execution.tool_results.append({
                "tool": tool_name,
                "success": result.success,
                "result": result.result if result.success else None,
                "error": result.error,
            })
            self.state_manager.add_step(execution.execution_id, f"tool:{tool_name}")
        
        return result

    def format_tool_results(self, results: list[dict]) -> str:
        """Format tool results for LLM context."""
        if not results:
            return ""
        
        parts = []
        for r in results:
            tool = r.get("tool", "unknown")
            if r.get("success"):
                parts.append(f"[Tool: {tool}]\n{r.get('result')}")
            else:
                parts.append(f"[Tool: {tool}] Error: {r.get('error')}")
        
        return "\n\n".join(parts)
