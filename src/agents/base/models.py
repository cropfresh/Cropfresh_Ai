"""
Base Agent Models
=================
Data models for the Base Agent.
"""

from typing import Optional
from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """Standard response from any agent."""
    
    # Core response
    content: str
    
    # Metadata
    agent_name: str
    confidence: float = 0.8
    
    # Sources and reasoning
    sources: list[str] = Field(default_factory=list)
    reasoning: str = ""
    
    # Execution details
    tools_used: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    
    # For follow-up
    suggested_actions: list[str] = Field(default_factory=list)
    
    # Error info
    error: Optional[str] = None


class AgentConfig(BaseModel):
    """Configuration for an agent."""
    
    name: str
    description: str
    
    # Behavior
    max_retries: int = 2
    temperature: float = 0.7
    max_tokens: int = 1000
    
    # Tool access
    tool_categories: list[str] = Field(default_factory=list)
    
    # Knowledge base
    kb_categories: list[str] = Field(default_factory=list)
    
    # System prompt template
    system_prompt: str = ""
