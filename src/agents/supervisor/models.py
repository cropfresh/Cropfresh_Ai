"""
Models for the Supervisor Agent.
"""

from pydantic import BaseModel, Field

class RoutingDecision(BaseModel):
    """Decision about which agent to route to."""
    
    agent_name: str
    confidence: float
    reasoning: str
    requires_multiple: bool = False
    secondary_agents: list[str] = Field(default_factory=list)
