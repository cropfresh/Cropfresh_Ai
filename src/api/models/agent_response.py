from pydantic import BaseModel
from typing import Any

class AgentResponse(BaseModel):
    agent_type: str
    output: dict[str, Any]
    latency_ms: int | None = None
