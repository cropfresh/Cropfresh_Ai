from typing import Any

from pydantic import BaseModel


class AgentResponse(BaseModel):
    agent_type: str
    output: dict[str, Any]
    latency_ms: int | None = None
