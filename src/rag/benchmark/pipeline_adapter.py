from __future__ import annotations

from src.agents.knowledge_agent import KnowledgeAgent
from src.agents.knowledge_models import BenchmarkDebugResult


class KnowledgeAgentPipelineAdapter:
    """Adapt the product-facing KnowledgeAgent for benchmark runs."""

    def __init__(self, agent: KnowledgeAgent):
        self.agent = agent

    async def answer(self, query: str) -> BenchmarkDebugResult:
        """Query the canonical runtime path and return debug details."""
        return await self.agent.answer_with_debug(query)
