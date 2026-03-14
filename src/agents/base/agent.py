"""
Base Agent Core
===============
The main BaseAgent abstract class.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from loguru import logger

from src.memory.state_manager import AgentExecutionState, AgentStateManager
from src.tools.registry import ToolRegistry

from .llm import LLMMixin
from .models import AgentConfig, AgentResponse
from .retrieval import RetrievalMixin
from .tools import ToolMixin


class BaseAgent(ABC, RetrievalMixin, ToolMixin, LLMMixin):
    """
    Abstract base class for specialized agents.

    All domain agents (Agronomy, Commerce, Platform) inherit from this.

    Provides:
    - Unified interface for agent execution
    - Common tool access patterns
    - Memory integration
    - Structured response generation
    """

    def __init__(
        self,
        config: AgentConfig,
        llm=None,
        tool_registry: Optional[ToolRegistry] = None,
        state_manager: Optional[AgentStateManager] = None,
        knowledge_base=None,
    ):
        """
        Initialize base agent.
        """
        self.config = config
        self.llm = llm
        self.tools = tool_registry
        self.state_manager = state_manager
        self.knowledge_base = knowledge_base

        self._initialized = False

        logger.info(f"Agent '{config.name}' created")

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def description(self) -> str:
        return self.config.description

    async def initialize(self) -> bool:
        """
        Initialize agent resources.

        Override in subclasses for domain-specific initialization.
        """
        self._initialized = True
        logger.debug(f"Agent '{self.name}' initialized")
        return True

    @abstractmethod
    async def process(
        self,
        query: str,
        context: Optional[dict] = None,
        execution: Optional[AgentExecutionState] = None,
    ) -> AgentResponse:
        """
        Process a query and generate response.
        """
        pass

    @abstractmethod
    def _get_system_prompt(self, context: Optional[dict] = None) -> str:
        """
        Get domain-specific system prompt.
        """
        pass

    async def process_with_stream(
        self,
        query: str,
        context: Optional[dict] = None,
    ) -> AsyncIterator[str]:
        """
        Process query with streaming response.
        """
        response = await self.process(query, context)

        words = response.content.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.02)  # Slight delay for streaming effect
