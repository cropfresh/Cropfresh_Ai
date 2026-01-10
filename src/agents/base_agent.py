"""
Base Agent
==========
Abstract base class for all specialized agents in the multi-agent RAG system.

Provides:
- Common tool access patterns
- Memory integration
- Response formatting
- Error handling with retries
- Streaming support

Author: CropFresh AI Team
Version: 2.0.0
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Optional

from loguru import logger
from pydantic import BaseModel, Field

from src.memory.state_manager import AgentExecutionState, AgentStateManager, Message
from src.tools.registry import ToolRegistry, ToolResult


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


class BaseAgent(ABC):
    """
    Abstract base class for specialized agents.
    
    All domain agents (Agronomy, Commerce, Platform) inherit from this.
    
    Provides:
    - Unified interface for agent execution
    - Common tool access patterns
    - Memory integration
    - Structured response generation
    
    Subclasses must implement:
    - process(): Main agent logic
    - _get_system_prompt(): Domain-specific system prompt
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
        
        Args:
            config: Agent configuration
            llm: LLM provider
            tool_registry: Tool registry for tool access
            state_manager: State manager for memory
            knowledge_base: Knowledge base for retrieval
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
        """Agent name."""
        return self.config.name
    
    @property
    def description(self) -> str:
        """Agent description."""
        return self.config.description
    
    async def initialize(self) -> bool:
        """
        Initialize agent resources.
        
        Override in subclasses for domain-specific initialization.
        
        Returns:
            True if successful
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
        
        Must be implemented by subclasses.
        
        Args:
            query: User query
            context: Optional additional context
            execution: Optional execution state for tracking
            
        Returns:
            AgentResponse
        """
        pass
    
    @abstractmethod
    def _get_system_prompt(self, context: Optional[dict] = None) -> str:
        """
        Get domain-specific system prompt.
        
        Must be implemented by subclasses.
        
        Args:
            context: Optional context for dynamic prompts
            
        Returns:
            System prompt string
        """
        pass
    
    async def process_with_stream(
        self,
        query: str,
        context: Optional[dict] = None,
    ) -> AsyncIterator[str]:
        """
        Process query with streaming response.
        
        Args:
            query: User query
            context: Optional context
            
        Yields:
            Response tokens
        """
        # Default implementation - subclasses can override for true streaming
        response = await self.process(query, context)
        
        # Simulate streaming by yielding chunks
        words = response.content.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.02)  # Slight delay for streaming effect
    
    async def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        categories: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Retrieve relevant documents from knowledge base.
        
        Args:
            query: Search query
            top_k: Number of documents
            categories: Optional category filter
            
        Returns:
            List of document dicts
        """
        if not self.knowledge_base:
            return []
        
        try:
            # Use configured categories if not specified
            search_categories = categories or self.config.kb_categories
            
            result = await self.knowledge_base.search(
                query=query,
                top_k=top_k,
                category=search_categories[0] if search_categories else None,
            )
            
            return [
                {
                    "text": doc.text,
                    "source": doc.source,
                    "category": doc.category,
                    "score": doc.score,
                }
                for doc in result.documents
            ]
            
        except Exception as e:
            logger.warning(f"Retrieval failed: {e}")
            return []
    
    async def use_tool(
        self,
        tool_name: str,
        execution: Optional[AgentExecutionState] = None,
        **kwargs,
    ) -> ToolResult:
        """
        Execute a tool safely.
        
        Args:
            tool_name: Name of tool to execute
            execution: Optional execution state for tracking
            **kwargs: Tool arguments
            
        Returns:
            ToolResult
        """
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
    
    async def generate_with_llm(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate response using LLM.
        
        Args:
            messages: List of message dicts with role/content
            temperature: Override temperature
            max_tokens: Override max tokens
            
        Returns:
            Generated content
        """
        if not self.llm:
            raise ValueError("No LLM configured for agent")
        
        from src.orchestrator.llm_provider import LLMMessage
        
        llm_messages = [
            LLMMessage(role=m["role"], content=m["content"])
            for m in messages
        ]
        
        response = await self.llm.generate(
            llm_messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
        )
        
        return response.content
    
    def format_context(self, documents: list[dict]) -> str:
        """
        Format retrieved documents for LLM context.
        
        Args:
            documents: List of document dicts
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."
        
        parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.get("source", "Unknown")
            text = doc.get("text", "")
            score = doc.get("score", 0)
            
            parts.append(f"[Document {i}] (Source: {source}, Relevance: {score:.2f})\n{text}")
        
        return "\n\n".join(parts)
    
    def format_tool_results(self, results: list[dict]) -> str:
        """
        Format tool results for LLM context.
        
        Args:
            results: List of tool result dicts
            
        Returns:
            Formatted results string
        """
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
    
    async def _retry_operation(
        self,
        operation,
        *args,
        max_retries: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """
        Retry an async operation with exponential backoff.
        
        Args:
            operation: Async function to retry
            *args: Positional arguments
            max_retries: Override max retries
            **kwargs: Keyword arguments
            
        Returns:
            Operation result
        """
        retries = max_retries or self.config.max_retries
        last_error = None
        
        for attempt in range(retries + 1):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < retries:
                    wait = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Retry {attempt + 1}/{retries} after {wait}s: {e}")
                    await asyncio.sleep(wait)
        
        raise last_error
    
    def _extract_sources(self, documents: list[dict]) -> list[str]:
        """Extract unique source references from documents."""
        sources = []
        seen = set()
        
        for doc in documents:
            source = doc.get("source", "")
            if source and source not in seen:
                sources.append(source)
                seen.add(source)
        
        return sources
