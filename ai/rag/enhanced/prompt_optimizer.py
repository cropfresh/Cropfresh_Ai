"""
Prompt Optimizer
================
Context-aware prompt optimization for better LLM responses.

Features:
- Dynamic prompt construction
- Context window optimization
- Few-shot example selection
- Chain-of-thought injection
"""

from typing import Optional, Any

from loguru import logger
from pydantic import BaseModel, Field


class PromptConfig(BaseModel):
    """Prompt optimization configuration."""
    max_context_tokens: int = 4000
    include_examples: bool = True
    max_examples: int = 3
    use_chain_of_thought: bool = True
    persona_strength: float = 0.7  # How much to emphasize persona


class OptimizedPrompt(BaseModel):
    """Optimized prompt ready for LLM."""
    system_prompt: str
    user_prompt: str
    examples: list[dict] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    estimated_tokens: int = 0


# Persona templates for agricultural context
PERSONAS = {
    "agronomy": """You are an expert agronomist with deep knowledge of crop science, soil health, pest management, and sustainable farming practices. You provide scientifically accurate, practical advice tailored to Indian agricultural conditions.""",
    
    "commerce": """You are a agricultural market analyst with expertise in commodity pricing, market trends, and trading strategies. You help farmers make informed selling decisions based on current market data and historical patterns.""",
    
    "general": """You are CropFresh AI, a helpful agricultural assistant for Indian farmers. You provide accurate, practical information in a friendly manner, supporting both Hindi and English queries.""",
}

# Chain of thought templates
COT_TEMPLATES = {
    "analysis": """Let's analyze this step by step:
1. First, I'll identify the key aspects of the question
2. Then, I'll consider relevant factors and data
3. Finally, I'll provide a well-reasoned answer""",
    
    "comparison": """To compare effectively:
1. I'll list the characteristics of each option
2. Then identify key differences
3. And provide a recommendation based on the context""",
    
    "recommendation": """To provide a good recommendation:
1. I'll consider the current situation
2. Factor in relevant constraints
3. Suggest the best course of action with reasoning""",
}


class PromptOptimizer:
    """
    Optimizes prompts for better LLM responses.
    
    Usage:
        optimizer = PromptOptimizer()
        
        result = await optimizer.optimize(
            query="What tomato variety is best for Karnataka?",
            context=retrieved_docs,
            agent_type="agronomy"
        )
        
        # Use result.system_prompt and result.user_prompt
    """
    
    def __init__(
        self,
        config: Optional[PromptConfig] = None,
        example_store: Optional[dict] = None,
    ):
        """
        Initialize prompt optimizer.
        
        Args:
            config: Optimization configuration
            example_store: Store of few-shot examples
        """
        self.config = config or PromptConfig()
        self.example_store = example_store or {}
    
    async def optimize(
        self,
        query: str,
        context: Optional[list[str]] = None,
        agent_type: str = "general",
        task_type: Optional[str] = None,
    ) -> OptimizedPrompt:
        """
        Optimize a prompt for the given query.
        
        Args:
            query: User query
            context: Retrieved context documents
            agent_type: Type of agent (agronomy, commerce, general)
            task_type: Type of task (analysis, comparison, recommendation)
            
        Returns:
            OptimizedPrompt with system and user prompts
        """
        # Step 1: Build system prompt
        system_prompt = self._build_system_prompt(agent_type, task_type)
        
        # Step 2: Get relevant examples
        examples = []
        if self.config.include_examples:
            examples = self._select_examples(query, agent_type)
        
        # Step 3: Build user prompt with context
        user_prompt = self._build_user_prompt(query, context, examples)
        
        # Step 4: Optimize for context window
        system_prompt, user_prompt = self._optimize_for_tokens(
            system_prompt, user_prompt, context or []
        )
        
        # Estimate tokens
        estimated = self._estimate_tokens(system_prompt + user_prompt)
        
        result = OptimizedPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            examples=examples,
            estimated_tokens=estimated,
            metadata={
                "agent_type": agent_type,
                "task_type": task_type,
                "context_docs": len(context or []),
                "examples_included": len(examples),
            },
        )
        
        logger.debug(
            "Optimized prompt: ~{} tokens, {} examples",
            estimated, len(examples)
        )
        
        return result
    
    def _build_system_prompt(
        self,
        agent_type: str,
        task_type: Optional[str],
    ) -> str:
        """Build system prompt with persona and instructions."""
        parts = []
        
        # Add persona
        persona = PERSONAS.get(agent_type, PERSONAS["general"])
        parts.append(persona)
        
        # Add chain-of-thought if enabled
        if self.config.use_chain_of_thought and task_type:
            cot = COT_TEMPLATES.get(task_type)
            if cot:
                parts.append(f"\n{cot}")
        
        # Add general instructions
        parts.append("""
Guidelines:
- Be accurate and cite sources when available
- Use specific numbers and data when relevant
- Acknowledge uncertainty when appropriate
- Keep responses concise but complete""")
        
        return "\n".join(parts)
    
    def _build_user_prompt(
        self,
        query: str,
        context: Optional[list[str]],
        examples: list[dict],
    ) -> str:
        """Build user prompt with context and examples."""
        parts = []
        
        # Add context if available
        if context:
            parts.append("**Relevant Information:**")
            for i, doc in enumerate(context[:5], 1):  # Limit to 5 docs
                # Truncate long docs
                doc_text = doc[:500] if len(doc) > 500 else doc
                parts.append(f"{i}. {doc_text}")
            parts.append("")
        
        # Add examples if available
        if examples:
            parts.append("**Examples:**")
            for ex in examples[:self.config.max_examples]:
                parts.append(f"Q: {ex.get('query', '')}")
                parts.append(f"A: {ex.get('response', '')}")
                parts.append("")
        
        # Add the actual query
        parts.append("**Question:**")
        parts.append(query)
        
        return "\n".join(parts)
    
    def _select_examples(
        self,
        query: str,
        agent_type: str,
    ) -> list[dict]:
        """Select relevant few-shot examples."""
        # Get examples for this agent type
        agent_examples = self.example_store.get(agent_type, [])
        
        if not agent_examples:
            # Return default examples
            return self._get_default_examples(agent_type)
        
        # Simple keyword matching for relevance
        query_words = set(query.lower().split())
        scored = []
        
        for ex in agent_examples:
            ex_words = set(ex.get("query", "").lower().split())
            overlap = len(query_words & ex_words)
            scored.append((overlap, ex))
        
        # Sort by relevance
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [ex for _, ex in scored[:self.config.max_examples]]
    
    def _get_default_examples(self, agent_type: str) -> list[dict]:
        """Get default examples for agent type."""
        defaults = {
            "agronomy": [
                {
                    "query": "How to control tomato leaf curl?",
                    "response": "Tomato leaf curl is caused by whitefly-transmitted viruses. Control measures: 1) Use resistant varieties like Arka Rakshak 2) Install yellow sticky traps 3) Apply neem oil spray (5ml/L) weekly 4) Remove infected plants immediately."
                }
            ],
            "commerce": [
                {
                    "query": "Should I sell my onions now?",
                    "response": "Based on current market trends: Onion prices at ₹15-18/kg in Lasalgaon. Prices typically rise 20-30% in next 2 months due to reduced arrivals. Recommendation: Hold if storage is good, sell 30% now to cover costs."
                }
            ],
        }
        return defaults.get(agent_type, [])
    
    def _optimize_for_tokens(
        self,
        system_prompt: str,
        user_prompt: str,
        context: list[str],
    ) -> tuple[str, str]:
        """Optimize prompts to fit context window."""
        total_estimate = self._estimate_tokens(system_prompt + user_prompt)
        
        if total_estimate <= self.config.max_context_tokens:
            return system_prompt, user_prompt
        
        # Need to trim - prioritize user prompt
        excess = total_estimate - self.config.max_context_tokens
        
        # Trim context first
        if context:
            chars_to_remove = excess * 4  # Rough char/token ratio
            user_prompt = user_prompt[:-chars_to_remove] if len(user_prompt) > chars_to_remove else user_prompt
        
        return system_prompt, user_prompt
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Rough estimate: 4 chars per token
        return len(text) // 4
    
    def add_examples(self, agent_type: str, examples: list[dict]):
        """Add examples to the store."""
        if agent_type not in self.example_store:
            self.example_store[agent_type] = []
        self.example_store[agent_type].extend(examples)
