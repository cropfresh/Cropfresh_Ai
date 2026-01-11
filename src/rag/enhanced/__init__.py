"""
Enhanced RAG Module
===================
Advanced LLM-RAG integration patterns.

Components:
- Instructed Retriever: LLM-guided retrieval
- Strategy Selector: Dynamic retrieval strategy
- Bidirectional RAG: Knowledge expansion
- HiperMem: Hypergraph memory prototype
- Prompt Optimizer: Context-aware prompts
"""

from src.rag.enhanced.instructed_retriever import InstructedRetriever
from src.rag.enhanced.strategy_selector import StrategySelector, RetrievalStrategy
from src.rag.enhanced.bidirectional_rag import BidirectionalRAG
from src.rag.enhanced.prompt_optimizer import PromptOptimizer

__all__ = [
    "InstructedRetriever",
    "StrategySelector",
    "RetrievalStrategy",
    "BidirectionalRAG",
    "PromptOptimizer",
]
