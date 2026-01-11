"""
LangSmith Observability
=======================
Production tracing and debugging for RAG pipeline.

Features:
- LangSmith tracing integration
- Custom span decorators
- Metadata logging
- Evaluation dataset management
"""

import functools
import os
from contextlib import contextmanager
from typing import Any, Callable, Optional

from loguru import logger


# Check if LangSmith is available
def _check_langsmith_available() -> bool:
    """Check if LangSmith SDK is installed and configured."""
    try:
        import langsmith  # noqa
        return bool(os.getenv("LANGSMITH_API_KEY"))
    except ImportError:
        return False


LANGSMITH_AVAILABLE = _check_langsmith_available()


def configure_langsmith(
    api_key: Optional[str] = None,
    project_name: str = "cropfresh-ai",
    tracing_enabled: bool = True,
) -> bool:
    """
    Configure LangSmith for tracing.
    
    Args:
        api_key: LangSmith API key (reads from env if not provided)
        project_name: Project name in LangSmith
        tracing_enabled: Enable/disable tracing
        
    Returns:
        True if configured successfully
    """
    if not LANGSMITH_AVAILABLE:
        logger.warning("LangSmith not available. Install with: pip install langsmith")
        return False
    
    try:
        # Set environment variables
        if api_key:
            os.environ["LANGSMITH_API_KEY"] = api_key
        
        os.environ["LANGSMITH_PROJECT"] = project_name
        os.environ["LANGSMITH_TRACING_V2"] = "true" if tracing_enabled else "false"
        
        # Verify connection
        from langsmith import Client
        client = Client()
        
        # Quick health check
        client.list_projects(limit=1)
        
        logger.info(f"LangSmith configured for project: {project_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to configure LangSmith: {e}")
        return False


def trace_rag(
    name: Optional[str] = None,
    run_type: str = "chain",
    metadata: Optional[dict] = None,
):
    """
    Decorator to trace RAG operations in LangSmith.
    
    Args:
        name: Span name (defaults to function name)
        run_type: Type of run ("chain", "llm", "retriever", "tool")
        metadata: Additional metadata to log
    
    Usage:
        @trace_rag("knowledge_search", run_type="retriever")
        async def search_knowledge(query: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        if not LANGSMITH_AVAILABLE:
            return func
        
        try:
            from langsmith import traceable
            
            trace_name = name or func.__name__
            
            @functools.wraps(func)
            @traceable(name=trace_name, run_type=run_type, metadata=metadata or {})
            async def async_wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
            
            @functools.wraps(func)
            @traceable(name=trace_name, run_type=run_type, metadata=metadata or {})
            def sync_wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            # Check if async
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
            
        except ImportError:
            return func
    
    return decorator


@contextmanager
def trace_span(
    name: str,
    inputs: Optional[dict] = None,
    metadata: Optional[dict] = None,
):
    """
    Context manager for manual span creation.
    
    Usage:
        with trace_span("custom_operation", inputs={"query": q}):
            # ... operation code
    """
    if not LANGSMITH_AVAILABLE:
        yield
        return
    
    try:
        from langsmith.run_helpers import trace
        
        with trace(name=name, inputs=inputs or {}, metadata=metadata or {}):
            yield
            
    except ImportError:
        yield


def log_feedback(
    run_id: str,
    key: str,
    score: float,
    comment: Optional[str] = None,
) -> bool:
    """
    Log feedback for a specific run.
    
    Args:
        run_id: LangSmith run ID
        key: Feedback key (e.g., "relevance", "correctness")
        score: Score value (0-1)
        comment: Optional comment
        
    Returns:
        True if feedback logged successfully
    """
    if not LANGSMITH_AVAILABLE:
        return False
    
    try:
        from langsmith import Client
        
        client = Client()
        client.create_feedback(
            run_id=run_id,
            key=key,
            score=score,
            comment=comment,
        )
        return True
        
    except Exception as e:
        logger.error(f"Failed to log feedback: {e}")
        return False


class RAGEvaluator:
    """
    Evaluation helper for RAG quality metrics.
    
    Supports:
    - Relevance scoring
    - Faithfulness checking
    - Answer correctness
    """
    
    def __init__(self, dataset_name: str = "cropfresh-eval"):
        """
        Initialize evaluator.
        
        Args:
            dataset_name: Name for evaluation dataset
        """
        self.dataset_name = dataset_name
        self._client = None
    
    @property
    def client(self):
        """Lazy load LangSmith client."""
        if self._client is None and LANGSMITH_AVAILABLE:
            from langsmith import Client
            self._client = Client()
        return self._client
    
    def create_dataset(
        self,
        examples: list[dict[str, Any]],
        description: str = "CropFresh RAG evaluation dataset",
    ) -> Optional[str]:
        """
        Create evaluation dataset in LangSmith.
        
        Args:
            examples: List of {"input": ..., "expected_output": ...}
            description: Dataset description
            
        Returns:
            Dataset ID if created
        """
        if not self.client:
            logger.warning("LangSmith not available")
            return None
        
        try:
            # Create dataset
            dataset = self.client.create_dataset(
                dataset_name=self.dataset_name,
                description=description,
            )
            
            # Add examples
            for example in examples:
                self.client.create_example(
                    inputs=example.get("input", {}),
                    outputs=example.get("expected_output", {}),
                    dataset_id=dataset.id,
                )
            
            logger.info(f"Created dataset '{self.dataset_name}' with {len(examples)} examples")
            return dataset.id
            
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            return None
    
    def evaluate_response(
        self,
        query: str,
        response: str,
        retrieved_docs: list[str],
        expected: Optional[str] = None,
    ) -> dict[str, float]:
        """
        Evaluate a RAG response locally (without LangSmith).
        
        Simple heuristic evaluation:
        - Relevance: Query term overlap with response
        - Grounding: Response overlap with retrieved docs
        - Completeness: Response length adequacy
        
        Args:
            query: User query
            response: Generated response
            retrieved_docs: Retrieved document texts
            expected: Expected answer (optional)
            
        Returns:
            Dict of metric scores
        """
        import re
        
        def tokenize(text: str) -> set[str]:
            return set(re.sub(r'[^\w\s]', '', text.lower()).split())
        
        query_tokens = tokenize(query)
        response_tokens = tokenize(response)
        
        # Relevance: query terms in response
        if query_tokens:
            relevance = len(query_tokens & response_tokens) / len(query_tokens)
        else:
            relevance = 0.0
        
        # Grounding: response terms from docs
        doc_tokens = set()
        for doc in retrieved_docs:
            doc_tokens.update(tokenize(doc))
        
        if response_tokens and doc_tokens:
            grounding = len(response_tokens & doc_tokens) / len(response_tokens)
        else:
            grounding = 0.0
        
        # Completeness: adequate response length
        word_count = len(response.split())
        if word_count < 20:
            completeness = 0.3
        elif word_count < 50:
            completeness = 0.6
        elif word_count < 200:
            completeness = 1.0
        else:
            completeness = 0.8  # Too verbose
        
        # Correctness (if expected provided)
        correctness = None
        if expected:
            expected_tokens = tokenize(expected)
            if expected_tokens:
                correctness = len(expected_tokens & response_tokens) / len(expected_tokens)
        
        result = {
            "relevance": round(relevance, 3),
            "grounding": round(grounding, 3),
            "completeness": round(completeness, 3),
        }
        
        if correctness is not None:
            result["correctness"] = round(correctness, 3)
        
        return result


# Sample evaluation dataset for CropFresh
SAMPLE_EVAL_DATASET = [
    {
        "input": {"query": "How to grow tomatoes in Karnataka?"},
        "expected_output": {
            "should_contain": ["soil", "water", "season", "variety"],
            "category": "agronomy",
        },
    },
    {
        "input": {"query": "What is the current price of onions?"},
        "expected_output": {
            "should_contain": ["price", "per kg", "mandi"],
            "category": "market",
        },
    },
    {
        "input": {"query": "How do I register on CropFresh?"},
        "expected_output": {
            "should_contain": ["register", "phone", "app"],
            "category": "platform",
        },
    },
    {
        "input": {"query": "What farmers grow tomatoes in Kolar?"},
        "expected_output": {
            "should_contain": ["farmer", "Kolar", "tomato"],
            "category": "graph",
        },
    },
]
