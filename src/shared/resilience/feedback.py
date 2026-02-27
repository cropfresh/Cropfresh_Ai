"""
Feedback Loops
==============
Continuous improvement through feedback mechanisms.

Features:
- Query-response feedback collection
- Success/failure pattern learning
- Automatic threshold adjustment
- Improvement suggestions
"""

from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional, Any

from loguru import logger
from pydantic import BaseModel, Field


class Feedback(BaseModel):
    """Single feedback entry."""
    feedback_id: str
    query: str
    response: str
    agent_name: str
    rating: float = 0.0  # -1 to 1 (negative = bad, positive = good)
    was_helpful: bool = True
    error_type: Optional[str] = None
    user_correction: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict = Field(default_factory=dict)


class FeedbackStats(BaseModel):
    """Aggregated feedback statistics."""
    total_feedback: int = 0
    positive_count: int = 0
    negative_count: int = 0
    avg_rating: float = 0.0
    common_errors: list[str] = Field(default_factory=list)
    improvement_score: float = 0.0  # Change over time


class FeedbackLoop:
    """
    Collects and analyzes feedback for continuous improvement.
    
    Usage:
        loop = FeedbackLoop()
        
        # Record feedback
        loop.record_feedback(
            query="Price of tomatoes?",
            response="₹50/kg",
            agent_name="commerce_agent",
            was_helpful=True
        )
        
        # Get insights
        stats = loop.get_stats("commerce_agent")
        patterns = loop.get_failure_patterns()
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize feedback loop.
        
        Args:
            max_history: Maximum feedback entries to retain
        """
        self.max_history = max_history
        self._feedback: list[Feedback] = []
        self._by_agent: dict[str, list[Feedback]] = defaultdict(list)
        self._error_patterns: dict[str, int] = defaultdict(int)
    
    def record_feedback(
        self,
        query: str,
        response: str,
        agent_name: str,
        was_helpful: bool = True,
        rating: Optional[float] = None,
        error_type: Optional[str] = None,
        user_correction: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Record feedback for a query-response pair.
        
        Args:
            query: Original query
            response: Agent response
            agent_name: Which agent responded
            was_helpful: Whether response was helpful
            rating: Numeric rating (-1 to 1)
            error_type: Type of error if any
            user_correction: User's correction if provided
            metadata: Additional context
            
        Returns:
            Feedback ID
        """
        import hashlib
        
        feedback_id = hashlib.md5(
            f"{query}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        # Calculate rating from helpfulness if not provided
        if rating is None:
            rating = 1.0 if was_helpful else -0.5
        
        feedback = Feedback(
            feedback_id=feedback_id,
            query=query,
            response=response,
            agent_name=agent_name,
            rating=rating,
            was_helpful=was_helpful,
            error_type=error_type,
            user_correction=user_correction,
            metadata=metadata or {},
        )
        
        # Store feedback
        self._feedback.append(feedback)
        self._by_agent[agent_name].append(feedback)
        
        # Track error patterns
        if error_type:
            self._error_patterns[error_type] += 1
        
        # Trim history
        if len(self._feedback) > self.max_history:
            self._feedback = self._feedback[-self.max_history:]
        
        logger.debug(
            "Recorded feedback: {} (agent={}, helpful={})",
            feedback_id, agent_name, was_helpful
        )
        
        return feedback_id
    
    def get_stats(self, agent_name: Optional[str] = None) -> FeedbackStats:
        """
        Get feedback statistics.
        
        Args:
            agent_name: Filter by agent (None for all)
            
        Returns:
            FeedbackStats with aggregated metrics
        """
        feedback_list = (
            self._by_agent.get(agent_name, [])
            if agent_name else self._feedback
        )
        
        if not feedback_list:
            return FeedbackStats()
        
        positive = sum(1 for f in feedback_list if f.was_helpful)
        negative = len(feedback_list) - positive
        avg_rating = sum(f.rating for f in feedback_list) / len(feedback_list)
        
        # Get common errors
        errors = defaultdict(int)
        for f in feedback_list:
            if f.error_type:
                errors[f.error_type] += 1
        common_errors = sorted(errors.keys(), key=lambda k: errors[k], reverse=True)[:5]
        
        # Calculate improvement (compare recent vs older)
        improvement = self._calculate_improvement(feedback_list)
        
        return FeedbackStats(
            total_feedback=len(feedback_list),
            positive_count=positive,
            negative_count=negative,
            avg_rating=avg_rating,
            common_errors=common_errors,
            improvement_score=improvement,
        )
    
    def get_failure_patterns(
        self,
        min_occurrences: int = 3,
    ) -> list[dict]:
        """
        Identify common failure patterns.
        
        Args:
            min_occurrences: Minimum occurrences to be a pattern
            
        Returns:
            List of failure patterns with details
        """
        patterns = []
        
        # Group negative feedback by similar queries
        negative_feedback = [f for f in self._feedback if not f.was_helpful]
        
        # Simple keyword-based grouping
        keyword_groups = defaultdict(list)
        for f in negative_feedback:
            keywords = set(f.query.lower().split())
            for kw in keywords:
                if len(kw) > 3:  # Skip short words
                    keyword_groups[kw].append(f)
        
        for keyword, feedbacks in keyword_groups.items():
            if len(feedbacks) >= min_occurrences:
                # Check if same agent failing
                agents = [f.agent_name for f in feedbacks]
                common_agent = max(set(agents), key=agents.count)
                
                patterns.append({
                    "keyword": keyword,
                    "occurrences": len(feedbacks),
                    "primary_agent": common_agent,
                    "error_types": list(set(f.error_type for f in feedbacks if f.error_type)),
                    "sample_query": feedbacks[0].query,
                })
        
        return sorted(patterns, key=lambda p: p["occurrences"], reverse=True)
    
    def get_improvement_suggestions(
        self,
        agent_name: Optional[str] = None,
    ) -> list[str]:
        """
        Generate improvement suggestions based on feedback.
        
        Args:
            agent_name: Agent to get suggestions for
            
        Returns:
            List of suggestion strings
        """
        suggestions = []
        stats = self.get_stats(agent_name)
        
        # High negative feedback rate
        if stats.total_feedback > 10:
            negative_rate = stats.negative_count / stats.total_feedback
            if negative_rate > 0.3:
                suggestions.append(
                    f"High failure rate ({negative_rate:.0%}). Review common error patterns."
                )
        
        # Common errors
        for error in stats.common_errors[:3]:
            count = self._error_patterns.get(error, 0)
            suggestions.append(f"Frequent error: '{error}' ({count} occurrences)")
        
        # Declining performance
        if stats.improvement_score < -0.1:
            suggestions.append(
                f"Performance declining (score: {stats.improvement_score:.2f}). "
                "Review recent changes."
            )
        
        # User corrections available
        corrections = [f for f in self._feedback if f.user_correction]
        if corrections:
            suggestions.append(
                f"{len(corrections)} user corrections available for learning."
            )
        
        return suggestions
    
    def learn_from_corrections(self) -> list[dict]:
        """
        Extract learning from user corrections.
        
        Returns:
            List of learning examples
        """
        examples = []
        
        for feedback in self._feedback:
            if feedback.user_correction:
                examples.append({
                    "query": feedback.query,
                    "wrong_response": feedback.response,
                    "correct_response": feedback.user_correction,
                    "agent": feedback.agent_name,
                })
        
        return examples
    
    def _calculate_improvement(self, feedback_list: list[Feedback]) -> float:
        """Calculate improvement score over time."""
        if len(feedback_list) < 20:
            return 0.0
        
        # Compare recent vs older
        mid = len(feedback_list) // 2
        old_ratings = [f.rating for f in feedback_list[:mid]]
        new_ratings = [f.rating for f in feedback_list[mid:]]
        
        old_avg = sum(old_ratings) / len(old_ratings)
        new_avg = sum(new_ratings) / len(new_ratings)
        
        return new_avg - old_avg


# Global feedback loop instance
_global_feedback_loop: Optional[FeedbackLoop] = None


def get_feedback_loop() -> FeedbackLoop:
    """Get or create global feedback loop."""
    global _global_feedback_loop
    if _global_feedback_loop is None:
        _global_feedback_loop = FeedbackLoop()
    return _global_feedback_loop
