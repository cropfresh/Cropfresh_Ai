"""
Reflection Engine
=================
Agent self-correction via output analysis.

Implements the Reflection pattern:
1. Generate initial response
2. Analyze output for errors/issues
3. Self-correct if problems found
4. Return improved response
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Any, Callable

from loguru import logger
from pydantic import BaseModel, Field


class IssueType(str, Enum):
    """Types of issues detected in output."""
    FACTUAL_ERROR = "factual_error"
    INCOMPLETE = "incomplete"
    INCONSISTENT = "inconsistent"
    OFF_TOPIC = "off_topic"
    HALLUCINATION = "hallucination"
    FORMAT_ERROR = "format_error"
    CONFIDENCE_LOW = "confidence_low"


class Issue(BaseModel):
    """Detected issue in agent output."""
    issue_type: IssueType
    description: str
    severity: float = 0.5  # 0-1
    suggested_fix: Optional[str] = None


class ReflectionResult(BaseModel):
    """Result of reflection analysis."""
    original_output: str
    corrected_output: Optional[str] = None
    issues_found: list[Issue] = Field(default_factory=list)
    was_corrected: bool = False
    iterations: int = 1
    confidence_before: float = 0.0
    confidence_after: float = 0.0
    reflection_time_ms: float = 0.0


REFLECTION_PROMPT = """Analyze the following response for potential issues:

Original Query: {query}
Response: {response}

Check for:
1. Factual errors or inaccuracies
2. Incomplete information (missing key points)
3. Inconsistencies within the response
4. Off-topic content
5. Potential hallucinations (claims without basis)
6. Formatting issues

If issues are found, provide corrections. If the response is good, confirm it.

Respond with JSON:
{{
    "issues": [
        {{"type": "factual_error|incomplete|...", "description": "...", "severity": 0-1}}
    ],
    "corrected_response": "improved response if needed, or null",
    "overall_quality": 0-1
}}"""


class ReflectionEngine:
    """
    Implements agent self-reflection and correction.
    
    Usage:
        reflector = ReflectionEngine(llm=groq_llm)
        
        result = await reflector.reflect(
            query="What is the price of tomatoes?",
            response="Tomatoes cost $100 per kg.",  # Error!
        )
        
        if result.was_corrected:
            print(result.corrected_output)
    """
    
    def __init__(
        self,
        llm=None,
        max_iterations: int = 2,
        min_confidence: float = 0.7,
    ):
        """
        Initialize reflection engine.
        
        Args:
            llm: LLM for reflection analysis
            max_iterations: Maximum correction iterations
            min_confidence: Minimum quality threshold
        """
        self.llm = llm
        self.max_iterations = max_iterations
        self.min_confidence = min_confidence
    
    async def reflect(
        self,
        query: str,
        response: str,
        context: Optional[dict] = None,
    ) -> ReflectionResult:
        """
        Reflect on and potentially correct a response.
        
        Args:
            query: Original query
            response: Agent's response
            context: Optional context
            
        Returns:
            ReflectionResult with analysis and corrections
        """
        start_time = datetime.now()
        
        result = ReflectionResult(
            original_output=response,
            confidence_before=self._estimate_confidence(response),
        )
        
        current_response = response
        
        for iteration in range(self.max_iterations):
            # Analyze current response
            analysis = await self._analyze(query, current_response, context)
            
            if not analysis["issues"]:
                # No issues found
                result.confidence_after = analysis.get("overall_quality", 0.8)
                break
            
            # Record issues
            for issue_data in analysis["issues"]:
                result.issues_found.append(Issue(
                    issue_type=self._parse_issue_type(issue_data.get("type", "")),
                    description=issue_data.get("description", ""),
                    severity=issue_data.get("severity", 0.5),
                ))
            
            # Apply correction if provided
            if analysis.get("corrected_response"):
                current_response = analysis["corrected_response"]
                result.was_corrected = True
                result.corrected_output = current_response
            
            result.iterations = iteration + 1
            result.confidence_after = analysis.get("overall_quality", 0.5)
            
            # Stop if quality is good enough
            if result.confidence_after >= self.min_confidence:
                break
        
        result.reflection_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(
            "Reflection: {} issues, corrected={}, conf: {:.2f} -> {:.2f}",
            len(result.issues_found),
            result.was_corrected,
            result.confidence_before,
            result.confidence_after,
        )
        
        return result
    
    async def _analyze(
        self,
        query: str,
        response: str,
        context: Optional[dict],
    ) -> dict:
        """Analyze response for issues."""
        if not self.llm:
            # Fallback: simple heuristic analysis
            return self._heuristic_analysis(response)
        
        try:
            import json
            
            prompt = REFLECTION_PROMPT.format(query=query, response=response)
            
            llm_response = await self.llm.agenerate(prompt)
            
            # Parse JSON
            response_text = llm_response.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            return json.loads(response_text)
            
        except Exception as e:
            logger.warning("Reflection analysis failed: {}", str(e))
            return self._heuristic_analysis(response)
    
    def _heuristic_analysis(self, response: str) -> dict:
        """Simple heuristic-based analysis."""
        issues = []
        quality = 0.7
        
        # Check for common issues
        if len(response) < 50:
            issues.append({
                "type": "incomplete",
                "description": "Response is very short",
                "severity": 0.4,
            })
            quality -= 0.1
        
        # Check for uncertainty markers
        uncertainty_phrases = ["i don't know", "i'm not sure", "might be", "possibly"]
        if any(phrase in response.lower() for phrase in uncertainty_phrases):
            issues.append({
                "type": "confidence_low",
                "description": "Response contains uncertainty markers",
                "severity": 0.3,
            })
            quality -= 0.1
        
        # Check for missing structure
        if len(response) > 500 and not any(c in response for c in ['-', '•', '1.', '*']):
            issues.append({
                "type": "format_error",
                "description": "Long response without structure",
                "severity": 0.2,
            })
        
        return {
            "issues": issues,
            "corrected_response": None,
            "overall_quality": max(0.3, quality),
        }
    
    def _estimate_confidence(self, response: str) -> float:
        """Estimate initial confidence from response."""
        score = 0.6  # Base
        
        if len(response) > 100:
            score += 0.1
        if len(response) > 300:
            score += 0.1
        
        # Has specific data
        if any(c.isdigit() for c in response):
            score += 0.05
        
        # Has structure
        if any(c in response for c in ['-', '•', '1.']):
            score += 0.05
        
        return min(0.9, score)
    
    def _parse_issue_type(self, type_str: str) -> IssueType:
        """Parse issue type string."""
        mapping = {
            "factual_error": IssueType.FACTUAL_ERROR,
            "incomplete": IssueType.INCOMPLETE,
            "inconsistent": IssueType.INCONSISTENT,
            "off_topic": IssueType.OFF_TOPIC,
            "hallucination": IssueType.HALLUCINATION,
            "format_error": IssueType.FORMAT_ERROR,
            "confidence_low": IssueType.CONFIDENCE_LOW,
        }
        return mapping.get(type_str.lower(), IssueType.INCOMPLETE)
