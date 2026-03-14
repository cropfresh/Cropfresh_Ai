"""
Agronomy Helpers
================
Pure-function utilities used by AgronomyAgent for parsing LLM output
and computing confidence scores. No side effects, no I/O.

Author: CropFresh AI Team
Version: 3.0.0
"""

import re

# * Regex to locate the "Follow-up Questions" section in structured output
# Matches the clipboard emoji and any translated section header until the end of line
_FOLLOWUP_SECTION_RE = re.compile(
    r"(?:📋[^\n]*|follow[\s-]*up\s+questions?)[:\s]*\n"
    r"((?:\s*[-•*]\s*.+\n?)+)",
    re.IGNORECASE | re.MULTILINE,
)
_FOLLOWUP_BULLET_RE = re.compile(r"[-•*]\s*(.+)")


def parse_follow_ups(llm_output: str) -> list[str]:
    """
    Extract follow-up questions from the LLM's structured output.

    The LLM is instructed to produce a '📋 Follow-up Questions' section
    with bullet points in the user's language. We parse those bullets
    so they can be surfaced in AgentResponse.suggested_actions.

    Returns:
        Up to 3 cleaned follow-up question strings.
    """
    section_match = _FOLLOWUP_SECTION_RE.search(llm_output)
    if not section_match:
        return []

    bullets_text = section_match.group(1)
    follow_ups = _FOLLOWUP_BULLET_RE.findall(bullets_text)
    return [q.strip().rstrip("?").strip() + "?" for q in follow_ups[:3]]  # type: ignore[index]


def compute_confidence(documents: list[dict], has_tool_data: bool) -> float:
    """
    Compute response confidence from RAG relevance scores.

    Scoring:
    - Base: 0.4 (general knowledge only, no docs)
    - +0.0-0.4 from average document relevance score
    - +0.1 bonus if real-time tool data was used
    - Capped at 0.95
    """
    if not documents:
        return 0.4

    avg = avg_score(documents)
    doc_contribution = min(avg * 0.4, 0.4)

    confidence = 0.5 + doc_contribution
    if has_tool_data:
        confidence += 0.1

    return min(confidence, 0.95)


def avg_score(documents: list[dict]) -> float:
    """Average relevance score of retrieved documents."""
    if not documents:
        return 0.0
    return sum(d.get("score", 0) for d in documents) / len(documents)
