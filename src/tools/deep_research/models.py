"""
Deep Research Models
====================
Data structures for the deep research tool.
"""

from pydantic import BaseModel, Field


class PageContent(BaseModel):
    """Raw content retrieved from a single web page."""
    url: str
    markdown: str = ""
    success: bool = False
    error: str = ""


class ExtractedFact(BaseModel):
    """Facts isolated from one page that are relevant to the query."""
    url: str
    facts: str = ""
    skipped: bool = False


class DeepResearchResult(BaseModel):
    """Final deep research output."""
    query: str
    answer: str
    sources: list[str] = Field(default_factory=list)
    pages_fetched: int = 0
    pages_useful: int = 0
