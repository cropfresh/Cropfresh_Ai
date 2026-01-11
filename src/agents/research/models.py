"""
Research Data Models
====================
Pydantic models for research agent.
"""

from datetime import date, datetime
from enum import Enum
from typing import Optional, Any

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Types of information sources."""
    WEB = "web"
    KNOWLEDGE_BASE = "kb"
    GRAPH = "graph"
    API = "api"
    RSS = "rss"


class StepStatus(str, Enum):
    """Status of research step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Citation(BaseModel):
    """Academic-style citation."""
    ref_id: str = ""  # [1], [2], etc.
    title: str
    url: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    date_published: Optional[date] = None
    date_accessed: date = Field(default_factory=date.today)
    source_type: SourceType = SourceType.WEB
    
    def to_apa(self) -> str:
        """Format as APA citation."""
        parts = []
        if self.author:
            parts.append(f"{self.author}.")
        if self.date_published:
            parts.append(f"({self.date_published.year}).")
        parts.append(f"{self.title}.")
        if self.publisher:
            parts.append(f"{self.publisher}.")
        if self.url:
            parts.append(f"Retrieved from {self.url}")
        return " ".join(parts)
    
    def to_inline(self) -> str:
        """Format as inline citation marker."""
        return f"[{self.ref_id}]"


class Finding(BaseModel):
    """A single research finding from a source."""
    content: str
    summary: str = ""
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    source_type: SourceType = SourceType.WEB
    reliability_score: float = 0.5  # 0.0-1.0
    relevance_score: float = 0.5  # 0.0-1.0
    timestamp: datetime = Field(default_factory=datetime.now)
    citation: Optional[Citation] = None
    metadata: dict = Field(default_factory=dict)


class ResearchStep(BaseModel):
    """A single step in a research plan."""
    step_id: str
    question: str
    description: str = ""
    source_types: list[SourceType] = Field(default_factory=lambda: [SourceType.WEB, SourceType.KNOWLEDGE_BASE])
    depends_on: list[str] = Field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    findings: list[Finding] = Field(default_factory=list)
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ResearchPlan(BaseModel):
    """Multi-step research plan."""
    plan_id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4())[:8])
    original_query: str = ""
    refined_query: str = ""
    objective: str = ""
    sub_questions: list[str] = Field(default_factory=list)
    steps: list[ResearchStep] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    estimated_time_sec: int = 60
    
    def get_next_step(self) -> Optional[ResearchStep]:
        """Get the next pending step that has all dependencies met."""
        completed_ids = {s.step_id for s in self.steps if s.status == StepStatus.COMPLETED}
        
        for step in self.steps:
            if step.status == StepStatus.PENDING:
                deps_met = all(dep in completed_ids for dep in step.depends_on)
                if deps_met:
                    return step
        return None
    
    @property
    def progress(self) -> float:
        """Get completion percentage."""
        if not self.steps:
            return 0.0
        completed = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        return completed / len(self.steps)
    
    @property
    def all_findings(self) -> list[Finding]:
        """Get all findings from all steps."""
        findings = []
        for step in self.steps:
            findings.extend(step.findings)
        return findings


class ResearchReport(BaseModel):
    """Final research report with citations."""
    title: str
    query: str
    summary: str
    sections: list[dict] = Field(default_factory=list)  # [{title, content}]
    findings: list[Finding] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    confidence_score: float = 0.0
    total_sources: int = 0
    
    def to_markdown(self) -> str:
        """Convert report to markdown format."""
        md = []
        md.append(f"# {self.title}\n")
        md.append(f"*Generated: {self.created_at.strftime('%Y-%m-%d %H:%M')}*\n")
        md.append(f"\n## Summary\n{self.summary}\n")
        
        for section in self.sections:
            md.append(f"\n## {section.get('title', 'Section')}\n")
            md.append(f"{section.get('content', '')}\n")
        
        if self.citations:
            md.append("\n## References\n")
            for i, cite in enumerate(self.citations, 1):
                cite.ref_id = str(i)
                md.append(f"{i}. {cite.to_apa()}\n")
        
        return "\n".join(md)


class ResearchMemoryEntry(BaseModel):
    """Entry in research memory for learning."""
    query: str
    query_embedding: list[float] = Field(default_factory=list)
    findings_summary: str
    key_facts: list[str] = Field(default_factory=list)
    source_count: int = 0
    avg_reliability: float = 0.0
    created_at: datetime = Field(default_factory=datetime.now)
    accessed_count: int = 0
    last_accessed: Optional[datetime] = None
