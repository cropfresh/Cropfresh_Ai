"""
Citation Manager
================
Manages academic-style citations and references.

Features:
- APA/MLA formatting
- Inline citation markers
- Bibliography generation
- Deduplication
"""

from datetime import date
from typing import Optional

from loguru import logger

from src.agents.research.models import Citation, Finding, SourceType


class CitationManager:
    """
    Manages citations for research reports.
    
    Usage:
        manager = CitationManager()
        
        # Add findings with citations
        manager.add_finding(finding1)
        manager.add_finding(finding2)
        
        # Get formatted bibliography
        bibliography = manager.get_bibliography(style="apa")
        
        # Get text with inline citations
        cited_text = manager.insert_citations(text, findings)
    """
    
    def __init__(self, start_ref_id: int = 1):
        """
        Initialize citation manager.
        
        Args:
            start_ref_id: Starting reference number
        """
        self._citations: dict[str, Citation] = {}  # url/id -> Citation
        self._ref_counter = start_ref_id
    
    def add_finding(self, finding: Finding) -> str:
        """
        Add a finding and return its citation reference.
        
        Args:
            finding: Finding with citation info
            
        Returns:
            Reference ID (e.g., "[1]")
        """
        # Generate unique key
        key = finding.source_url or f"{finding.source_type.value}_{finding.source_title}"
        
        # Check for duplicate
        if key in self._citations:
            return self._citations[key].ref_id
        
        # Create citation
        citation = finding.citation or Citation(
            title=finding.source_title or "Unknown Source",
            url=finding.source_url,
            source_type=finding.source_type,
        )
        
        # Assign reference ID
        citation.ref_id = str(self._ref_counter)
        self._ref_counter += 1
        
        # Store
        self._citations[key] = citation
        
        # Update finding
        finding.citation = citation
        
        return f"[{citation.ref_id}]"
    
    def add_citation(self, citation: Citation) -> str:
        """Add a citation directly."""
        key = citation.url or f"{citation.source_type.value}_{citation.title}"
        
        if key in self._citations:
            return f"[{self._citations[key].ref_id}]"
        
        citation.ref_id = str(self._ref_counter)
        self._ref_counter += 1
        self._citations[key] = citation
        
        return f"[{citation.ref_id}]"
    
    def get_citation(self, ref_id: str) -> Optional[Citation]:
        """Get citation by reference ID."""
        for citation in self._citations.values():
            if citation.ref_id == ref_id:
                return citation
        return None
    
    def get_all_citations(self) -> list[Citation]:
        """Get all citations sorted by reference ID."""
        return sorted(
            self._citations.values(),
            key=lambda c: int(c.ref_id) if c.ref_id.isdigit() else 999,
        )
    
    def get_bibliography(self, style: str = "apa") -> str:
        """
        Generate formatted bibliography.
        
        Args:
            style: Citation style ("apa", "mla", "simple")
            
        Returns:
            Formatted bibliography string
        """
        citations = self.get_all_citations()
        
        if not citations:
            return ""
        
        lines = ["## References\n"]
        
        for citation in citations:
            if style == "apa":
                lines.append(f"[{citation.ref_id}] {citation.to_apa()}")
            elif style == "mla":
                lines.append(f"[{citation.ref_id}] {self._format_mla(citation)}")
            else:
                lines.append(f"[{citation.ref_id}] {citation.title}. {citation.url or ''}")
        
        return "\n".join(lines)
    
    def insert_citations(
        self,
        text: str,
        findings: list[Finding],
    ) -> str:
        """
        Insert inline citations into text based on findings used.
        
        Args:
            text: Text to add citations to
            findings: Findings used in the text
            
        Returns:
            Text with inline citations
        """
        # Add citations for each finding
        citation_refs = []
        for finding in findings:
            if finding.citation and finding.citation.ref_id:
                citation_refs.append(f"[{finding.citation.ref_id}]")
            else:
                ref = self.add_finding(finding)
                citation_refs.append(ref)
        
        # Append citations at the end of text
        if citation_refs:
            unique_refs = sorted(set(citation_refs), key=lambda x: int(x.strip('[]')))
            text = f"{text} {' '.join(unique_refs)}"
        
        return text
    
    def _format_mla(self, citation: Citation) -> str:
        """Format as MLA citation."""
        parts = []
        if citation.author:
            parts.append(f"{citation.author}.")
        parts.append(f'"{citation.title}."')
        if citation.publisher:
            parts.append(f"{citation.publisher},")
        if citation.date_published:
            parts.append(f"{citation.date_published.strftime('%d %b %Y')}.")
        if citation.url:
            parts.append(f"Web. {citation.date_accessed.strftime('%d %b %Y')}.")
        return " ".join(parts)
    
    def clear(self):
        """Clear all citations."""
        self._citations.clear()
        self._ref_counter = 1
    
    @property
    def count(self) -> int:
        """Number of citations."""
        return len(self._citations)
